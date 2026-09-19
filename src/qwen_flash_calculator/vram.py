"""Read how much video memory is in use, live.

Two rules shaped this module: no `nvidia-smi` process, and no
administrator rights. Both are met.

**First way: NVML.** `nvml.dll` is NVIDIA's own library. The driver
installs it into `C:\\Windows\\System32`, `nvidia-smi` is only a program
that calls it, and any user can load it. Measured on the reference
machine on 2026-09-18: loading the library, reading the card and
unloading again took 30 ms and reported 2,108 MiB in use of 16,376.

**Second way: a Windows performance counter.** When NVML is missing, for
example on a card of another make, `\\GPU Adapter Memory(*)\\Dedicated
Usage` answers for any adapter. It also needs no rights. Measured the
same day: 1,759 MiB, and it took 1.3 s, which is why it is the fallback
and why every reading happens on a worker thread.

The two numbers differ because they count different things. NVML reports
what the driver holds on the card; the counter sums the dedicated
allocations Windows knows about. A reading therefore carries the name of
the source that produced it, and the window shows that name.

**Who holds it.** NVML cannot say. Under the Windows display driver
model it answers "not supported" for every process's memory: measured,
36 processes listed and not one figure among them. Windows itself does
know, because Task Manager shows the column, and it comes from a third
counter set, `\\GPU Process Memory(*)\\Dedicated Usage`, whose instance
names carry the process id. Measured 2026-09-19: 15 processes above 10
MiB, 1.3 s for the call, no administrator rights. That is where the
breakdown in a reading comes from.

The two sources count differently, so their totals differ by a little.
The window keeps NVML's total, because that is what the card really
holds, and lets the rest of it fall into one block beside the named
processes.

NVML is still asked which processes are on the card, because the names
come through even when the sizes do not. When llama-server is one of
them, the figure already contains the model, and subtracting it again
would count the same memory twice.
"""

import ctypes
import dataclasses
import subprocess
import time

#: Where the driver puts NVML. The bare name is tried too, in case the
#: library is on the search path but not in the system folder.
NVML_CANDIDATES = (
    r"C:\Windows\System32\nvml.dll",
    "nvml.dll",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
)

#: The process this calculator is about. When it appears among the
#: processes on the card, the reading says so.
SERVER_PROCESS = "llama-server"

#: NVML returns this for a process whose memory it cannot report.
_NOT_SUPPORTED = (1 << 64) - 1

_MIB = 1024 * 1024

#: Hide the console window a PowerShell fallback would otherwise flash.
_NO_WINDOW = 0x08000000


@dataclasses.dataclass(frozen=True)
class Reading:
    """One measurement of the card, with its provenance."""

    used_mib: int
    total_mib: int | None
    source: str
    server_on_card: bool
    #: Largest first, ``(process name, MiB)``. Empty when the per-process
    #: counter could not be read; the window then shows one block instead
    #: of a breakdown.
    processes: tuple[tuple[str, int], ...] = ()
    taken_at: float = dataclasses.field(default_factory=time.time)

    @property
    def free_mib(self) -> int | None:
        if self.total_mib is None:
            return None
        return max(0, self.total_mib - self.used_mib)


class Unavailable(Exception):
    """This way of reading the card did not work."""


# ---------------------------------------------------------------------------
# NVML
# ---------------------------------------------------------------------------


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Process(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


def _load_nvml():
    for candidate in NVML_CANDIDATES:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise Unavailable("nvml.dll not found")


def _server_is_on_card(lib, handle) -> bool:
    """True when a process named like the server holds the card.

    Only the names are read. The memory figures beside them are not
    available under Windows, and this function does not pretend otherwise.
    """
    for entry in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
    ):
        listing = getattr(lib, entry, None)
        if listing is None:
            continue
        count = ctypes.c_uint(0)
        listing(handle, ctypes.byref(count), None)
        if count.value == 0:
            return False
        processes = (_Process * count.value)()
        if listing(handle, ctypes.byref(count), processes) != 0:
            return False
        name = ctypes.create_string_buffer(512)
        for process in processes[: count.value]:
            if lib.nvmlSystemGetProcessName(process.pid, name, 512) != 0:
                continue
            if SERVER_PROCESS in name.value.decode("utf-8", "replace").lower():
                return True
        return False
    return False


def read_nvml() -> Reading:
    lib = _load_nvml()
    if lib.nvmlInit_v2() != 0:
        raise Unavailable("nvmlInit failed")
    try:
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            raise Unavailable("no device 0")
        memory = _Memory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
            raise Unavailable("memory query failed")
        return Reading(
            used_mib=int(memory.used // _MIB),
            total_mib=int(memory.total // _MIB),
            source="NVML",
            server_on_card=_server_is_on_card(lib, handle),
        )
    finally:
        lib.nvmlShutdown()


# ---------------------------------------------------------------------------
# The Windows performance counter
# ---------------------------------------------------------------------------

_COUNTER_COMMAND = (
    "$c = Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' "
    "-ErrorAction Stop; "
    "[int](($c.CounterSamples | Measure-Object CookedValue -Sum).Sum / 1MB)"
)


def read_counter(runner=None) -> Reading:
    """Sum the dedicated video memory Windows reports for every adapter.

    This one knows nothing about the total size of the card, so the
    reading carries no total and the window keeps using the size in the
    field above it.
    """
    run = runner or _run_powershell
    text = run(_COUNTER_COMMAND)
    line = text.strip().splitlines()[-1].strip() if text.strip() else ""
    if not line.isdigit():
        raise Unavailable(f"counter returned {line[:40]!r}")
    return Reading(
        used_mib=int(line),
        total_mib=None,
        source="Windows GPU counter",
        server_on_card=False,
    )


def _run_powershell(command: str) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=_NO_WINDOW,
        check=False,
    )
    if completed.returncode != 0:
        raise Unavailable((completed.stderr or "counter failed").strip()[:80])
    return completed.stdout


# ---------------------------------------------------------------------------
# Who holds the memory
# ---------------------------------------------------------------------------

#: The counter set behind the "Dedicated GPU memory" column in Task
#: Manager. It is the only way on Windows to see the memory per process:
#: NVML answers "not supported" here, because under this driver model
#: Windows, not the card's driver, owns the allocation.
#:
#: Measured 2026-09-19 on the reference machine: 15 processes above 10
#: MiB, 1,973 MiB in total, 1.3 s for the call, no administrator rights.
_PROCESS_COMMAND = r"""
$c = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop
foreach ($s in $c.CounterSamples) {
  if ($s.CookedValue -lt 1048576) { continue }
  $id = 0
  if ($s.InstanceName -match 'pid_(\d+)') { $id = [int]$Matches[1] }
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  if ($p) { $n = $p.ProcessName } else { $n = "pid $id" }
  "$n|" + [int]($s.CookedValue / 1048576)
}
"""


def read_processes(runner=None) -> tuple[tuple[str, int], ...]:
    """Every process holding video memory, largest first.

    One process can appear once per adapter, so the rows are added up by
    name rather than listed twice.
    """
    run = runner or _run_powershell
    totals: dict[str, int] = {}
    for line in run(_PROCESS_COMMAND).splitlines():
        name, sep, mib = line.strip().rpartition("|")
        if not sep or not mib.isdigit() or not name:
            continue
        totals[name] = totals.get(name, 0) + int(mib)
    return tuple(sorted(totals.items(), key=lambda pair: -pair[1]))


# ---------------------------------------------------------------------------
# What the window calls
# ---------------------------------------------------------------------------


def read(ways=None, processes=None) -> Reading | tuple[None, str]:
    """The first way that answers, or ``(None, reason)`` when none does.

    The reason is shown to the person rather than swallowed: a blank
    block beside a ticked box reads as a broken calculator.
    """
    reasons = []
    for way in ways or (read_nvml, read_counter):
        try:
            reading = way()
        except Unavailable as error:
            reasons.append(str(error))
            continue
        except Exception as error:  # noqa: BLE001 - a driver can fail in its own ways
            reasons.append(f"{type(error).__name__}: {error}")
            continue
        # The total comes from the fastest source that works; the
        # breakdown always comes from the Windows counter, because
        # nothing else on this system has it. A failure here is not worth
        # losing the total over, so the breakdown is simply left empty.
        try:
            return dataclasses.replace(
                reading, processes=(processes or read_processes)()
            )
        except Exception:  # noqa: BLE001 - falling back to the reading without a breakdown
            return reading
    return None, "; ".join(reasons) or "no way to read the card"


# ---------------------------------------------------------------------------
# The machine this is running on
# ---------------------------------------------------------------------------


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def system_memory_gb():
    """Physical memory installed in this machine, whole GB, or None.

    Asked of Windows directly, which costs nothing worth measuring. The
    answer is rounded: a 96 GB machine reports 95.9 GiB, and a field that
    said 95.9 would invite the reader to wonder where the rest went.
    """
    try:
        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.total_physical / (1024**3))
    except Exception:  # noqa: BLE001 - ctypes into Windows can fail in its own ways
        return None


def list_gpus():
    """Every NVIDIA card NVML can see, as ``(name, total MiB)``.

    Empty when the library is missing, which is the normal answer on a
    machine with a card of another make. The window then offers its
    plain sizes instead and says so.
    """
    try:
        lib = _load_nvml()
    except Unavailable:
        return ()
    if lib.nvmlInit_v2() != 0:
        return ()
    try:
        count = ctypes.c_uint(0)
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return ()
        found = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            name = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(handle, name, 96) != 0:
                continue
            memory = _Memory()
            if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                continue
            found.append(
                (name.value.decode("utf-8", "replace"), int(memory.total // _MIB))
            )
        return tuple(found)
    finally:
        lib.nvmlShutdown()
