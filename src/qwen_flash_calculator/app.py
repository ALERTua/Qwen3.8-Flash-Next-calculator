"""Desktop calculator for llama.cpp speed and memory on Qwen3.8-Flash-Next.

Run with:  uv run qwen-flash-calculator

Standard library only: tkinter and ttk. Every control recalculates the whole
estimate on the spot, including while a slider is being dragged.
"""

import json
import os
import queue
import threading
import time
import tkinter as tk
import typing
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

from . import estimator as E
from . import measured as M
from . import vram as card

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

#: How often the card is read while the box is ticked, seconds.
VRAM_POLL_SECONDS = 15

#: How many processes are named in the bar before the rest are added up.
BUSY_PROCESS_COUNT = 5

#: Greys for the processes holding the card, darkest first, so the whole
#: busy part reads as one block from a distance and still separates into
#: its owners up close. They are deliberately colourless: the coloured
#: segments are the ones this calculator can do something about.
BUSY_COLOURS = ("#3C3C3C", "#535353", "#6B6B6B", "#848484", "#9C9C9C")
BUSY_REST_COLOUR = "#B4B4B4"

#: Grey belongs to the processes above, which this calculator cannot
#: move. Everything it can move carries a colour, so the brown margin and
#: the grey free block were reading as part of that same dead family.
COLOUR = {
    "busy": "#4A4A4A",
    "dense": "#4C78A8",
    "experts": "#59A14F",
    "kv": "#F28E2B",
    "compute": "#E15759",
    "vision": "#B07AA1",
    "margin": "#EDC948",
    "free": "#7FD4C1",
    "overflow": "#B02020",
}

#: The fixed segments of the bar, in the words the legend uses.
SEGMENT_NAMES = {
    "busy": "Other programs",
    "dense": "Dense and attention",
    "experts": "Expert layers",
    "kv": "Key-value cache",
    "compute": "Compute buffers",
    "vision": "Vision projector",
    "margin": "Fit margin",
    "free": "Free VRAM",
    "overflow": "Asked for beyond the card",
}

STATUS_COLOUR = {
    E.OK: "#1B7F3B",
    E.DEGRADED: "#B06000",
    E.FAIL: "#B02020",
}

#: The three states, in the same words the estimator uses. Nothing else is
#: ever shown: a run that loads with layers on the processor is a spill, not
#: a failure to fit.
STATUS_WORD = {
    E.OK: "FITS",
    E.DEGRADED: "SPILLS TO THE PROCESSOR",
    E.FAIL: "DOES NOT START",
}

#: The window's own title. It carries the two things that used to sit in
#: bold and in grey inside the window: what this is, and which file it
#: was fitted to. Neither needs a line of height on every screen.
WINDOW_TITLE = (
    f"{M.MODEL_NAME} on llama.cpp - fit and speed calculator - "
    f"{M.MODEL_QUANT}, {M.BLOCK_COUNT} blocks, "
    f"{M.NATIVE_CONTEXT:,} tokens, mixture of experts"
)

#: The legend hugs the left edge in as many columns as fit, and a column
#: is never started for fewer rows than this. Spread thinner, thirteen
#: entries became a row of stubs across the whole window.
MIN_LEGEND_ROWS = 4
LEGEND_ROW_HEIGHT = 18

#: Room for the colour patch in front of a legend label, and for the gap
#: after it before the next column starts.
LEGEND_PATCH_WIDTH = 12
LEGEND_TEXT_INSET = 18
LEGEND_COLUMN_GAP = 24

#: The widest the parameter column may grow, pixels, and the width its
#: text wraps at inside that. Without a ceiling the column grew to the
#: longest caption in it, about 900 pixels, and took that much room
#: away from the results beside it.
PARAMETER_COLUMN_WIDTH = 600
PARAMETER_TEXT_WRAP = 560

#: No label wraps narrower than this, however small the window gets.
MINIMUM_WRAP = 240

#: The tallest the launcher panel may grow, pixels.
LAUNCHER_MAX_HEIGHT = 800

#: Where the three controls stand when the window opens. They are the
#: interface's own starting points, not the estimator's defaults:
#: DEFAULT_FIT_MARGIN_MIB stays at 1,024 because the measured grid and
#: the tests over it are anchored there, while 512 is the margin this
#: machine actually runs and the one the main grid was measured with.
STARTING_FIT_MARGIN_MIB = 512
STARTING_SPECULATION = True
STARTING_SHOW_VRAM = True

#: How much of the text is text the model has already seen, in percent.
#: Zero, because that is what the speed table behind this calculator was
#: fitted on. Measured on this machine on 2026-09-19: on text the model
#: has not seen, speculation returns nothing, and a prompt of fresh word
#: salad generated at 17.7 tokens per second while this calculator,
#: standing at 5 percent, promised 24.5. Raise it to ask what repeated
#: text would do, not to describe a measurement.
STARTING_REPEAT_PERCENT = 0

#: What the controls held when the window last closed. It sits with the
#: settings of other programs, not beside this code, because the code can
#: be run from a folder that is thrown away after the run.
SETTINGS_FILE = (
    Path(os.environ.get("APPDATA") or Path.home()) / "Qwen3.8-Flash-calculator.json"
)

#: The button beside every control that puts its own default back.
RESET_MARK = "↺"

#: The dropdown entry for a card this machine cannot name on its own.
#: Picking it is what unlocks the memory field for typing.
CUSTOM_GPU_LABEL = "Custom"

BAR_HEIGHT = 40
BAR_WIDTH = 620

#: The bar segments add up to the card's capacity within a rounding error,
#: so the "has the bar already run past the capacity line" test needs a
#: tolerance. Without one the overflow tail disappeared exactly when the
#: configuration overflowed, and the legend and the bar disagreed.
BAR_TOLERANCE_MIB = 0.5

#: Which cache type costs what, said on the control itself.
KV_LABEL = {
    "q8_0": "8-bit q8_0, half the memory",
    "f16": "16-bit f16",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def nearest_index(values: list[int], wanted: int) -> int:
    best = 0
    for i, value in enumerate(values):
        if abs(value - wanted) < abs(values[best] - wanted):
            best = i
    return best


def read_number(variable: tk.StringVar, fallback: float, low: float, high: float):
    """Read a typed-in number. Bad text keeps the last good value."""
    text = variable.get().strip().replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return fallback, False
    if value < low or value > high:
        return value, False
    return value, True


def forget_settings() -> None:
    """Delete the settings file. Failing to delete it is not worth an error."""
    try:
        SETTINGS_FILE.unlink()
    except OSError:
        pass


def load_settings() -> dict:
    """What the controls held last time, or nothing when it cannot be read.

    A file that does not read back is deleted here, and the window opens on
    its defaults. Half the controls arriving from an old file and half from
    the defaults is a state nobody can explain later.
    """
    if not SETTINGS_FILE.exists():
        return {}
    try:
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        forget_settings()
        return {}
    if not isinstance(stored, dict):
        forget_settings()
        return {}
    return stored


def save_settings(values: dict) -> None:
    """Remember the controls. Failing to do so is not worth an error."""
    try:
        SETTINGS_FILE.write_text(
            json.dumps(values, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
    except OSError:
        pass


def legend_columns(count: int, available: float, column_width: float) -> list[int]:
    """How many rows go in each legend column, left to right.

    No column is started for fewer than MIN_LEGEND_ROWS rows, so a short
    legend stays one tall column at the left edge instead of a row of
    stubs across the window. What does fit is spread evenly, so the last
    column is never the only short one.
    """
    if count <= 0:
        return []
    fits = max(1, int(available // column_width)) if column_width > 0 else 1
    columns = max(1, min(fits, count // MIN_LEGEND_ROWS))
    base, extra = divmod(count, columns)
    return [base + (1 if index < extra else 0) for index in range(columns)]


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


class Calculator(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=10)
        self.ready = False

        self.last_vram = float(M.REFERENCE_VRAM_MIB)
        self.last_ram = M.REFERENCE_RAM_GB

        # The live card reading. The worker thread never touches a widget:
        # it puts its answer in the queue and the window's own loop picks
        # it up, because tkinter may only be called from the thread that
        # created it.
        self._results_labels: list[ttk.Label] = []
        self._window_labels: list[ttk.Label] = []
        self._results_wrap = BAR_WIDTH
        self._window_wrap = 980
        self.body_canvas: tk.Canvas | None = None
        self._scrolling_canvases: list[tk.Canvas] = []
        self._gpu_options: list[tuple[str, int]] | None = None
        self._machine_ram_gb = card.system_memory_gb()
        self._last_drawn: tuple[E.Config, E.Estimate] | None = None
        self._busy_labels: dict[str, str] = {}
        self.vram_reading: card.Reading | None = None
        self.vram_error = ""
        self._vram_queue: queue.Queue[object] = queue.Queue()
        self._vram_job: str | None = None
        self._vram_running = False
        self._vram_due = 0.0
        # The legend sizes its columns to the longest label in them, and
        # a process name can be any length, so the width is measured
        # rather than guessed.
        self._legend_font = tkfont.Font(family="Segoe UI", size=9)

        self.winfo_toplevel().title(WINDOW_TITLE)
        self._make_variables()

        # The bar first, at the top of the window and the width of it: it
        # is the answer the eye goes to, and nothing above it may move it.
        # What the numbers are worth is said at the foot of the parameter
        # column instead, where it does not push the bar about.
        self._build_memory_row(self)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self._build_inputs(ttk.Frame(self._scrolling_column(body)))
        results = self._scrolling_column(body, fixed_width=False)
        self._build_outputs(ttk.Frame(results))
        results.master.bind("<Configure>", self._results_resized)
        self.bind("<Configure>", self._window_resized)

        self._load()
        self.ready = True
        # The box may start ticked, and the command on a checkbutton runs
        # only when a person clicks it, so the first reading has to be
        # started here. It recalculates on the way through.
        self._toggle_vram_poll()

    def _scrolling_column(
        self, parent: ttk.Frame, fixed_width: bool = True
    ) -> ttk.Frame:
        """One column, in a frame that scrolls on its own.

        Measured on this machine: this column asks for about 900 pixels of
        height, more than a laptop screen leaves once the bar above it and
        the launcher below it have taken theirs. Only this column scrolls.
        The results beside it stay where they are, so a slider and the
        number it changes are never scrolled apart.
        """
        holder = ttk.Frame(parent)
        if fixed_width:
            holder.pack(side="left", fill="y", padx=(0, 12))
        else:
            holder.pack(side="left", fill="both", expand=True)

        canvas = tk.Canvas(holder, highlightthickness=0, borderwidth=0)
        down = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=down.set)
        canvas.pack(side="left", fill="both", expand=True)
        down.pack(side="right", fill="y")

        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _inner_changed(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            if fixed_width:
                # The parameters keep the width their widgets ask for, up
                # to a ceiling. Their text wraps at that ceiling, so
                # nothing is cut off when it bites.
                canvas.configure(
                    width=min(inner.winfo_reqwidth(), PARAMETER_COLUMN_WIDTH)
                )

        def _canvas_changed(event: tk.Event) -> None:
            if not fixed_width:
                # The results take whatever width is left, exactly as they
                # did before they were put in a canvas.
                canvas.itemconfigure(window, width=event.width)

        inner.bind("<Configure>", _inner_changed)
        canvas.bind("<Configure>", _canvas_changed)

        # The wheel has to work over the labels and sliders, not only over
        # the canvas itself, and a label does not forward the event to its
        # parent. So the binding is global while the pointer is inside
        # this area, and taken away again when it leaves, which keeps the
        # wheel over the launcher text belonging to the launcher text.
        canvas.bind(
            "<Enter>",
            lambda _e, c=canvas: c.bind_all(
                "<MouseWheel>", lambda event, canvas=c: self._wheel(event, canvas)
            ),
        )
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        if self.body_canvas is None:
            # The first one built is the parameters. The wheel handler and
            # the tests speak about that one.
            self.body_canvas = canvas
        self._scrolling_canvases.append(canvas)
        return inner

    def _wheel(self, event: tk.Event, canvas=None) -> None:
        (canvas or self.body_canvas).yview_scroll(-int(event.delta / 120), "units")

    # -- variables ---------------------------------------------------------

    #: Where every control stands when nothing has been remembered. One
    #: table, so the starting value, the button beside the control and
    #: the button that puts them all back cannot say different things.
    DEFAULTS: typing.ClassVar[dict[str, float | str | bool]] = {
        "ctx_index": float(nearest_index(M.CTX_CHOICES, 163840)),
        "ubatch_index": float(M.UBATCH_CHOICES.index(512)),
        "kv_type": "q8_0",
        "threads": 10.0,
        "fit_margin": float(STARTING_FIT_MARGIN_MIB),
        "vision": False,
        "speculation": STARTING_SPECULATION,
        "repetitiveness": float(STARTING_REPEAT_PERCENT),
        "show_vram": STARTING_SHOW_VRAM,
    }

    def _make_variables(self) -> None:
        # Every scale-backed value is a DoubleVar on purpose. ttk.Scale writes
        # a floating point number into whatever variable it is linked to, so an
        # IntVar here would raise as soon as the slider landed between stops.
        # The snapping happens when the value is read, in current_config().
        self.ctx_index = tk.DoubleVar(value=self.DEFAULTS["ctx_index"])
        self.ubatch_index = tk.DoubleVar(value=self.DEFAULTS["ubatch_index"])
        self.kv_type = tk.StringVar(value=self.DEFAULTS["kv_type"])
        self.threads = tk.DoubleVar(value=self.DEFAULTS["threads"])
        self.fit_margin = tk.DoubleVar(value=self.DEFAULTS["fit_margin"])
        self.vision = tk.BooleanVar(value=self.DEFAULTS["vision"])
        self.speculation = tk.BooleanVar(value=self.DEFAULTS["speculation"])
        self.repetitiveness = tk.DoubleVar(value=self.DEFAULTS["repetitiveness"])
        self.gpu_text = tk.StringVar(value="")
        self.show_vram = tk.BooleanVar(value=self.DEFAULTS["show_vram"])
        self.vram_text = tk.StringVar(value=str(M.REFERENCE_VRAM_MIB))
        self.ram_text = tk.StringVar(
            value=f"{self._machine_ram_gb or M.REFERENCE_RAM_GB:g}"
        )

        for variable in (
            self.ctx_index,
            self.ubatch_index,
            self.kv_type,
            self.threads,
            self.fit_margin,
            self.vision,
            self.speculation,
            self.repetitiveness,
            self.show_vram,
            self.vram_text,
            self.ram_text,
        ):
            variable.trace_add("write", self._on_change)

    def _on_change(self, *_args) -> None:
        if self.ready:
            self.recalculate()

    # -- remembering the controls -----------------------------------------

    def variables(self) -> dict:
        """Every control that is worth remembering, by name."""
        named = {name: getattr(self, name) for name in self.DEFAULTS}
        named["gpu_text"] = self.gpu_text
        named["vram_text"] = self.vram_text
        return named

    def save(self) -> None:
        """Write the controls down. Called when the window closes."""
        save_settings(
            {name: variable.get() for name, variable in self.variables().items()}
        )

    def _load(self) -> None:
        """Put back what was remembered, or start clean when it is broken.

        A value of the wrong kind condemns the whole file: the file goes,
        and every control opens on its default. A key this version does
        not know is not a fault, it is an older or newer file, and the
        window passes over it and keeps the rest.
        """
        stored = load_settings()
        wanted = self.variables()
        broken = [
            name
            for name in wanted
            if name in stored and not self._fits(stored[name], name)
        ]
        if broken:
            forget_settings()
            stored = {}
        for name, variable in wanted.items():
            if name in stored:
                variable.set(stored[name])
        if self.gpu_text.get() not in [label for label, _mib in self._gpu_choices()]:
            self._select_first_gpu()
        self._on_gpu_chosen()

    def _fits(self, value, name: str) -> bool:
        """Whether this remembered value may be put into that control.

        Checked before it is set, not after. A slider holding the word
        "nonsense" raises on every later read, from inside a trace where
        nothing can catch it, and the window goes on printing the same
        error for the rest of its life.
        """
        wanted = self.DEFAULTS.get(name, "")
        if isinstance(wanted, bool):
            return isinstance(value, bool)
        if isinstance(wanted, float):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, type(wanted))

    def _reapply(self, name: str) -> None:
        """Run the work a control switches, after its value was written.

        A variable with a trace on it recalculates the estimate by itself.
        Two controls do not. A checkbutton runs its command only when a
        person clicks it, so a reset that only wrote the variable moved
        the tick and left the reading it switches exactly as it was. Any
        control whose work lives outside the trace belongs here.
        """
        if name == "show_vram":
            self._toggle_vram_poll()
        elif name in ("gpu_text", "vram_text"):
            self._on_gpu_chosen()

    def reset(self, name: str) -> None:
        """One control back to its default, and its work put back with it."""
        if name in ("gpu_text", "vram_text"):
            self._select_first_gpu()
            self._reapply(name)
            return
        self.variables()[name].set(self.DEFAULTS[name])
        self._reapply(name)

    def reset_all(self) -> None:
        """Every control back to its default, each one re-applied as it goes."""
        for name in self.variables():
            self.reset(name)

    def _reset_button(self, parent, row: int, column: int, name: str):
        """The small button that puts one control back."""
        button = ttk.Button(
            parent,
            text=RESET_MARK,
            width=3,
            command=lambda n=name: self.reset(n),
        )
        button.grid(row=row, column=column, sticky="e", padx=(6, 0))
        return button

    # -- inputs ------------------------------------------------------------

    def _build_trust(self, frame: ttk.Frame) -> None:
        """How far these numbers can be trusted, at the foot of the column.

        It reads last because it qualifies everything above it, and it
        sits here rather than under the bar because its height changes
        with the width of the window, and up there every resize moved the
        bar down the screen.
        """
        trust = ttk.Frame(frame)
        trust.pack(fill="x", pady=(12, 0))
        self.disclaimer_label = ttk.Label(
            trust,
            text=M.DISCLAIMER,
            foreground="#8A5A00",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.disclaimer_label.pack(anchor="w")
        worst_pp, worst_tg = E.worst_validation_error_pct()
        self.validation_label = ttk.Label(
            trust,
            text=f"Checked against five directly measured configurations "
            f"(README.md has the table): the estimate is within "
            f"{worst_pp:.0f}% on prefill and within {worst_tg:.0f}% on "
            f"generation across the five.",
            foreground="#8A5A00",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.validation_label.pack(anchor="w", pady=(6, 0))

    def _build_inputs(self, frame: ttk.Frame) -> None:
        frame.pack(side="left", fill="y", padx=(0, 12))

        ttk.Button(
            frame,
            text=f"{RESET_MARK} Reset every setting",
            command=self.reset_all,
        ).pack(anchor="w", pady=(0, 6))

        run = ttk.LabelFrame(frame, text="Run", padding=8)
        run.pack(fill="x")

        self.ctx_caption = ttk.Label(
            run,
            text="",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.ctx_caption.grid(row=0, column=0, sticky="w")
        ttk.Scale(
            run,
            from_=0,
            to=len(M.CTX_CHOICES) - 1,
            orient="horizontal",
            length=300,
            variable=self.ctx_index,
        ).grid(row=1, column=0, sticky="ew")
        self.ctx_limit = ttk.Label(run, text="", foreground="#777777")
        self.ctx_limit.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.ubatch_caption = ttk.Label(
            run,
            text="",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.ubatch_caption.grid(row=3, column=0, sticky="w")
        ttk.Scale(
            run,
            from_=0,
            to=len(M.UBATCH_CHOICES) - 1,
            orient="horizontal",
            length=300,
            variable=self.ubatch_index,
        ).grid(row=4, column=0, sticky="ew")
        self.ubatch_limit = ttk.Label(run, text="", foreground="#777777")
        self.ubatch_limit.grid(row=5, column=0, sticky="w", pady=(0, 8))

        cache = ttk.Frame(run)
        cache.grid(row=6, column=0, sticky="w", pady=(0, 8))
        ttk.Label(cache, text="Key-value cache:").pack(side="left")
        for name in M.KV_TYPES:
            ttk.Radiobutton(
                cache, text=KV_LABEL[name], value=name, variable=self.kv_type
            ).pack(side="left", padx=(8, 0))

        self.threads_caption = ttk.Label(
            run,
            text="",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.threads_caption.grid(row=7, column=0, sticky="w")
        ttk.Scale(
            run,
            from_=M.THREAD_MIN,
            to=M.THREAD_MAX,
            orient="horizontal",
            length=300,
            variable=self.threads,
        ).grid(row=8, column=0, sticky="ew")
        ttk.Label(
            run,
            text=M.THREAD_DEPTH_CAVEAT,
            foreground="#777777",
            wraplength=300,
            justify="left",
        ).grid(row=9, column=0, sticky="w", pady=(0, 8))

        self.margin_caption = ttk.Label(
            run,
            text="",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.margin_caption.grid(row=10, column=0, sticky="w")
        ttk.Scale(
            run,
            from_=M.FIT_MARGIN_MIN,
            to=M.FIT_MARGIN_MAX,
            orient="horizontal",
            length=300,
            variable=self.fit_margin,
        ).grid(row=11, column=0, sticky="ew", pady=(0, 8))

        # The checkbox and the advice line say the same words, from the
        # same constant, so the two can never drift apart.
        ttk.Checkbutton(
            run,
            text=f"Switch {E.IMAGE_INPUT_LABEL} on "
            f"({M.VISION_PROJECTOR_MIB:,} MiB of video memory)",
            variable=self.vision,
        ).grid(row=12, column=0, sticky="w")
        ttk.Checkbutton(
            run,
            text="Speed-up on repeated text "
            "(n-gram speculation, --spec-type ngram-mod)",
            variable=self.speculation,
        ).grid(row=13, column=0, sticky="w")
        ttk.Label(
            run,
            text="Speculation builds its pool out of the text of the "
            "conversation and costs no video memory. It is not the "
            f"{M.NGRAM_TABLE_GB:g} GB n-gram table inside the model file; "
            "the two only share a word.",
            foreground="#777777",
            wraplength=300,
            justify="left",
        ).grid(row=14, column=0, sticky="w", pady=(0, 4))

        self.repetition_caption = ttk.Label(
            run,
            text="",
            wraplength=PARAMETER_TEXT_WRAP,
            justify="left",
        )
        self.repetition_caption.grid(row=15, column=0, sticky="w")
        self.repetition_scale = ttk.Scale(
            run,
            from_=0,
            to=100,
            orient="horizontal",
            length=300,
            variable=self.repetitiveness,
        )
        self.repetition_scale.grid(row=16, column=0, sticky="ew")

        machine = ttk.LabelFrame(frame, text="Machine", padding=8)
        machine.pack(fill="x", pady=(10, 0))

        ttk.Label(machine, text="Graphics card:").grid(row=0, column=0, sticky="w")
        # A frame of its own, so the name and its memory are two controls
        # side by side but still one cell of the machine grid.
        gpu_row = ttk.Frame(machine)
        gpu_row.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.gpu_choice = ttk.Combobox(
            gpu_row,
            textvariable=self.gpu_text,
            values=[label for label, _mib in self._gpu_choices()],
            state="readonly",
            width=36,
        )
        self.gpu_choice.grid(row=0, column=0, sticky="w")
        self.gpu_choice.bind("<<ComboboxSelected>>", self._on_gpu_chosen)
        self.vram_entry = ttk.Entry(gpu_row, textvariable=self.vram_text, width=10)
        self.vram_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(gpu_row, text="MiB").grid(row=0, column=2, sticky="w", padx=(4, 0))
        self._select_first_gpu()
        # Lock or unlock the field to match whichever entry that picked.
        self._on_gpu_chosen()
        ttk.Label(
            machine,
            text=self._gpu_source_note(),
            foreground="#777777",
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        ttk.Checkbutton(
            machine,
            text="Show video memory in use",
            variable=self.show_vram,
            command=self._toggle_vram_poll,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self.vram_live_label = ttk.Label(machine, text="")
        self.vram_live_label.grid(
            row=3, column=0, columnspan=2, sticky="w", padx=(18, 0)
        )
        self.vram_live_note = ttk.Label(
            machine,
            text="",
            foreground="#777777",
            wraplength=320,
            justify="left",
        )
        self.vram_live_note.grid(
            row=4, column=0, columnspan=2, sticky="w", padx=(18, 0), pady=(0, 8)
        )

        # System memory is not a choice: it is whatever this machine has,
        # and Windows answers that in microseconds. A field to type it in
        # was one more thing to get wrong.
        ttk.Label(
            machine,
            text=self._ram_note(),
            wraplength=320,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w")

        self.machine_note = ttk.Label(
            frame, text="", foreground="#B02020", wraplength=330, justify="left"
        )
        self.machine_note.pack(anchor="w", pady=(6, 0))

        # One button per control, in a column of its own beside them, so
        # a setting can be put back without touching the others.
        run.columnconfigure(0, weight=1)
        machine.columnconfigure(1, weight=1)
        for row, name in (
            (0, "ctx_index"),
            (3, "ubatch_index"),
            (6, "kv_type"),
            (7, "threads"),
            (10, "fit_margin"),
            (12, "vision"),
            (13, "speculation"),
            (15, "repetitiveness"),
        ):
            self._reset_button(run, row, 1, name)
        self._reset_button(machine, 0, 2, "gpu_text")
        self._reset_button(machine, 2, 2, "show_vram")

        self._build_trust(frame)

    # -- the live card reading --------------------------------------------

    def _toggle_vram_poll(self) -> None:
        """Start or stop reading the card, and redraw either way."""
        if self.show_vram.get():
            self._vram_due = 0.0  # read now, not in fifteen seconds
            if self._vram_job is None:
                self._vram_tick()
        else:
            if self._vram_job is not None:
                self.after_cancel(self._vram_job)
                self._vram_job = None
            self.vram_reading = None
            self.vram_error = ""
        self._update_vram_block()
        self.recalculate()

    def _vram_tick(self) -> None:
        """One second of the poll, on the window's own thread.

        The tick is short and the reading is not: NVML answers in about
        30 ms but the fallback counter takes two seconds, and a window
        that freezes for two seconds every fifteen is worse than no
        reading at all. So the tick only starts a worker and collects
        what earlier workers left.
        """
        self._vram_job = None
        if not self.show_vram.get():
            return
        self._drain_vram_queue()
        if not self._vram_running and time.monotonic() >= self._vram_due:
            self._vram_running = True
            self._vram_due = time.monotonic() + VRAM_POLL_SECONDS
            threading.Thread(target=self._vram_worker, daemon=True).start()
        self._vram_job = self.after(1000, self._vram_tick)

    def _vram_worker(self) -> None:
        self._vram_queue.put(card.read())

    def _drain_vram_queue(self) -> None:
        answered = False
        while True:
            try:
                answer = self._vram_queue.get_nowait()
            except queue.Empty:
                break
            self._vram_running = False
            answered = True
            if isinstance(answer, card.Reading):
                self.vram_reading, self.vram_error = answer, ""
            else:
                self.vram_reading = None
                self.vram_error = (
                    answer[1] if isinstance(answer, tuple) else str(answer)
                )
        if answered:
            self._update_vram_block()
            self.recalculate()

    def _update_vram_block(self) -> None:
        if not self.show_vram.get():
            self.vram_live_label.configure(text="")
            self.vram_live_note.configure(text="", foreground="#777777")
            return

        if self.vram_error:
            self.vram_live_label.configure(text="The card could not be read.")
            self.vram_live_note.configure(
                text=f"{self.vram_error}  The estimates use the full card size above.",
                foreground="#B02020",
            )
            return

        reading = self.vram_reading
        if reading is None:
            self.vram_live_label.configure(text="Reading the card...")
            self.vram_live_note.configure(text="", foreground="#777777")
            return

        total = int(self.last_vram)
        left = max(0, total - reading.used_mib)
        self.vram_live_label.configure(
            text=f"In use now: {reading.used_mib:,} MiB   "
            f"Left for this run: {left:,} MiB"
        )
        age = max(0, int(time.time() - reading.taken_at))
        note = (
            f"Read from {reading.source} {age} s ago, again every "
            f"{VRAM_POLL_SECONDS} s. Every number below is for the "
            f"{left:,} MiB that are left, not for the whole card."
        )
        colour = "#777777"
        if reading.server_on_card:
            note += (
                "  llama-server is on the card, so this figure already "
                "contains the model. Stop the server before trusting it."
            )
            colour = "#B06000"
        self.vram_live_note.configure(text=note, foreground=colour)

    def destroy(self) -> None:
        self.unbind_all("<MouseWheel>")
        if self._vram_job is not None:
            self.after_cancel(self._vram_job)
            self._vram_job = None
        super().destroy()

    # -- what this machine has ---------------------------------------------

    def _gpu_choices(self) -> list[tuple[str, int]]:
        """The cards to offer: the real ones first, then a way to type any other.

        Custom stays last so a real card, when there is one, is always
        the first entry and the one already chosen when the window opens.
        """
        if self._gpu_options is None:
            options = [(name, mib) for name, mib in card.list_gpus()]
            options.append((CUSTOM_GPU_LABEL, 0))
            self._gpu_options = options
        return self._gpu_options

    def _gpu_source_note(self) -> str:
        found = [
            label for label, _mib in self._gpu_choices() if label != CUSTOM_GPU_LABEL
        ]
        if found:
            return (
                "Names are read from the card itself through NVML. Pick "
                "Custom to type a figure for a machine that is not this one."
            )
        return (
            "No NVIDIA card answered, so only Custom is offered. Type the "
            "figure for your card."
        )

    def _ram_note(self) -> str:
        if self._machine_ram_gb is None:
            return (
                f"System memory: {M.REFERENCE_RAM_GB:g} GB assumed. Windows "
                f"did not answer, so this is the reference machine's figure."
            )
        return f"System memory: {self._machine_ram_gb:g} GB, read from this machine."

    def _select_first_gpu(self) -> None:
        """Start on the card this machine has, not on an empty box.

        Real cards come first in the list, so the first entry is the
        right default. When none answered, the first and only entry is
        Custom, and the reference figure stands in until one is typed.
        """
        choices = self._gpu_choices()
        if not choices:
            return
        label, mib = choices[0]
        self.gpu_text.set(label)
        if label == CUSTOM_GPU_LABEL:
            self.vram_text.set(str(M.REFERENCE_VRAM_MIB))
        else:
            self.vram_text.set(str(mib))

    def _on_gpu_chosen(self, _event=None) -> None:
        # Runs during panel construction too, before vram_entry exists.
        entry = getattr(self, "vram_entry", None)
        label = self.gpu_text.get()
        if label == CUSTOM_GPU_LABEL:
            if entry is not None:
                entry.configure(state="normal")
            return
        for choice_label, mib in self._gpu_choices():
            if choice_label == label:
                self.vram_text.set(str(mib))
                break
        if entry is not None:
            entry.configure(state="readonly")

    # -- text that follows the width it is given ---------------------------

    def _wrap_with_column(self, label: ttk.Label) -> ttk.Label:
        """Register a label that wraps to the results column."""
        self._results_labels.append(label)
        label.configure(wraplength=self._results_wrap, justify="left")
        return label

    def _wrap_with_window(self, label: ttk.Label) -> ttk.Label:
        """Register a label that wraps to the whole window."""
        self._window_labels.append(label)
        label.configure(wraplength=self._window_wrap, justify="left")
        return label

    def _results_resized(self, event: tk.Event) -> None:
        """The results column changed width, so its text has to follow.

        A fixed wraplength is a promise about the width of the window,
        and this window has no fixed width: at 1,100 pixels the right
        column was cutting sentences off at its edge with nothing to say
        so.
        """
        wrap = max(MINIMUM_WRAP, event.width - 28)
        if wrap == self._results_wrap:
            return
        self._results_wrap = wrap
        for label in self._results_labels:
            label.configure(wraplength=wrap)
        # Two of these are written again on every recalculation, with
        # their own wraplength, so the numbers have to be redrawn too.
        if self.ready and self._last_drawn is not None:
            cfg, out = self._last_drawn
            self._update_system(cfg, out)
            self._update_speed(cfg, out)

    def _window_resized(self, event: tk.Event) -> None:
        wrap = max(MINIMUM_WRAP, event.width - 44)
        if wrap == self._window_wrap:
            return
        self._window_wrap = wrap
        for label in self._window_labels:
            label.configure(wraplength=wrap)

    # -- outputs -----------------------------------------------------------

    def _build_outputs(self, frame: ttk.Frame) -> None:
        frame.pack(side="left", fill="both", expand=True)

        self.status_label = ttk.Label(frame, text="", font=("Segoe UI", 12, "bold"))
        self.status_label.pack(anchor="w")
        self.headline_label = self._wrap_with_column(ttk.Label(frame, text=""))
        self.headline_label.pack(anchor="w")
        self.note_label = self._wrap_with_column(
            ttk.Label(frame, text="", foreground="#8A5A00")
        )
        self.note_label.pack(anchor="w", pady=(0, 8))

        speed = ttk.LabelFrame(frame, text="Speed", padding=8)
        speed.pack(fill="x")
        self.pp_label = ttk.Label(speed, text="", font=("Segoe UI", 11))
        self.pp_label.grid(row=0, column=0, sticky="w")
        self.tg_label = ttk.Label(speed, text="", font=("Segoe UI", 11))
        self.tg_label.grid(row=1, column=0, sticky="w")
        self.speed_detail = self._wrap_with_column(
            ttk.Label(speed, text="", foreground="#555555")
        )
        self.speed_detail.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.measured_label = self._wrap_with_column(
            ttk.Label(speed, text="", foreground="#1B7F3B")
        )
        self.measured_label.grid(row=3, column=0, sticky="w", pady=(4, 0))

        system = ttk.LabelFrame(frame, text="System memory", padding=8)
        system.pack(fill="x", pady=(10, 0))
        self.system_label = self._wrap_with_column(ttk.Label(system, text=""))
        self.system_label.pack(anchor="w")

        advice = ttk.LabelFrame(frame, text="What to change", padding=8)
        advice.pack(fill="both", expand=True, pady=(10, 0))
        self.advice_label = self._wrap_with_column(ttk.Label(advice, text=""))
        self.advice_label.pack(anchor="w")

        # The launcher belongs with the results, not at the foot of the
        # window: it is what the rest of this column adds up to.
        self._build_launcher_panel(frame)

    # -- the launcher panel -------------------------------------------------

    #: Keys that move the cursor or extend a selection instead of editing.
    #: Shift+Left is still keysym "Left", so this list covers the
    #: shift-held case too without checking the modifier.
    _LAUNCHER_NAV_KEYSYMS: typing.ClassVar[set[str]] = {
        "Left",
        "Right",
        "Up",
        "Down",
        "Home",
        "End",
        "Prior",
        "Next",
        "Shift_L",
        "Shift_R",
        "Control_L",
        "Control_R",
        "Alt_L",
        "Alt_R",
        "Tab",
    }
    #: With Control held, the only keys let through: copy and select-all.
    _LAUNCHER_ALLOWED_CONTROL_KEYSYMS: typing.ClassVar[set[str]] = {
        "c",
        "C",
        "a",
        "A",
        "Insert",
    }

    def _build_launcher_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, padding=8)
        frame.pack(fill="both", expand=True, pady=(10, 0))

        # The Copy button sits beside the title, so the title is built as
        # a small frame and handed to the labelframe as its labelwidget
        # instead of using the plain text= option.
        label_row = ttk.Frame(frame)
        ttk.Label(
            label_row,
            text="Launcher for these settings (save as a .cmd file)",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        ttk.Button(label_row, text="Copy", command=self._copy_launcher_script).pack(
            side="left", padx=(8, 0)
        )
        self.copy_confirmation = ttk.Label(label_row, text="", foreground="#1B7F3B")
        self.copy_confirmation.pack(side="left", padx=(6, 0))
        frame.configure(labelwidget=label_row)

        # The script starts with "pushd %~dp0" and calls llama-server.exe
        # by its bare name, so it only runs from next to that executable.
        self.launcher_folder_note = ttk.Label(
            frame,
            text=(
                "Save this file in the llama.cpp folder, next to "
                "llama-server.exe. The script makes its own folder the "
                "working folder."
            ),
        )
        self.launcher_folder_note.pack(fill="x", anchor="w")

        # A ceiling in pixels, not in lines: inside a column that scrolls,
        # a text widget with no ceiling grows to the length of the script
        # and pushes everything above it out of reach.
        text_row = ttk.Frame(frame, height=LAUNCHER_MAX_HEIGHT)
        text_row.pack(fill="both", expand=True)
        text_row.pack_propagate(False)
        self.launcher_text = tk.Text(
            text_row,
            # Twelve lines, not twenty. The panel is packed first and takes
            # its height off the top of the window, so every line here is a
            # line the sliders above do not get. The text has its own
            # scrollbar and the Copy button takes the whole script anyway.
            height=12,
            width=100,
            font=("Consolas", 9),
            wrap="none",
            undo=False,
        )
        self.launcher_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            text_row, orient="vertical", command=self.launcher_text.yview
        )
        scrollbar.pack(side="left", fill="y")
        self.launcher_text.configure(yscrollcommand=scrollbar.set)

        # The widget stays in the normal state, not "disabled", so the
        # mouse can still select and drag; every keystroke that would
        # change the text is refused here instead, one by one.
        self.launcher_text.bind("<Key>", self._on_launcher_text_key)

    def _on_launcher_text_key(self, event: tk.Event) -> str | None:
        if event.keysym in self._LAUNCHER_NAV_KEYSYMS:
            return None
        if (
            event.state & 0x4  # Control held
            and event.keysym in self._LAUNCHER_ALLOWED_CONTROL_KEYSYMS
        ):
            return None
        return "break"

    def _copy_launcher_script(self) -> None:
        text = self.launcher_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_confirmation.configure(text="Copied")
        self.after(2000, lambda: self.copy_confirmation.configure(text=""))

    def _update_launcher(self, cfg: E.Config, out: E.Estimate) -> None:
        text = E.launcher_script(cfg, out)
        widget = self.launcher_text
        # Keep the scroll position across a rebuild: otherwise dragging a
        # slider would keep snapping the view back to the top.
        top, _bottom = widget.yview()
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.yview_moveto(top)

    # -- reading the controls ---------------------------------------------

    def current_config(self) -> tuple[E.Config, list[str]]:
        problems: list[str] = []

        vram, vram_ok = read_number(self.vram_text, self.last_vram, 1024, 262144)
        if vram_ok:
            self.last_vram = vram
        else:
            problems.append(
                "Video memory must be a number between 1,024 and "
                "262,144 MiB. "
                f"Using {self.last_vram:,.0f} MiB."
            )
            vram = self.last_vram

        ram, ram_ok = read_number(self.ram_text, self.last_ram, 4, 2048)
        if ram_ok:
            self.last_ram = ram
        else:
            problems.append(
                "System memory must be a number between 4 and 2,048 GB. "
                f"Using {self.last_ram:g} GB."
            )
            ram = self.last_ram

        cfg = E.Config(
            ctx=M.CTX_CHOICES[self._stop(self.ctx_index, M.CTX_CHOICES)],
            ubatch=M.UBATCH_CHOICES[self._stop(self.ubatch_index, M.UBATCH_CHOICES)],
            kv_type=self.kv_type.get(),
            threads=self._clamped_int(self.threads, M.THREAD_MIN, M.THREAD_MAX),
            fit_margin=self._snapped_margin(),
            vision=self.vision.get(),
            speculation=self.speculation.get(),
            repetitiveness=max(0.0, min(100.0, self.repetitiveness.get())),
            vram_total_mib=int(vram),
            vram_busy_mib=self._busy_mib(int(vram)),
            ram_total_gb=ram,
        )
        return cfg, problems

    def _busy_mib(self, total: int) -> int:
        """Video memory other programs hold, as the estimator wants it.

        Zero unless the box is ticked and a reading has arrived. It is
        capped at the card size: a reading larger than the field above
        would otherwise produce a negative capacity rather than an
        obvious "nothing left".
        """
        if not self.show_vram.get() or self.vram_reading is None:
            return 0
        return max(0, min(int(self.vram_reading.used_mib), total))

    @staticmethod
    def _stop(variable: tk.DoubleVar, choices: list[int]) -> int:
        """Snap a slider position to the nearest stop in a list of choices."""
        index = round(variable.get())
        return max(0, min(len(choices) - 1, index))

    @staticmethod
    def _clamped_int(variable: tk.DoubleVar, low: int, high: int) -> int:
        return max(low, min(high, round(variable.get())))

    def _snapped_margin(self) -> int:
        step = M.FIT_MARGIN_STEP
        snapped = round(self.fit_margin.get() / step) * step
        return max(M.FIT_MARGIN_MIN, min(M.FIT_MARGIN_MAX, snapped))

    # -- the live recalculation -------------------------------------------

    def recalculate(self) -> None:
        cfg, problems = self.current_config()
        out = E.estimate(cfg)

        self.machine_note.configure(text="  ".join(problems))
        self._update_captions(cfg)
        self._update_status(out)
        self._update_speed(cfg, out)
        self._last_drawn = (cfg, out)
        segments, tail = self._bar_parts(cfg, out)
        self._draw_bar(cfg, out, segments, tail)
        self._draw_legend(out, segments)
        self._update_layers(out)
        self._update_system(cfg, out)
        self._update_advice(out)
        self._update_launcher(cfg, out)

    def _build_memory_row(self, parent: ttk.Frame) -> None:
        """The video-memory bar, across the whole window.

        It used to live in the right-hand column. With a segment for each
        process holding the card it needs the width: at a third of the
        window a 40 MiB process is under one pixel wide.
        """
        memory = ttk.LabelFrame(parent, text="Video memory", padding=8)
        memory.pack(fill="x", pady=(10, 0))
        self.canvas = tk.Canvas(
            memory,
            height=BAR_HEIGHT + 22,
            highlightthickness=0,
            background="#FFFFFF",
        )
        self.canvas.pack(fill="x")
        self.legend = tk.Canvas(
            memory,
            height=76,
            highlightthickness=0,
            background="#FFFFFF",
        )
        self.legend.pack(fill="x", pady=(4, 0))
        # The bar is drawn in pixels, so it has to be drawn again whenever
        # the window changes width. The estimate behind it does not change,
        # so the last one is reused rather than recomputed.
        self.canvas.bind("<Configure>", self._redraw_bar)
        self.layers_label = ttk.Label(memory, text="", font=("Segoe UI", 10, "bold"))
        self.layers_label.pack(anchor="w", pady=(4, 0))
        # Two different layer counts appear side by side, so they are named
        # once, here, and every other line points back at this sentence.
        two_counts = ttk.Label(
            memory,
            text=f"Two counts, two meanings: llama.cpp counts the "
            f"{M.BLOCK_COUNT} blocks plus the output layer and reports "
            f'{M.LLAMA_LAYER_COUNT}; "expert layers on the card" '
            f"counts only the blocks whose expert weights sit on the "
            f"card, out of {M.BLOCK_COUNT}.",
            foreground="#777777",
        )
        self._wrap_with_window(two_counts)
        two_counts.pack(anchor="w", pady=(2, 0))

    def _redraw_bar(self, _event=None) -> None:
        if not self.ready or self._last_drawn is None:
            return
        cfg, out = self._last_drawn
        segments, tail = self._bar_parts(cfg, out)
        self._draw_bar(cfg, out, segments, tail)
        self._draw_legend(out, segments)

    def _bar_pixels(self) -> float:
        """The width to draw into, never zero before the window is mapped."""
        width = self.canvas.winfo_width()
        return float(width if width > 1 else BAR_WIDTH)

    def _segment_colour(self, name: str) -> str:
        if name.startswith("proc"):
            return BUSY_COLOURS[int(name[4:]) % len(BUSY_COLOURS)]
        if name == "busy_rest":
            return BUSY_REST_COLOUR
        return COLOUR[name]

    def _segment_label(self, name: str) -> str:
        if name in self._busy_labels:
            return self._busy_labels[name]
        return SEGMENT_NAMES[name]

    def _busy_segments(self, cfg: E.Config) -> list[tuple[str, float]]:
        """The left end of the bar: who is holding the card right now.

        The named processes are capped by the total, and whatever is left
        becomes one block. That block is never empty by accident: the
        total comes from NVML, which counts memory no process claims,
        while the breakdown comes from a Windows counter that only counts
        processes. The difference is real and belongs on the bar.
        """
        self._busy_labels = {}
        if cfg.vram_busy_mib <= 0:
            return []

        listed = []
        if self.vram_reading is not None:
            listed = list(self.vram_reading.processes[:BUSY_PROCESS_COUNT])

        segments: list[tuple[str, float]] = []
        taken = 0
        for index, (process, mib) in enumerate(listed):
            mib = min(int(mib), cfg.vram_busy_mib - taken)
            if mib <= 0:
                break
            key = f"proc{index}"
            # The name only. The legend puts the size after every label it
            # draws, and a label carrying its own size read "dwm: 1,418
            # MiB: 1,418 MiB".
            self._busy_labels[key] = process
            segments.append((key, float(mib)))
            taken += mib

        rest = cfg.vram_busy_mib - taken
        if rest > 0:
            segments.append(("busy_rest", float(rest)))
            self._busy_labels["busy_rest"] = (
                "Other processes and the driver" if segments[:-1] else "Other programs"
            )
        return segments

    def _update_captions(self, cfg: E.Config) -> None:
        self.ctx_caption.configure(text=f"Context window: {cfg.ctx:,} tokens")
        ubatch_text = (
            f"Micro-batch: {cfg.ubatch:,} tokens, the tokens llama.cpp pushes "
            f"through the model at a time. The batch is twice the micro-batch "
            f"and is not editable here."
        )
        if cfg.ubatch_clamped:
            ubatch_text += (
                f"  Larger than the window, so {cfg.effective_ubatch:,} "
                f"tokens are used, with a batch of {cfg.batch:,} tokens."
            )
        else:
            ubatch_text += f"  Batch: {cfg.batch:,} tokens."
        self.ubatch_caption.configure(text=ubatch_text)
        self.threads_caption.configure(
            text=f"Threads: {cfg.threads}   (above 16 threads the difference "
            f"is inside the measurement noise)"
        )
        self.margin_caption.configure(
            text=f"Fit margin: {cfg.fit_margin:,} MiB   (the video memory "
            f"--fit is told to leave free, -fitt)"
        )

        if cfg.speculation:
            self.repetition_scale.state(["!disabled"])
            multiplier = E.speculation_factor(True, cfg.repetitiveness)
            self.repetition_caption.configure(
                text=f"How repetitive is your text: {cfg.repetitiveness:.0f}% "
                f"(speculation x{multiplier:.2f})"
            )
        else:
            self.repetition_scale.state(["disabled"])
            self.repetition_caption.configure(
                text="How repetitive is your text: switch speculation on first"
            )

        # The levers constrain each other in plain sight. Nothing is clamped:
        # the slider keeps whatever the user chose and the line below says
        # what the card can actually hold.
        biggest_ubatch = E.largest_ubatch_that_fits(cfg)
        if biggest_ubatch is None:
            self.ubatch_limit.configure(
                text="No micro-batch fits this window on this card.",
                foreground="#B02020",
            )
        elif cfg.ubatch > biggest_ubatch:
            self.ubatch_limit.configure(
                text=f"Largest micro-batch that still fits here: {biggest_ubatch:,} tokens.",
                foreground="#B02020",
            )
        else:
            self.ubatch_limit.configure(
                text=f"Largest micro-batch that still fits here: {biggest_ubatch:,} tokens.",
                foreground="#777777",
            )

        biggest_ctx = E.largest_ctx_that_fits(cfg)
        if biggest_ctx is None:
            self.ctx_limit.configure(
                text="No window fits this micro-batch on this card.",
                foreground="#B02020",
            )
        elif cfg.ctx > biggest_ctx:
            self.ctx_limit.configure(
                text=f"Largest window that still fits here: {biggest_ctx:,} tokens.",
                foreground="#B02020",
            )
        else:
            self.ctx_limit.configure(
                text=f"Largest window that still fits here: {biggest_ctx:,} tokens.",
                foreground="#777777",
            )

    def _update_status(self, out: E.Estimate) -> None:
        self.status_label.configure(
            text=STATUS_WORD[out.status], foreground=STATUS_COLOUR[out.status]
        )
        self.headline_label.configure(text=out.headline)
        self.note_label.configure(text="  ".join(out.notes))

    def _update_speed(self, cfg: E.Config, out: E.Estimate) -> None:
        if out.status == E.FAIL:
            self.pp_label.configure(text="Prompt processing: -")
            self.tg_label.configure(text="Generation: -")
            self.speed_detail.configure(text="The server does not start.")
            self.measured_label.configure(text="")
            return

        pp_text = f"Prompt processing: {out.pp_tps:,.0f} tokens/s"
        if out.pp_extrapolated:
            pp_text += "   extrapolated"
        self.pp_label.configure(text=pp_text)

        tg_text = f"Generation: {out.tg_tps:,.2f} tokens/s"
        if out.tg_extrapolated:
            tg_text += "   extrapolated"
        self.tg_label.configure(text=tg_text)

        # The line reads as the formula it is, with units and operators.
        formula = (
            f"Generation = {out.tg_before_threads:,.2f} tokens/s base "
            f"x {out.thread_factor:.2f} threads"
        )
        if cfg.speculation:
            formula += f" x {out.speculation_factor:.2f} speculation"
        formula += f" = {out.tg_tps:,.2f} tokens/s."
        detail = formula + (
            "  Prompt processing does not depend on threads or on "
            "speculation; it was measured against the window and the "
            "micro-batch only."
        )
        marks = []
        if out.tg_extrapolated:
            marks.append(f"Generation is extrapolated: {out.tg_extrapolation_reason}.")
        if out.pp_extrapolated:
            marks.append(
                f"Prompt processing is extrapolated: "
                f"{out.pp_extrapolation_reason}. The line carries on with "
                f"the slope of the nearest measured edge of the table."
            )
        if marks:
            detail += "  " + "  ".join(marks)
        self.speed_detail.configure(text=detail)

        self.measured_label.configure(
            text="\n".join(E.validation_comparison_lines(cfg))
        )

    def _bar_parts(
        self, cfg: E.Config, out: E.Estimate
    ) -> tuple[list[tuple[str, float]], float]:
        """The segments of the bar and the overflow tail, computed once.

        The bar and the legend both read this, so they cannot disagree.
        """
        segments = [
            ("dense", out.dense_mib),
            ("experts", out.expert_mib),
            ("kv", out.kv_mib),
            ("compute", out.compute_mib),
            ("vision", out.vision_mib),
            ("margin", out.margin_mib),
            ("free", out.free_mib),
        ]
        # Memory other programs hold goes first, at the left edge, split
        # into the processes holding it. Nothing is drawn when the box is
        # unticked: a legend row reading "Other programs 0" is noise.
        segments[:0] = self._busy_segments(cfg)

        drawn = sum(value for _name, value in segments)
        capacity = float(cfg.vram_total_mib)

        # A tail is only added when the segments stop at the capacity line.
        # For a run that does not start, the segments already run past the
        # line by exactly the overflow, so a tail would count the same MiB
        # twice; what is past the line is the overflow either way. The
        # comparison carries a tolerance because the segments add up to the
        # capacity to within a rounding error, and the bare "drawn <
        # capacity" test threw the tail away exactly when it was needed.
        tail = out.overflow_mib if drawn <= capacity + BAR_TOLERANCE_MIB else 0.0
        return segments, tail

    def _draw_bar(
        self,
        cfg: E.Config,
        out: E.Estimate,
        segments: list[tuple[str, float]],
        tail: float,
    ) -> None:
        canvas = self.canvas
        canvas.delete("all")

        drawn = sum(value for _name, value in segments)
        capacity = float(cfg.vram_total_mib)
        scale_total = max(capacity, drawn + tail, 1.0)
        bar_width = self._bar_pixels()
        pixels_per_mib = bar_width / scale_total

        x = 0.0
        for name, value in segments:
            if value <= 0:
                continue
            width = value * pixels_per_mib
            canvas.create_rectangle(
                x, 0, x + width, BAR_HEIGHT, fill=self._segment_colour(name), outline=""
            )
            x += width

        if tail > 0:
            width = tail * pixels_per_mib
            canvas.create_rectangle(
                x,
                0,
                x + width,
                BAR_HEIGHT,
                fill=COLOUR["overflow"],
                outline="",
                stipple="gray50",
            )
            x += width

        # The card's real capacity, so the overflow is visible as overflow.
        capacity_x = cfg.vram_total_mib * pixels_per_mib
        canvas.create_line(
            capacity_x, -2, capacity_x, BAR_HEIGHT + 6, fill="#000000", width=2
        )
        canvas.create_text(
            min(capacity_x, bar_width - 4),
            BAR_HEIGHT + 13,
            text=f"card holds {cfg.vram_total_mib:,} MiB",
            anchor="e" if capacity_x > bar_width * 0.5 else "w",
            font=("Segoe UI", 8),
        )

    def _draw_legend(self, out: E.Estimate, segments: list[tuple[str, float]]) -> None:
        legend = self.legend
        legend.delete("all")

        entries = [(name, self._segment_label(name), value) for name, value in segments]
        # Free memory and an overflow are never both above zero, so these two
        # lines can never contradict each other.
        if out.overflow_mib > 0:
            entries.append(("overflow", "Asked for beyond the card", out.overflow_mib))

        texts = [f"{label}: {value:,.0f} MiB" for _name, label, value in entries]
        column_width = (
            max(self._legend_font.measure(text) for text in texts)
            + LEGEND_TEXT_INSET
            + LEGEND_COLUMN_GAP
        )
        sizes = legend_columns(len(entries), self._bar_pixels(), column_width)
        legend.configure(height=max(sizes) * LEGEND_ROW_HEIGHT + 8)

        index = 0
        for column, rows in enumerate(sizes):
            for row in range(rows):
                name = entries[index][0]
                x = column * column_width + 2
                y = row * LEGEND_ROW_HEIGHT + 4
                legend.create_rectangle(
                    x,
                    y,
                    x + LEGEND_PATCH_WIDTH,
                    y + LEGEND_PATCH_WIDTH,
                    fill=self._segment_colour(name),
                    outline="#888888",
                )
                legend.create_text(
                    x + LEGEND_TEXT_INSET,
                    y + 6,
                    text=texts[index],
                    anchor="w",
                    font=("Segoe UI", 9),
                )
                index += 1

    def _update_layers(self, out: E.Estimate) -> None:
        """One line, and it always agrees with the state above it."""
        if out.status == E.FAIL:
            self.layers_label.configure(
                text="Nothing is placed on the card: the run does not start.",
                foreground=STATUS_COLOUR[E.FAIL],
            )
        elif out.status == E.DEGRADED:
            if out.llama_layers == 0:
                second = (
                    f"Of the {M.LLAMA_LAYER_COUNT} layers llama.cpp counts, "
                    f"none stays on the card: every layer runs on the "
                    f"processor, attention included."
                )
            else:
                second = (
                    f"Of the {M.LLAMA_LAYER_COUNT} layers llama.cpp counts, "
                    f"about {out.llama_layers} stay on the card, attention "
                    f"included; the rest run on the processor."
                )
            self.layers_label.configure(
                text=f"No expert layer keeps its expert weights on the card: "
                f"0 of {out.expert_layers_max}. {second}",
                foreground=STATUS_COLOUR[E.DEGRADED],
            )
        else:
            self.layers_label.configure(
                text=f"Expert layers with their expert weights on the card: "
                f"{out.expert_layers} of {out.expert_layers_max}. All "
                f"{M.LLAMA_LAYER_COUNT} layers llama.cpp counts stay on "
                f"the card.",
                foreground=STATUS_COLOUR[E.OK],
            )

    def _update_system(self, cfg: E.Config, out: E.Estimate) -> None:
        table = (
            f"The {M.NGRAM_TABLE_GB:g} GB n-gram table inside the model file "
            f"stays on the solid-state drive, about "
            f"{M.NGRAM_TABLE_KB_PER_TOKEN:g} KB of it is read per token, and "
            f"it is not counted here."
        )
        if out.all_weights_on_card:
            # The subtraction is floored at zero, and this is the case where
            # it would have gone negative: the card holds the lot.
            first = (
                f"Weights in system memory: none. The card holds every "
                f"weight of this model file, so nothing of the "
                f"{M.TOTAL_WEIGHTS_GB:g} GB it needs is left for system "
                f"memory."
            )
        else:
            first = (
                f"Weights in system memory: {out.weights_system_gb:,.1f} GB "
                f"of the {M.TOTAL_WEIGHTS_GB:g} GB this model file needs."
            )
        parts = [first]
        if out.cache_system_gb > 0:
            parts.append(
                f"Key-value cache pushed off the card: {out.cache_system_gb:,.1f} GB."
            )
            parts.append(f"In system memory altogether: {out.system_gb:,.1f} GB.")
        parts.append(
            f"With {M.RAM_HEADROOM_GB:g} GB kept for everything else: "
            f"{out.system_gb + M.RAM_HEADROOM_GB:,.1f} GB against "
            f"{cfg.ram_total_gb:g} GB installed."
        )
        parts.append(table)
        text = "  ".join(parts)
        self.system_label.configure(
            text=text,
            foreground="#000000" if out.ram_ok else STATUS_COLOUR[E.FAIL],
            wraplength=self._results_wrap,
            justify="left",
        )

    def _update_advice(self, out: E.Estimate) -> None:
        if out.advice:
            text = "\n".join(f"- {line}" for line in out.advice)
        else:
            text = "- Nothing to change. This configuration fits as it is."
        self.advice_label.configure(
            text=text,
            foreground="#000000" if out.status == E.OK else STATUS_COLOUR[out.status],
        )


def main() -> None:
    root = tk.Tk()
    root.minsize(1020, 760)
    window = Calculator(root)
    window.pack(fill="both", expand=True)

    def closing() -> None:
        # Only here, and not in destroy: the tests build and tear down
        # this window many times, and none of them should write over
        # what a person left in the settings file.
        window.save()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", closing)
    root.mainloop()


if __name__ == "__main__":
    main()
