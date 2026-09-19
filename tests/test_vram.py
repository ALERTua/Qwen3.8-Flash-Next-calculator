"""Reading the card: the probe, and the per-process breakdown it returns.

Run with:  uv run pytest tests/test_vram.py
"""

import pytest

from qwen_flash_calculator import estimator as E


class TestBusyVideoMemory:
    """Memory other programs hold is capacity this run never gets."""

    def test_the_card_keeps_its_real_size(self):
        cfg = E.Config(vram_total_mib=16376, vram_busy_mib=2075)
        assert cfg.vram_total_mib == 16376
        assert cfg.vram_for_us_mib == 14301

    def test_a_full_card_leaves_nothing_rather_than_a_negative_number(self):
        cfg = E.Config(vram_total_mib=16376, vram_busy_mib=20000)
        assert cfg.vram_for_us_mib == 0

    def test_every_busy_mib_comes_off_the_room_for_experts(self):
        empty = E.Config()
        busy = E.Config(vram_busy_mib=2048)
        assert E.free_for_experts(empty) - E.free_for_experts(busy) == pytest.approx(
            2048, abs=1e-6
        )

    def test_a_busy_card_costs_expert_layers_and_speed(self):
        empty = E.estimate(E.Config())
        busy = E.estimate(E.Config(vram_busy_mib=2080))
        assert busy.expert_layers < empty.expert_layers
        assert busy.tg_tps < empty.tg_tps

    def test_a_card_held_entirely_by_other_programs_does_not_start(self):
        out = E.estimate(E.Config(vram_busy_mib=16376))
        assert out.status == E.FAIL

    def test_the_launcher_says_what_was_held_and_only_when_it_was(self):
        cfg = E.Config(vram_busy_mib=2075)
        text = E.launcher_script(cfg, E.estimate(cfg))
        assert "Other programs held 2,075 MiB" in text
        clean = E.Config()
        assert "Other programs held" not in E.launcher_script(clean, E.estimate(clean))


class TestReadingTheCard:
    """The probe itself, driven with fakes instead of a real card."""

    def setup_method(self):
        from qwen_flash_calculator import vram

        self.vram = vram

    def test_the_first_way_that_answers_wins(self):
        good = lambda: self.vram.Reading(1, 2, "first", False)
        never = lambda: pytest.fail("the second way must not be tried")
        assert self.vram.read([good, never]).source == "first"

    def test_a_way_that_cannot_work_hands_over_to_the_next(self):
        def missing():
            raise self.vram.Unavailable("nvml.dll not found")

        second = lambda: self.vram.Reading(3, 4, "second", False)
        assert self.vram.read([missing, second]).source == "second"

    def test_when_nothing_answers_the_reason_comes_back(self):
        def missing():
            raise self.vram.Unavailable("no library")

        def broken():
            raise OSError("driver said no")

        answer = self.vram.read([missing, broken])
        assert answer[0] is None
        assert "no library" in answer[1]
        assert "driver said no" in answer[1]

    def test_the_counter_reads_the_last_line_and_rejects_junk(self):
        reading = self.vram.read_counter(runner=lambda _cmd: "\n1759\n")
        assert reading.used_mib == 1759
        assert reading.total_mib is None
        with pytest.raises(self.vram.Unavailable):
            self.vram.read_counter(runner=lambda _cmd: "Get-Counter : failed")

    def test_free_memory_is_only_offered_when_the_total_is_known(self):
        assert self.vram.Reading(2000, 16376, "NVML", False).free_mib == 14376
        assert self.vram.Reading(2000, None, "counter", False).free_mib is None


class TestWhoHoldsTheCard:
    """The per-process split, parsed from the Windows counter."""

    def setup_method(self):
        from qwen_flash_calculator import vram

        self.vram = vram

    def test_rows_are_added_up_by_name_and_sorted(self):
        text = "dwm|1201\nexplorer|141\ndwm|30\njunk\ncsrss|122\n"
        got = self.vram.read_processes(runner=lambda _cmd: text)
        assert got == (("dwm", 1231), ("explorer", 141), ("csrss", 122))

    def test_a_reading_carries_the_breakdown(self):
        base = lambda: self.vram.Reading(2000, 16376, "NVML", False)
        reading = self.vram.read(ways=[base], processes=lambda: (("dwm", 1201),))
        assert reading.processes == (("dwm", 1201),)

    def test_a_failed_breakdown_does_not_lose_the_total(self):
        base = lambda: self.vram.Reading(2000, 16376, "NVML", False)

        def broken():
            raise OSError("counter unavailable")

        reading = self.vram.read(ways=[base], processes=broken)
        assert reading.used_mib == 2000
        assert reading.processes == ()
