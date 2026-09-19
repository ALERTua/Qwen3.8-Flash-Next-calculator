"""The window itself: layout, controls, and what a person sees on screen.

Run with:  uv run pytest tests/test_window.py -q
"""

import pathlib
import re
import tempfile
from tkinter import ttk

import pytest

from qwen_flash_calculator import estimator as E
from qwen_flash_calculator import measured as M

_ROOT = None


def shared_root():
    """One Tk root for every class in this file.

    Nine roots created and destroyed in one process left Tcl without its
    own library path, and the classes that ran after that point skipped
    themselves with "invalid command name tcl_findLibrary". A skip hides
    a test, which is worse than a failure, so the root is built once and
    left alive until the process ends. Each class still destroys its own
    window in teardown.
    """
    global _ROOT
    if _ROOT is None:
        import tkinter as tk

        try:
            _ROOT = tk.Tk()
        except tk.TclError as error:  # no display
            pytest.skip(f"no tkinter display: {error}")
        _ROOT.withdraw()
    return _ROOT


class TestTheWindowItself:
    """The frame is built under a withdrawn root and destroyed again."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def load_profile(self, ctx: int, ubatch: int, threads: int) -> None:
        """What the removed profile buttons used to set.

        The buttons are gone from the window; these settings are still
        worth testing, because they are the five directly measured
        configurations and the comparison text hangs off them.
        """
        app = self.app
        app.ctx_index.set(M.CTX_CHOICES.index(ctx))
        app.ubatch_index.set(M.UBATCH_CHOICES.index(ubatch))
        app.threads.set(threads)
        app.kv_type.set("q8_0")
        app.fit_margin.set(M.DEFAULT_FIT_MARGIN_MIB)
        app.speculation.set(True)
        app.vision.set(True)

    def configure(self, ctx: int, ubatch: int, **rest) -> E.Config:
        app = self.app
        app.ctx_index.set(M.CTX_CHOICES.index(ctx))
        app.ubatch_index.set(M.UBATCH_CHOICES.index(ubatch))
        app.kv_type.set(rest.get("kv_type", "q8_0"))
        app.vision.set(rest.get("vision", False))
        app.vram_text.set(str(rest.get("vram_total_mib", M.REFERENCE_VRAM_MIB)))
        app.fit_margin.set(rest.get("fit_margin", M.DEFAULT_FIT_MARGIN_MIB))
        self.root.update_idletasks()
        cfg, _problems = app.current_config()
        return cfg

    def test_the_overflow_segment_is_drawn_whenever_the_run_overflows(self):
        """It used to vanish: the segments add up to the capacity exactly.

        The bar is what the eye reads first, so a spill that draws no
        overflow segment is the one error the legend cannot repair.
        """
        for ctx, ubatch in [(262144, 2048), (262144, 4096), (196608, 4096)]:
            cfg = self.configure(ctx, ubatch)
            out = E.estimate(cfg)
            _segments, tail = self.app._bar_parts(cfg, out)
            assert out.status == E.DEGRADED, f"ctx={ctx}, ubatch={ubatch}"
            assert out.overflow_mib > 0.0, f"ctx={ctx}, ubatch={ubatch}"
            assert tail == pytest.approx(out.overflow_mib, abs=1e-7), (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_the_bar_and_the_legend_never_disagree(self):
        """Either a tail is drawn, or the segments already run past the line
        by exactly the overflow the legend names."""
        for ctx, ubatch in [
            (163840, 512),
            (262144, 2048),
            (262144, 4096),
            (262144, 8192),
        ]:
            cfg = self.configure(ctx, ubatch)
            out = E.estimate(cfg)
            segments, tail = self.app._bar_parts(cfg, out)
            drawn = sum(value for _name, value in segments)
            past_the_line = max(0.0, drawn - cfg.vram_total_mib) + tail
            assert past_the_line == pytest.approx(out.overflow_mib, abs=10**-6), (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_the_status_word_and_the_headline_agree(self):
        for ctx, ubatch, word in [
            (163840, 512, "FITS"),
            (262144, 2048, "SPILLS TO THE PROCESSOR"),
            (262144, 8192, "DOES NOT START"),
        ]:
            self.configure(ctx, ubatch)
            assert self.app.status_label.cget("text") == word, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_the_layer_line_agrees_with_the_state(self):
        self.configure(262144, 2048)
        line = self.app.layers_label.cget("text")
        assert "No expert layer" in line
        assert f"0 of {M.LLAMA_LAYER_COUNT}" not in line

        self.configure(163840, 512)
        line = self.app.layers_label.cget("text")
        assert f"4 of {M.BLOCK_COUNT}" in line
        assert f"All {M.LLAMA_LAYER_COUNT} layers" in line

    def test_the_clamped_micro_batch_is_said_on_screen(self):
        self.configure(4096, 8192)
        assert "larger than the window" in self.app.note_label.cget("text")
        assert "4,096" in self.app.ubatch_caption.cget("text")

    def test_an_extrapolated_generation_figure_is_marked_on_screen(self):
        self.configure(163840, 512, vram_total_mib=49152)
        assert "extrapolated" in self.app.tg_label.cget("text")
        self.configure(163840, 512, vram_total_mib=M.REFERENCE_VRAM_MIB)
        assert "extrapolated" not in self.app.tg_label.cget("text")

    def test_the_card_that_holds_everything_says_so_in_words(self):
        self.configure(8192, 512, vram_total_mib=98304)
        text = self.app.system_label.cget("text")
        assert "Weights in system memory: none" in text
        # And no negative figure anywhere, which is what the old line printed.
        assert not re.search(r"-\d", text)

    def test_the_image_input_switch_and_the_advice_use_the_same_words(self):
        cfg = self.configure(196608, 2048, vision=True)
        out = E.estimate(cfg)
        assert any(E.IMAGE_INPUT_LABEL in line for line in out.advice)
        assert E.IMAGE_INPUT_LABEL in self.app.advice_label.cget("text")

    def test_the_launcher_panel_regenerates_when_a_control_moves(self):
        self.configure(163840, 512)
        self.root.update_idletasks()
        first_text = self.app.launcher_text.get("1.0", "end-1c")
        assert first_text.startswith("@echo off")

        self.app.threads.set(24)
        self.app.kv_type.set("f16")
        self.root.update_idletasks()
        second_text = self.app.launcher_text.get("1.0", "end-1c")

        assert first_text != second_text
        assert 'set "LLAMA_ARG_THREADS=24"' in second_text
        assert 'set "LLAMA_ARG_CACHE_TYPE_K=f16"' in second_text

    def test_the_launcher_panel_says_where_the_file_must_live(self):
        text = self.app.launcher_folder_note.cget("text")
        assert text == (
            "Save this file in the llama.cpp folder, next to "
            "llama-server.exe. The script makes its own folder the "
            "working folder."
        )

    def test_the_copy_button_puts_the_text_on_the_clipboard(self):
        self.configure(163840, 512)
        self.root.update_idletasks()
        expected = self.app.launcher_text.get("1.0", "end-1c")

        self.app._copy_launcher_script()
        self.root.update_idletasks()

        assert self.app.clipboard_get() == expected
        assert self.app.copy_confirmation.cget("text") == "Copied"

    def test_the_launcher_text_refuses_to_be_typed_into(self):
        self.configure(163840, 512)
        self.root.update_idletasks()
        before = self.app.launcher_text.get("1.0", "end-1c")

        self.app.launcher_text.focus_set()
        self.app.launcher_text.event_generate("<Key>", keysym="a")
        self.root.update_idletasks()

        assert self.app.launcher_text.get("1.0", "end-1c") == before

    def test_each_shipped_profile_sets_its_own_window_micro_batch_and_threads(self):
        for label, ctx, ubatch, threads in M.CONFIG_PRESETS:
            self.load_profile(ctx, ubatch, threads)
            self.root.update_idletasks()
            cfg, _problems = self.app.current_config()
            assert cfg.ctx == ctx, f"profile={label}"
            assert cfg.ubatch == ubatch, f"profile={label}"
            assert cfg.threads == threads, f"profile={label}"

    def test_a_shipped_profile_shows_the_measured_comparison(self):
        for label, ctx, ubatch, threads in M.CONFIG_PRESETS:
            self.load_profile(ctx, ubatch, threads)
            self.root.update_idletasks()
            text = self.app.measured_label.cget("text")
            assert "Measured here" in text, f"profile={label}"
            assert "prefill" in text, f"profile={label}"
            assert "No direct measurement" not in text, f"profile={label}"

    def test_a_configuration_off_the_five_points_says_no_measurement(self):
        # 4096/256 is not one of the five validated configurations.
        self.configure(4096, 256)
        self.app.threads.set(7)
        self.root.update_idletasks()
        text = self.app.measured_label.cget("text")
        assert "No direct measurement" in text
        assert "Measured here" not in text

    def test_the_middle_profile_matches_the_readme_figures_despite_its_own_switches(
        self,
    ):
        """The Middle profile also turns on image input and speculation.
        Those two must not be allowed to change the measured or modelled
        figures README.md documents for 131,072/2,048/24 -- they may only
        add the third line naming that they are on."""
        self.load_profile(131072, 2048, 24)
        self.root.update_idletasks()
        text = self.app.measured_label.cget("text")
        assert "prefill 525" in text
        assert "generation 20.42" in text
        assert "prefill 573" in text
        assert "9% high on prefill" in text
        assert "14% high on generation" in text

    def test_the_middle_profile_names_the_settings_it_turns_on(self):
        """The button sets image input on and speculation on; the third
        line must say exactly that, and nothing else, appears only because
        of the button, and is absent again once the settings match the
        calibration ones."""
        self.load_profile(131072, 2048, 24)
        self.root.update_idletasks()
        text = self.app.measured_label.cget("text")
        assert (
            "Your settings differ from the measured run: "
            "image input on, speculation on."
        ) in text

        self.app.vision.set(False)
        self.app.speculation.set(False)
        self.root.update_idletasks()
        text = self.app.measured_label.cget("text")
        assert "Your settings differ" not in text


class TestTheBusyPartOfTheBar:
    """Five processes by name, everything else in one block."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G
        from qwen_flash_calculator import vram

        cls.G = G
        cls.vram = vram
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def reading(self, processes, used=2000):
        return self.vram.Reading(used, 16376, "NVML", False, processes)

    def setup_method(self):
        self.app.show_vram.set(True)

    def teardown_method(self):
        self.app.show_vram.set(False)
        self.app.vram_reading = None

    def test_five_are_named_and_the_rest_are_one_block(self):
        self.app.vram_reading = self.reading(
            (
                ("dwm", 1200),
                ("explorer", 140),
                ("csrss", 120),
                ("claude", 90),
                ("terminal", 80),
                ("uniget", 70),
                ("razer", 25),
            ),
            used=2000,
        )
        cfg, _ = self.app.current_config()
        segments = self.app._busy_segments(cfg)
        names = [name for name, _value in segments]
        assert names == ["proc0", "proc1", "proc2", "proc3", "proc4", "busy_rest"]
        assert sum(value for _name, value in segments) == pytest.approx(2000, abs=0.5)
        # The label is the bare name: the legend adds the size itself, and
        # a label carrying its own size printed it twice.
        assert self.app._segment_label("proc0") == "dwm"
        assert "MiB" not in self.app._segment_label("proc0")
        assert "Other processes" in self.app._segment_label("busy_rest")

    def test_fewer_than_five_processes_still_balance(self):
        self.app.vram_reading = self.reading((("dwm", 1200),), used=2000)
        cfg, _ = self.app.current_config()
        segments = self.app._busy_segments(cfg)
        assert [n for n, _v in segments] == ["proc0", "busy_rest"]
        assert sum(v for _n, v in segments) == pytest.approx(2000, abs=0.5)

    def test_without_a_breakdown_there_is_one_block_named_plainly(self):
        self.app.vram_reading = self.reading((), used=2000)
        cfg, _ = self.app.current_config()
        segments = self.app._busy_segments(cfg)
        assert [n for n, _v in segments] == ["busy_rest"]
        assert "Other programs" in self.app._segment_label("busy_rest")

    def test_processes_never_add_up_past_the_total(self):
        """The two numbers come from two sources and can disagree."""
        self.app.vram_reading = self.reading(
            (("dwm", 5000), ("explorer", 5000)), used=1000
        )
        cfg, _ = self.app.current_config()
        segments = self.app._busy_segments(cfg)
        assert sum(v for _n, v in segments) == pytest.approx(1000, abs=0.5)

    def test_every_segment_has_a_colour_of_its_own(self):
        self.app.vram_reading = self.reading(
            (("a", 100), ("b", 90), ("c", 80), ("d", 70), ("e", 60)), used=600
        )
        cfg, _ = self.app.current_config()
        segments = self.app._busy_segments(cfg)
        colours = [self.app._segment_colour(name) for name, _v in segments]
        assert len(colours) == len(set(colours))

    def test_drawing_a_breakdown_really_draws(self):
        """The test that was missing. Building the segments is not the
        same as drawing them: the first version of this feature passed
        every other test here and raised on the colour of the first
        process, so the bar silently kept an older picture."""
        self.app.vram_reading = self.reading(
            (
                ("dwm", 1200),
                ("explorer", 140),
                ("csrss", 120),
                ("claude", 90),
                ("terminal", 80),
                ("uniget", 70),
            ),
            used=2000,
        )
        self.app.recalculate()
        self.root.update_idletasks()
        cfg, _out = self.app._last_drawn
        assert cfg.vram_busy_mib == 2000
        drawn = self.app.canvas.find_all()
        assert len(drawn) > 6
        legend_rows = [
            item
            for item in self.app.legend.find_all()
            if self.app.legend.type(item) == "text"
        ]
        labels = [self.app.legend.itemcget(item, "text") for item in legend_rows]
        assert [text for text in labels if "dwm" in text], labels

    def test_the_bar_is_drawn_to_the_width_of_the_window(self):
        """It is packed to fill, so the drawing must follow the widget."""
        assert self.app.canvas.pack_info()["fill"] == "x"


class TestTheLegendLayout:
    """Columns that hug the left edge, and never a column of stubs."""

    def test_a_short_legend_stays_one_column(self):
        from qwen_flash_calculator import app as G

        assert G.legend_columns(3, 2000, 200) == [3]
        assert G.legend_columns(7, 2000, 200) == [7]

    def test_a_column_is_never_started_for_fewer_than_four_rows(self):
        from qwen_flash_calculator import app as G

        for count in range(1, 40):
            sizes = G.legend_columns(count, 4000, 200)
            assert sum(sizes) == count, f"count={count}"
            assert min(sizes) >= min(count, 4), f"count={count}"

    def test_the_rows_are_spread_evenly_across_the_columns(self):
        """Thirteen entries used to leave the last column holding one."""
        from qwen_flash_calculator import app as G

        assert G.legend_columns(13, 4000, 200) == [5, 4, 4]
        assert G.legend_columns(8, 4000, 200) == [4, 4]

    def test_a_narrow_window_keeps_the_columns_it_can_fit(self):
        from qwen_flash_calculator import app as G

        assert G.legend_columns(13, 300, 200) == [13]
        assert G.legend_columns(13, 500, 200) == [7, 6]

    def test_nothing_to_draw_is_no_columns(self):
        from qwen_flash_calculator import app as G

        assert G.legend_columns(0, 2000, 200) == []


class TestGreyBelongsToTheProcesses:
    """Colour marks what the calculator can move."""

    @staticmethod
    def is_grey(colour: str) -> bool:
        red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
        return red == green == blue

    def test_every_process_shade_is_grey(self):
        from qwen_flash_calculator import app as G

        for colour in G.BUSY_COLOURS + (G.BUSY_REST_COLOUR, G.COLOUR["busy"]):
            assert self.is_grey(colour), f"colour={colour}"

    def test_nothing_the_calculator_can_move_is_grey(self):
        """The fit margin and the free block used to read as part of the
        dead part of the bar, which is the one part a person cannot
        change from this window."""
        from qwen_flash_calculator import app as G

        for name, colour in G.COLOUR.items():
            if name == "busy":
                continue
            assert not self.is_grey(colour), f"{name} is {colour}"


class TestTheWindowTitle:
    def test_it_carries_the_file_the_numbers_belong_to(self):
        from qwen_flash_calculator import app as G
        from qwen_flash_calculator import measured as M

        for piece in (
            M.MODEL_NAME,
            M.MODEL_QUANT,
            str(M.BLOCK_COUNT),
            f"{M.NATIVE_CONTEXT:,}",
            "mixture of experts",
        ):
            assert piece in G.WINDOW_TITLE, f"piece={piece}"


class TestWhereTheWindowOpens:
    """The three controls that now start somewhere else."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def test_the_card_is_read_from_the_first_moment(self):
        assert self.app.show_vram.get()

    def test_reading_the_card_actually_started(self):
        """The command on a checkbutton runs on a click, and there is no
        click when the box starts ticked."""
        assert self.app._vram_job is not None

    def test_resetting_the_box_starts_the_reading_again(self):
        """A reset used to write the variable and stop there. The tick came
        back and the reading stayed off, because a checkbutton runs its
        command only when a person clicks it."""
        self.app.show_vram.set(False)
        self.app._toggle_vram_poll()
        assert self.app._vram_job is None

        self.app.reset("show_vram")
        assert self.app.show_vram.get()
        assert self.app._vram_job is not None

    def test_reset_every_setting_starts_the_reading_again(self):
        """Whatever one reset does for one control, the button that resets
        the lot does for every control it touches."""
        self.app.show_vram.set(False)
        self.app._toggle_vram_poll()
        assert self.app._vram_job is None

        self.app.reset_all()
        assert self.app.show_vram.get()
        assert self.app._vram_job is not None

    def test_the_margin_starts_where_this_machine_runs(self):
        assert self.app.fit_margin.get() == 512
        cfg, _problems = self.app.current_config()
        assert cfg.fit_margin == 512

    def test_speculation_is_on_and_no_repetition_is_assumed(self):
        """The server runs with n-gram speculation, so the box is ticked.
        The slider stands at zero, because speculation returns nothing on
        text the model has not seen, and because that is the text the
        speed table behind this window was fitted on. Measured on this
        machine on 2026-09-19: fresh word salad generated at 17.7 tokens
        per second while this window, standing at 5 percent, promised
        24.5."""
        assert self.app.speculation.get()
        assert self.app.repetitiveness.get() == 0
        cfg, _problems = self.app.current_config()
        assert cfg.speculation
        assert cfg.repetitiveness == 0
        assert E.speculation_factor(True, cfg.repetitiveness) == 1.0

    def test_the_measured_grid_keeps_its_own_margin(self):
        """The interface moved; the anchor the estimator was fitted at
        did not, and the tests over that grid still stand on it."""
        from qwen_flash_calculator import measured as M

        assert M.DEFAULT_FIT_MARGIN_MIB == 1024


class TestRememberingTheControls:
    """What was left set is still set tomorrow."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.folder = tempfile.TemporaryDirectory()
        cls.settings_was = G.SETTINGS_FILE
        G.SETTINGS_FILE = pathlib.Path(cls.folder.name) / "settings.json"
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.G.SETTINGS_FILE = cls.settings_was
        cls.app.destroy()
        cls.folder.cleanup()

    def teardown_method(self):
        self.app.reset_all()

    def test_every_control_has_a_default_of_the_right_kind(self):
        for name, variable in self.app.variables().items():
            if name in ("gpu_text", "vram_text"):
                continue
            assert type(variable.get()) == type(self.app.DEFAULTS[name]), f"name={name}"

    def test_what_was_set_comes_back(self):
        self.app.threads.set(22)
        self.app.kv_type.set("f16")
        self.app.save()
        self.app.threads.set(4)
        self.app.kv_type.set("q8_0")
        self.app._load()
        assert self.app.threads.get() == 22
        assert self.app.kv_type.get() == "f16"

    def test_one_control_goes_back_on_its_own(self):
        self.app.threads.set(30)
        self.app.fit_margin.set(4096)
        self.app.reset("threads")
        assert self.app.threads.get() == self.app.DEFAULTS["threads"]
        assert self.app.fit_margin.get() == 4096

    def test_the_other_button_puts_every_one_back(self):
        self.app.threads.set(30)
        self.app.fit_margin.set(4096)
        self.app.vision.set(True)
        self.app.repetitiveness.set(80)
        self.app.reset_all()
        for name, wanted in self.app.DEFAULTS.items():
            assert self.app.variables()[name].get() == wanted, f"name={name}"

    def test_a_card_that_is_no_longer_there_falls_back_to_a_real_one(self):
        """The file may have been written on another machine, or with a
        second card plugged in."""
        self.app.gpu_text.set("A card that was unplugged")
        self.app.save()
        self.app._load()
        assert self.app.gpu_text.get() in [
            label for label, _mib in self.app._gpu_choices()
        ]

    def test_a_typed_custom_figure_comes_back_after_reload(self):
        label, _mib = self.app._gpu_choices()[-1]
        self.app.gpu_text.set(label)
        self.app._on_gpu_chosen()
        self.app.vram_text.set("31415")
        self.app.save()
        self.app.reset("gpu_text")
        self.app._load()
        assert self.app.gpu_text.get() == self.G.CUSTOM_GPU_LABEL
        assert self.app.vram_text.get() == "31415"

    def test_reset_all_does_not_raise(self):
        label, _mib = self.app._gpu_choices()[-1]
        self.app.gpu_text.set(label)
        self.app._on_gpu_chosen()
        self.app.vram_text.set("12345")
        self.app.reset_all()

    def test_a_damaged_file_is_not_an_error(self):
        self.G.SETTINGS_FILE.write_text("{ not json at all", encoding="utf-8")
        assert self.G.load_settings() == {}
        self.app._load()

    def test_a_value_of_the_wrong_kind_condemns_the_whole_file(self):
        """A slider holding a word raises on every later read, from
        inside a trace where nothing can catch it, and the window then
        prints that error for the rest of its life. One such value takes
        the file with it: the good entries beside it are not used either,
        because half the controls from an old file and half from the
        defaults is a state nobody can explain later."""
        self.app.kv_type.set("q8_0")
        self.app.threads.set(11.0)
        self.app.vision.set(False)
        self.G.SETTINGS_FILE.write_text(
            '{"threads": "not a number", "vision": "yes please", "kv_type": "f16"}',
            encoding="utf-8",
        )
        self.app._load()
        assert not self.G.SETTINGS_FILE.exists()
        assert self.app.kv_type.get() == "q8_0"
        assert self.app.threads.get() == 11.0
        assert self.app.vision.get() is False

    def test_text_that_is_not_json_takes_the_file_with_it(self):
        self.G.SETTINGS_FILE.write_text("{not json at all", encoding="utf-8")
        assert self.G.load_settings() == {}
        assert not self.G.SETTINGS_FILE.exists()

    def test_json_that_is_not_an_object_takes_the_file_with_it(self):
        self.G.SETTINGS_FILE.write_text("[1, 2, 3]", encoding="utf-8")
        assert self.G.load_settings() == {}
        assert not self.G.SETTINGS_FILE.exists()

    def test_an_unknown_key_is_not_a_fault(self):
        """An older or newer file is not a broken file. The window takes
        what it knows and leaves the rest where it is."""
        self.G.SETTINGS_FILE.write_text(
            '{"threads": 12, "something_else": 1}', encoding="utf-8"
        )
        self.app._load()
        assert self.G.SETTINGS_FILE.exists()
        assert self.app.threads.get() == 12

    def test_a_whole_number_in_the_file_still_reaches_a_slider(self):
        """JSON gives 22.0 back as 22, and refusing that would quietly
        forget every slider between one run and the next."""
        self.G.SETTINGS_FILE.write_text('{"threads": 22}', encoding="utf-8")
        self.app._load()
        assert self.app.threads.get() == 22

    def test_no_file_yet_is_not_an_error(self):
        self.G.SETTINGS_FILE.unlink(missing_ok=True)
        assert self.G.load_settings() == {}
        self.app._load()

    def test_a_folder_that_cannot_be_written_to_is_not_an_error(self):
        was = self.G.SETTINGS_FILE
        self.G.SETTINGS_FILE = pathlib.Path("Q:/no/such/place/settings.json")
        try:
            self.app.save()
        finally:
            self.G.SETTINGS_FILE = was

    def test_the_buttons_are_there_beside_the_controls(self):
        """One for every control, and one more for all of them.

        The graphics card name and its memory share one button: picking
        a card already sets the figure, so putting the pair back is one
        action, not two.
        """
        marks = [
            widget
            for widget in self.walk(self.app)
            if isinstance(widget, ttk.Button)
            and self.G.RESET_MARK in str(widget.cget("text"))
        ]
        controls = len(self.app.variables()) - 1
        assert len(marks) == controls + 1

    @staticmethod
    def walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from TestRememberingTheControls.walk(child)


class TestWhatTheNumbersAreWorth:
    """The two warnings, at the foot of the parameter column."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def test_both_warnings_are_still_shown(self):
        from qwen_flash_calculator import measured as M

        assert self.app.disclaimer_label.cget("text") == M.DISCLAIMER
        assert "five directly measured" in self.app.validation_label.cget("text")

    def test_they_sit_inside_the_scrolling_parameter_column(self):
        """Under the bar their height changed with the width of the
        window, and every resize moved the bar down the screen."""
        for label in (self.app.disclaimer_label, self.app.validation_label):
            parent = label.winfo_parent()
            assert parent != str(self.app), f"label={label}"
            assert str(self.app.body_canvas) in parent, f"label={label}"

    def test_they_wrap_to_the_column_and_not_to_the_window(self):
        for label in (self.app.disclaimer_label, self.app.validation_label):
            assert int(label.cget("wraplength")) == self.G.PARAMETER_TEXT_WRAP, (
                f"label={label}"
            )


class TestTheMachineBlock:
    """The card is chosen from a list; the memory is not asked at all."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def test_the_list_ends_with_custom(self):
        """A machine with no matching NVIDIA card still needs a way to
        say what card it has."""
        labels = [label for label, _mib in self.app._gpu_choices()]
        assert labels[-1] == self.G.CUSTOM_GPU_LABEL

    def test_the_labels_carry_no_memory_figure(self):
        """The figure has its own field now, so the name is not doubled
        up with it."""
        for label, _mib in self.app._gpu_choices():
            assert "MiB" not in label

    def test_a_real_card_comes_first_and_is_already_chosen(self):
        choices = self.app._gpu_choices()
        assert choices
        assert self.app.gpu_text.get() == choices[0][0]
        cfg, _problems = self.app.current_config()
        if choices[0][0] == self.G.CUSTOM_GPU_LABEL:
            # No NVIDIA card answered on this machine, so the reference
            # figure stands in until one is typed.
            assert cfg.vram_total_mib == M.REFERENCE_VRAM_MIB
        else:
            assert cfg.vram_total_mib == choices[0][1]

    def test_choosing_a_real_card_fills_the_field_and_locks_it(self):
        first = self.app.gpu_text.get()
        try:
            label, mib = self.app._gpu_choices()[0]
            if label == self.G.CUSTOM_GPU_LABEL:
                pytest.skip("no real card on this machine")
            self.app.gpu_text.set(label)
            self.app._on_gpu_chosen()
            assert self.app.vram_text.get() == str(mib)
            assert str(self.app.vram_entry.cget("state")) == "readonly"
        finally:
            self.app.gpu_text.set(first)
            self.app._on_gpu_chosen()

    def test_choosing_custom_keeps_the_figure_and_unlocks_it(self):
        first = self.app.gpu_text.get()
        first_vram = self.app.vram_text.get()
        try:
            label, _mib = self.app._gpu_choices()[-1]
            self.app.gpu_text.set(label)
            self.app._on_gpu_chosen()
            assert self.app.vram_text.get() == first_vram
            assert str(self.app.vram_entry.cget("state")) == "normal"
            self.app.vram_text.set("20000")
            cfg, _problems = self.app.current_config()
            assert cfg.vram_total_mib == 20000
        finally:
            self.app.gpu_text.set(first)
            self.app._on_gpu_chosen()

    def test_system_memory_is_read_rather_than_typed(self):
        note = self.app._ram_note()
        assert "System memory:" in note
        cfg, _problems = self.app.current_config()
        assert cfg.ram_total_gb > 0
        # The field that used to hold it is gone.
        assert not hasattr(self.app, "ram_entry")

    def test_the_launcher_sits_in_the_results_column(self):
        """It used to be a strip across the bottom of the window."""
        parent = self.app.launcher_text.winfo_parent()
        assert parent != str(self.app)
        assert str(self.app) in parent


class TestTextFollowsTheWidth:
    """A sentence must wrap, not run off the edge of its column."""

    @classmethod
    def setup_class(cls):
        # Shown, not hidden: these tests resize the window and read the
        # width back, and a hidden window reports no new width at all.
        cls.root = shared_root()
        cls.root.deiconify()
        from qwen_flash_calculator import app as G

        cls.G = G
        cls.app = G.Calculator(cls.root)
        cls.app.pack(fill="both", expand=True)

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()
        cls.root.withdraw()

    def at_width(self, width):
        self.root.geometry(f"{width}x900")
        self.root.update_idletasks()
        self.root.update()
        return self.app._results_wrap

    def test_the_results_wrap_narrower_in_a_narrower_window(self):
        wide = self.at_width(2000)
        narrow = self.at_width(1200)
        assert narrow < wide
        assert narrow >= self.G.MINIMUM_WRAP

    def test_every_registered_label_is_given_the_new_width(self):
        wrap = self.at_width(1400)
        for label in self.app._results_labels:
            assert int(label.cget("wraplength")) == wrap

    def test_the_numbers_written_on_every_pass_keep_the_width(self):
        """Two labels are rewritten by recalculate, with their own
        wraplength. They used to be written back at the old fixed width
        and lose the wrapping again on the next keystroke."""
        wrap = self.at_width(1400)
        self.app.recalculate()
        self.root.update_idletasks()
        assert int(self.app.system_label.cget("wraplength")) == wrap

    def test_nothing_wraps_below_the_floor(self):
        assert self.at_width(600) >= self.G.MINIMUM_WRAP


class TestTheBodyScrolls:
    """A short window must hide nothing without saying so.

    Measured on the reference machine: the left column alone asks for 933
    pixels and the whole window asks for 1,405. On a laptop screen the
    shipped-profile buttons were simply cut off, which is how this was
    found.
    """

    @classmethod
    def setup_class(cls):
        # Shown, not hidden, for the same reason as the class above.
        cls.root = shared_root()
        cls.root.deiconify()
        cls.root.geometry("1020x700")
        from qwen_flash_calculator import app as G

        cls.app = G.Calculator(cls.root)
        cls.app.pack(fill="both", expand=True)
        cls.root.update_idletasks()
        cls.root.update()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()
        cls.root.withdraw()

    def test_the_column_is_tall_enough_to_need_scrolling(self):
        """The fact behind the change: the parameters do not fit."""
        box = self.app.body_canvas.bbox("all")
        assert box is not None
        assert box[3] > 500

    def test_only_the_parameter_column_scrolls(self):
        """The results must not move when the parameters do: a slider and
        the number it changes belong on the screen together."""
        canvas = self.app.body_canvas
        assert canvas.cget("yscrollcommand")
        assert not canvas.cget("xscrollcommand")
        bars = [
            child
            for child in canvas.master.winfo_children()
            if child.winfo_class() == "TScrollbar"
        ]
        assert len(bars) == 1
        # The results column is a sibling of the scrolling holder, not a
        # child of it.
        assert self.app.status_label.winfo_parent() not in str(canvas)

    def test_the_view_really_moves_when_the_region_is_larger(self):
        """Wiring proves nothing on its own: this moves the view."""
        canvas = self.app.body_canvas
        original = canvas.cget("scrollregion")
        try:
            canvas.configure(scrollregion=(0, 0, 100, 5000))
            canvas.yview_moveto(1.0)
            self.root.update_idletasks()
            assert canvas.yview()[0] > 0.0
        finally:
            canvas.yview_moveto(0.0)
            canvas.configure(scrollregion=original)

    def test_the_shipped_profile_buttons_are_gone(self):
        """They were removed on request. Nothing should bring them back
        quietly: the three measured configurations still live in
        measured.py and in the documents."""
        found = []

        def walk(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    found.append(str(child.cget("text")))
                walk(child)

        walk(self.app)
        for label, _ctx, _ubatch, _threads in M.CONFIG_PRESETS:
            assert label not in found


class TestTheLiveBlockInTheWindow:
    """The box, the block and what they do to the estimate."""

    @classmethod
    def setup_class(cls):
        cls.root = shared_root()
        from qwen_flash_calculator import app as G
        from qwen_flash_calculator import vram

        cls.G = G
        cls.vram = vram
        cls.app = G.Calculator(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def teardown_class(cls):
        cls.app.destroy()

    def setup_method(self):
        self.app.show_vram.set(False)
        self.app.vram_reading = None
        self.app.vram_error = ""

    def test_an_unticked_box_ignores_a_reading_entirely(self):
        self.app.vram_reading = self.vram.Reading(2075, 16376, "NVML", False)
        cfg, _ = self.app.current_config()
        assert cfg.vram_busy_mib == 0

    def test_a_ticked_box_hands_the_reading_to_the_estimator(self):
        self.app.show_vram.set(True)
        self.app.vram_reading = self.vram.Reading(2075, 16376, "NVML", False)
        cfg, _ = self.app.current_config()
        assert cfg.vram_busy_mib == 2075
        assert cfg.vram_for_us_mib == cfg.vram_total_mib - 2075

    def test_a_reading_larger_than_the_card_is_capped(self):
        self.app.show_vram.set(True)
        self.app.vram_reading = self.vram.Reading(99999, 16376, "NVML", False)
        cfg, _ = self.app.current_config()
        assert cfg.vram_busy_mib == cfg.vram_total_mib

    def test_the_bar_still_adds_up_to_the_whole_card(self):
        """The busy segment is what keeps the bar honest: without it the
        segments would stop short of the capacity line and the eye would
        read the gap as free memory."""
        self.app.show_vram.set(True)
        self.app.vram_reading = self.vram.Reading(2075, 16376, "NVML", False)
        cfg, _ = self.app.current_config()
        out = E.estimate(cfg)
        segments, _tail = self.app._bar_parts(cfg, out)
        names = [name for name, _value in segments]
        assert [
            name for name in names if name.startswith("proc") or name == "busy_rest"
        ], names
        total = sum(value for _name, value in segments)
        assert total == pytest.approx(cfg.vram_total_mib, abs=1.0)

    def test_the_block_names_its_source_and_warns_about_the_server(self):
        self.app.show_vram.set(True)
        self.app.vram_reading = self.vram.Reading(2075, 16376, "NVML", True)
        self.app._update_vram_block()
        assert "2,075 MiB" in self.app.vram_live_label.cget("text")
        note = self.app.vram_live_note.cget("text")
        assert "NVML" in note
        assert "llama-server is on the card" in note

    def test_a_failed_reading_says_so_instead_of_showing_nothing(self):
        self.app.show_vram.set(True)
        self.app.vram_error = "nvml.dll not found"
        self.app._update_vram_block()
        assert "could not be read" in self.app.vram_live_label.cget("text")
        assert "nvml.dll not found" in self.app.vram_live_note.cget("text")

    def test_unticking_stops_the_poll_and_clears_the_block(self):
        self.app.show_vram.set(True)
        self.app._toggle_vram_poll()
        assert self.app._vram_job is not None
        self.app.show_vram.set(False)
        self.app._toggle_vram_poll()
        assert self.app._vram_job is None
        assert self.app.vram_live_label.cget("text") == ""
