"""Checks the estimation model against the measured points.

Run with:  uv run pytest tests/test_estimator.py
"""

import re
import typing
from unittest import mock

import pytest

from qwen_flash_calculator import estimator as E
from qwen_flash_calculator import measured as M


def relative_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured


class TestMeasuredBreakdowns:
    """results.md section 7.2, the four printed CUDA0 breakdowns."""

    def test_key_value_cache_within_4_percent(self):
        # The degraded row is skipped: its printed cache is only the part that
        # stayed on the card after llama.cpp pushed sixteen layers off.
        for (
            ctx,
            ubatch,
            _model,
            kv,
            _compute,
            _layers,
            llama_layers,
        ) in M.MEASURED_BREAKDOWNS:
            if llama_layers < M.LLAMA_LAYER_COUNT:
                continue
            assert relative_error(E.kv_cache_mib(ctx, "q8_0"), kv) < 0.04, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_compute_buffers_within_4_percent(self):
        # This one holds on all four rows, degraded included: the compute
        # buffers are allocated before any layer is moved.
        for ctx, ubatch, _model, _kv, compute, _layers, _ll in M.MEASURED_BREAKDOWNS:
            assert relative_error(E.compute_buffer_mib(ctx, ubatch), compute) < 0.04, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_model_weights_within_4_percent(self):
        for (
            ctx,
            ubatch,
            model,
            _kv,
            _compute,
            layers,
            llama_layers,
        ) in M.MEASURED_BREAKDOWNS:
            if llama_layers < M.LLAMA_LAYER_COUNT:
                continue
            predicted = M.DENSE_AND_ATTENTION_MIB + layers * M.PER_EXPERT_LAYER_MIB
            assert relative_error(predicted, model) < 0.04, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_whole_expert_layer_count_matches(self):
        """At the recommended fit margin of 1024 the count is exact on all four."""
        for ctx, ubatch, _model, _kv, _compute, layers, _ll in M.MEASURED_BREAKDOWNS:
            cfg = E.Config(
                ctx=ctx,
                ubatch=ubatch,
                kv_type="q8_0",
                fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
                vram_total_mib=M.REFERENCE_VRAM_MIB,
            )
            assert E.estimate(cfg).expert_layers == layers, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_only_the_full_window_with_ubatch_2048_is_degraded(self):
        for ctx, ubatch, _m, _k, _c, _layers, llama_layers in M.MEASURED_BREAKDOWNS:
            cfg = E.Config(
                ctx=ctx,
                ubatch=ubatch,
                kv_type="q8_0",
                fit_margin=M.REFERENCE_FIT_MARGIN_MIB,
                vram_total_mib=M.REFERENCE_VRAM_MIB,
            )
            expected = E.DEGRADED if llama_layers < M.LLAMA_LAYER_COUNT else E.OK
            assert E.estimate(cfg).status == expected, f"ctx={ctx}, ubatch={ubatch}"


class TestPromptProcessingGrid:
    """results.md section 8, the prefill half of the 24-run table."""

    def test_every_grid_point_is_returned_exactly(self):
        for ctx, row in M.PP_GRID.items():
            for ubatch, value in zip(M.PP_UBATCH_COLUMNS, row):
                assert E.prompt_processing(ctx, ubatch) == pytest.approx(
                    value, abs=1e-6
                ), f"ctx={ctx}, ubatch={ubatch}"

    def test_between_two_rows_the_answer_sits_between_them(self):
        low = E.prompt_processing(65536, 2048)
        high = E.prompt_processing(98304, 2048)
        middle = E.prompt_processing((65536 + 98304) // 2, 2048)
        assert min(low, high) < middle
        assert middle < max(low, high)

    def test_micro_batch_256_follows_the_tables_own_slope(self):
        """It used to return the 512 value, which is 54% too high.

        Row 163840 runs 341 at micro-batch 512 and 460 at 1024. One step of
        log2 below 512 is therefore 341 - 119 = 222 tokens/s.
        """
        assert E.prompt_processing(163840, 256) == pytest.approx(222.0, abs=1e-6)
        assert E.prompt_processing(163840, 256) < E.prompt_processing(163840, 512)

    def test_micro_batch_8192_keeps_climbing_at_the_top_edge_slope(self):
        """Row 163840 runs 570 at 2048 and 630 at 4096, so 8192 reads 690."""
        assert E.prompt_processing(163840, 8192) == pytest.approx(690.0, abs=1e-6)

    def test_a_window_below_the_grid_follows_the_top_two_rows(self):
        """Rows 65536 and 98304 at micro-batch 512 are 323 and 318."""
        slope = (318 - 323) / (98304 - 65536)
        expected = 323 + slope * (4096 - 65536)
        assert E.prompt_processing(4096, 512) == pytest.approx(expected, abs=1e-6)

    def test_everything_outside_the_grid_is_marked_extrapolated(self):
        assert not E.prompt_processing_is_extrapolated(163840, 512)
        assert E.prompt_processing_is_extrapolated(163840, 256)
        assert E.prompt_processing_is_extrapolated(163840, 8192)
        assert E.prompt_processing_is_extrapolated(4096, 512)

    def test_the_extrapolated_line_never_reaches_zero_or_turns_negative(self):
        assert E.prompt_processing(4096, 1) >= E.PP_FLOOR_TPS


class TestThreadCurve:
    def test_ten_threads_is_the_unit(self):
        assert E.thread_factor(10) == pytest.approx(1.0, abs=1e-9)

    def test_measured_points_are_reproduced(self):
        reference = dict(M.THREAD_SAMPLES)[M.THREAD_REFERENCE]
        for threads, value in M.THREAD_SAMPLES:
            assert E.thread_factor(threads) == pytest.approx(
                value / reference, abs=1e-9
            ), f"threads={threads}"

    def test_between_points_it_interpolates(self):
        middle = E.thread_factor(5)
        assert E.thread_factor(4) < middle
        assert middle < E.thread_factor(6)

    def test_below_four_threads_it_falls_towards_zero(self):
        """The measured curve starts at 4 threads; below it nothing was run.

        Holding the 4-thread value there claimed that one thread is as fast
        as four. The line now runs from the origin to the first measured
        point, so two threads is half of four.
        """
        assert E.thread_factor(2) == pytest.approx(E.thread_factor(4) / 2, abs=1e-9)
        assert E.thread_factor(1) == pytest.approx(E.thread_factor(4) / 4, abs=1e-9)
        assert E.thread_factor(3) < E.thread_factor(4)

    def test_above_the_last_point_it_still_holds(self):
        assert E.thread_factor(64) == pytest.approx(E.thread_factor(32), abs=1e-9)

    def test_below_four_threads_is_marked_extrapolated(self):
        assert E.thread_factor_is_extrapolated(3)
        assert not E.thread_factor_is_extrapolated(4)
        assert not E.thread_factor_is_extrapolated(10)


class TestGenerationBase:
    def test_no_expert_layers_matches_the_ncmoe_48_run(self):
        assert E.base_generation(0) == pytest.approx(16.87, abs=1e-6)

    def test_one_expert_layer_matches_the_ncmoe_47_run(self):
        assert E.base_generation(1) == pytest.approx(17.61, abs=1e-6)

    def test_four_expert_layers_reach_the_measured_19_8(self):
        assert E.base_generation(4) == pytest.approx(19.83, abs=1e-6)

    def test_the_line_is_kept_but_marked_above_four_layers(self):
        """Fitted on 0, 1, 2 and 4 layers only, so 5 and up are marked.

        The line itself does not change: nothing else was measured, so an
        invented curve would be worse than a marked straight one.
        """
        assert not E.base_generation_is_extrapolated(4)
        assert E.base_generation_is_extrapolated(5)
        assert E.base_generation(12) == pytest.approx(
            M.TG_BASE_INTERCEPT + 12 * M.TG_PER_LAYER, abs=1e-7
        )

    def test_the_24_and_48_gb_presets_come_out_extrapolated(self):
        """Both put far more than four expert layers on the card."""
        for _label, vram in M.VRAM_PRESETS:
            if vram < 24576:
                continue
            out = E.estimate(E.Config(ctx=163840, ubatch=512, vram_total_mib=vram))
            assert out.status == E.OK, f"vram={vram}"
            assert out.expert_layers > M.TG_MAX_FITTED_LAYERS, f"vram={vram}"
            assert out.tg_extrapolated, f"vram={vram}"
            assert "expert layers" in out.tg_extrapolation_reason, f"vram={vram}"

    def test_the_measured_card_at_the_default_margin_is_not_extrapolated(self):
        out = E.estimate(E.Config(ctx=163840, ubatch=512))
        assert not out.tg_extrapolated


class TestSpeculation:
    def test_off_is_one(self):
        assert E.speculation_factor(False, 100.0) == pytest.approx(1.0, abs=1e-7)

    def test_new_text_is_one(self):
        assert E.speculation_factor(True, 0.0) == pytest.approx(
            M.SPECULATION_MIN, abs=1e-7
        )

    def test_fully_repeated_text_is_the_top_multiplier(self):
        assert E.speculation_factor(True, 100.0) == pytest.approx(
            M.SPECULATION_MAX, abs=1e-7
        )

    def test_halfway_is_halfway(self):
        assert E.speculation_factor(True, 50.0) == pytest.approx(1.95, abs=1e-7)


class TestDegradedBranch:
    """The one measured run where whole layers were pushed onto the processor."""

    def anchor_config(self):
        return E.Config(
            ctx=M.DEGRADED_ANCHOR_CTX,
            ubatch=M.DEGRADED_ANCHOR_UBATCH,
            kv_type="q8_0",
            threads=M.THREAD_REFERENCE,
            fit_margin=M.REFERENCE_FIT_MARGIN_MIB,
            vision=False,
            vram_total_mib=M.REFERENCE_VRAM_MIB,
        )

    def test_the_anchor_run_is_flagged_degraded(self):
        assert E.estimate(self.anchor_config()).status == E.DEGRADED

    def test_the_anchor_run_reproduces_8_1_tokens_per_second(self):
        out = E.estimate(self.anchor_config())
        assert out.tg_tps == pytest.approx(M.DEGRADED_ANCHOR_TG, abs=1e-6)

    def test_the_anchor_run_reproduces_33_of_49_layers(self):
        out = E.estimate(self.anchor_config())
        assert out.llama_layers == M.DEGRADED_ANCHOR_LLAMA_LAYERS

    def test_the_degraded_split_lands_near_the_printed_breakdown(self):
        # Printed: model 4281 MiB, cache 2828 MiB. The split is an anchored
        # approximation, so the tolerance is wider than the 4% of the
        # healthy formula.
        out = E.estimate(self.anchor_config())
        assert relative_error(out.dense_mib, 4281) < 0.10
        assert relative_error(out.kv_mib, 2828) < 0.10

    def test_a_deeper_shortfall_generates_more_slowly(self):
        shallow = E.degraded_generation(E.REFERENCE_SHORTFALL_MIB * 0.5)
        deep = E.degraded_generation(E.REFERENCE_SHORTFALL_MIB * 2.0)
        assert deep < shallow
        assert shallow < M.TG_BASE_INTERCEPT

    def test_no_shortfall_gives_the_undegraded_base(self):
        assert E.degraded_generation(0.0) == pytest.approx(
            M.TG_BASE_INTERCEPT, abs=1e-7
        )

    def test_it_also_matches_collapsed_cells_it_was_not_fitted_to(self):
        """Two further collapsed cells from the 24-run table, within 10%."""
        # (ctx, ubatch, measured generation) at -fitt 512, 10 threads.
        for ctx, ubatch, measured_tg in [
            (163840, 4096, 6.99),
            (196608, 4096, 4.25),
        ]:
            cfg = _grid_config(ctx, ubatch)
            out = E.estimate(cfg)
            assert out.status == E.DEGRADED, f"ctx={ctx}, ubatch={ubatch}"
            assert relative_error(out.tg_tps, measured_tg) < 0.10, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_the_deepest_shortfall_is_under_estimated_not_over(self):
        """262144 with micro-batch 4096: measured 4.12, the model says ~2.5.

        The anchored straight line keeps steepening past the one point it was
        fitted to, so the far corner of the grid comes out pessimistic. That
        is the safe direction for a fit calculator, and it is recorded here
        so the deviation is not mistaken for a regression.
        """
        out = E.estimate(_grid_config(262144, 4096))
        assert out.status == E.DEGRADED
        assert out.tg_tps < 4.12


def _grid_config(ctx: int, ubatch: int, fit_margin: int | None = None) -> E.Config:
    """The conditions the 24-run table was measured under.

    ``fit_margin`` defaults to the 512 MiB the table was measured with. The
    interface opens at 1024, so the grid is also checked there.
    """
    return E.Config(
        ctx=ctx,
        ubatch=ubatch,
        kv_type="q8_0",
        threads=M.THREAD_REFERENCE,
        fit_margin=(M.REFERENCE_FIT_MARGIN_MIB if fit_margin is None else fit_margin),
        vram_total_mib=M.REFERENCE_VRAM_MIB,
    )


#: The generation half of the 24-run table, results.md section 8.
#: Keys are the micro-batch, values follow M.PP_CTX_ROWS.
MEASURED_GENERATION = {
    512: [19.23, 19.45, 19.99, 19.56, 19.56, 19.84],
    1024: [19.25, 19.10, 19.01, 18.80, 18.99, 17.67],
    2048: [19.09, 18.83, 18.54, 17.28, 17.37, 8.29],
    4096: [18.65, 17.84, 10.54, 6.99, 4.25, 4.12],
}

#: The two cells the model is known to miss, with the reason.
KNOWN_GENERATION_OUTLIERS = {
    # The real fitter kept nothing usable here while the formula still finds
    # 195 MiB free, so the model calls it healthy and reality collapsed.
    (131072, 4096),
    # The far corner: see test_the_deepest_shortfall_is_under_estimated_not_over.
    (262144, 4096),
}


class TestGenerationAgainstTheWholeGrid:
    """Every generation cell of the 24-run table, not only the fitted ones."""

    def test_every_cell_except_the_two_known_outliers_is_within_15_percent(self):
        for ubatch, column in MEASURED_GENERATION.items():
            for ctx, measured_tg in zip(M.PP_CTX_ROWS, column):
                if (ctx, ubatch) in KNOWN_GENERATION_OUTLIERS:
                    continue
                predicted = E.estimate(_grid_config(ctx, ubatch)).tg_tps
                assert relative_error(predicted, measured_tg) < 0.15, (
                    f"ctx={ctx}, ubatch={ubatch}"
                )

    def test_at_the_default_margin_three_cells_miss_and_they_are_named(self):
        """The same grid at the 1024 MiB margin the window opens with.

        The test above is pinned to the 512 MiB margin the table was
        measured at. At the default margin the fitter has 512 MiB less to
        work with, so three cells miss by more than 15%, not two, and each
        is asserted here at the tolerance it actually needs:

        - 262144 with micro-batch 2048: measured 8.29, model 7.04, 15.1%.
          This is the anchor run itself, measured at the other margin.
        - 131072 with micro-batch 4096: measured 10.54, model 14.11, 33.9%.
          The formula still finds room the real fitter did not; at 512 it
          called the same cell healthy, which was worse.
        - 262144 with micro-batch 4096: measured 4.12, model 2.40, 41.7%.
          The anchored line keeps steepening past the point it was fitted
          to, so the far corner comes out pessimistic.
        """
        needed = {
            (262144, 2048): 0.16,
            (131072, 4096): 0.35,
            (262144, 4096): 0.42,
        }
        missed = set()
        for ubatch, column in MEASURED_GENERATION.items():
            for ctx, measured_tg in zip(M.PP_CTX_ROWS, column):
                cfg = _grid_config(ctx, ubatch, M.DEFAULT_FIT_MARGIN_MIB)
                error = relative_error(E.estimate(cfg).tg_tps, measured_tg)
                tolerance = needed.get((ctx, ubatch), 0.15)
                if tolerance > 0.15:
                    missed.add((ctx, ubatch))
                assert error < tolerance, f"ctx={ctx}, ubatch={ubatch}"
        assert missed == set(needed)

    def test_at_the_default_margin_every_other_cell_is_within_11_percent(self):
        for ubatch, column in MEASURED_GENERATION.items():
            for ctx, measured_tg in zip(M.PP_CTX_ROWS, column):
                if (ctx, ubatch) in {(262144, 2048), (131072, 4096), (262144, 4096)}:
                    continue
                cfg = _grid_config(ctx, ubatch, M.DEFAULT_FIT_MARGIN_MIB)
                assert relative_error(E.estimate(cfg).tg_tps, measured_tg) < 0.11, (
                    f"ctx={ctx}, ubatch={ubatch}"
                )

    def test_most_cells_land_within_8_percent(self):
        close = 0
        total = 0
        for ubatch, column in MEASURED_GENERATION.items():
            for ctx, measured_tg in zip(M.PP_CTX_ROWS, column):
                total += 1
                predicted = E.estimate(_grid_config(ctx, ubatch)).tg_tps
                if relative_error(predicted, measured_tg) < 0.08:
                    close += 1
        assert close >= int(total * 0.7)


class TestFailure:
    def test_compute_buffers_larger_than_the_card_will_not_start(self):
        cfg = E.Config(ctx=262144, ubatch=8192, fit_margin=1024, vram_total_mib=16376)
        out = E.estimate(cfg)
        assert out.status == E.FAIL
        assert out.overflow_mib > 0
        assert out.tg_tps == 0.0

    def test_a_failing_run_names_a_smaller_micro_batch(self):
        cfg = E.Config(ctx=262144, ubatch=8192, fit_margin=1024, vram_total_mib=16376)
        out = E.estimate(cfg)
        assert any("Micro-batch" in line for line in out.advice)

    def test_a_bigger_card_rescues_the_same_configuration(self):
        cfg = E.Config(ctx=262144, ubatch=2048, fit_margin=1024, vram_total_mib=49152)
        out = E.estimate(cfg)
        assert out.status == E.OK
        assert out.expert_layers > 10


class TestLevers:
    def test_a_larger_micro_batch_never_leaves_more_room(self):
        previous = None
        for ubatch in M.UBATCH_CHOICES:
            free = E.free_for_experts(E.Config(ctx=163840, ubatch=ubatch))
            if previous is not None:
                assert free <= previous
            previous = free

    def test_f16_costs_exactly_twice_the_q8_0_cache(self):
        assert E.kv_cache_mib(131072, "f16") == pytest.approx(
            2 * E.kv_cache_mib(131072, "q8_0"), abs=1e-7
        )

    def test_vision_takes_900_mib_off_the_expert_budget(self):
        without = E.free_for_experts(E.Config(vision=False))
        with_vision = E.free_for_experts(E.Config(vision=True))
        assert without - with_vision == pytest.approx(M.VISION_PROJECTOR_MIB, abs=1e-7)

    def test_layers_never_exceed_the_block_count(self):
        cfg = E.Config(ctx=4096, ubatch=256, vram_total_mib=98304)
        assert E.estimate(cfg).expert_layers == M.BLOCK_COUNT


class TestSystemMemory:
    def test_layers_on_the_card_come_off_the_system_figure(self):
        few = E.estimate(E.Config(ctx=262144, ubatch=2048, vram_total_mib=16376))
        many = E.estimate(E.Config(ctx=8192, ubatch=512, vram_total_mib=49152))
        assert many.system_gb < few.system_gb

    def test_weights_in_system_memory_stay_between_zero_and_the_published_size(self):
        """The two sources disagree, so the subtraction has to be floored.

        6200 + 48 * 1138 MiB is 59.4 GiB of weights on the card, against the
        54.5 GB published for the same model file. A big enough card
        therefore "holds more than everything", and without the floor the
        panel would print a negative amount of weights in system memory.
        """
        for vram in (12288, 16376, 24576, 49152, 98304, 262144):
            out = E.estimate(E.Config(ctx=8192, ubatch=512, vram_total_mib=vram))
            assert out.weights_system_gb >= 0.0, f"vram={vram}"
            assert out.weights_system_gb <= M.TOTAL_WEIGHTS_GB, f"vram={vram}"

    def test_a_card_that_holds_everything_says_so_instead_of_a_number(self):
        out = E.estimate(E.Config(ctx=8192, ubatch=512, vram_total_mib=98304))
        assert out.expert_layers == M.BLOCK_COUNT
        assert out.all_weights_on_card
        assert out.weights_system_gb == 0.0

    def test_the_measured_card_keeps_most_of_the_weights_in_system_memory(self):
        out = E.estimate(E.Config(ctx=163840, ubatch=512))
        assert not out.all_weights_on_card
        assert out.weights_system_gb <= M.TOTAL_WEIGHTS_GB

    def test_cache_pushed_off_the_card_is_counted_apart_from_the_weights(self):
        out = E.estimate(E.Config(ctx=262144, ubatch=4096, vram_total_mib=16376))
        assert out.status == E.DEGRADED
        assert out.cache_system_gb > 0.0
        assert out.system_gb == pytest.approx(
            out.weights_system_gb + out.cache_system_gb, abs=1e-7
        )

    def test_a_small_machine_is_warned(self):
        out = E.estimate(E.Config(ctx=163840, ubatch=512, ram_total_gb=32.0))
        assert not out.ram_ok
        assert any("System memory" in line for line in out.advice)

    def test_the_measured_machine_is_not_warned(self):
        out = E.estimate(E.Config(ctx=163840, ubatch=512, ram_total_gb=96.0))
        assert out.ram_ok


class TestContextChoices:
    def test_the_stops_start_at_4096_and_end_at_the_native_window(self):
        assert M.CTX_CHOICES[0] == 4096
        assert M.CTX_CHOICES[-1] == M.NATIVE_CONTEXT

    def test_every_stop_above_the_first_is_a_multiple_of_8192(self):
        for value in M.CTX_CHOICES[1:]:
            assert value % M.CTX_STEP == 0


class TestMicroBatchLargerThanTheWindow:
    """A micro-batch above the window is not a configuration."""

    def test_it_is_clamped_to_the_window(self):
        cfg = E.Config(ctx=4096, ubatch=8192)
        assert cfg.ubatch_clamped
        assert cfg.effective_ubatch == 4096
        assert cfg.batch == 8192

    def test_the_estimate_is_the_one_for_the_clamped_configuration(self):
        clamped = E.estimate(E.Config(ctx=4096, ubatch=8192))
        honest = E.estimate(E.Config(ctx=4096, ubatch=4096))
        assert clamped.compute_mib == honest.compute_mib
        assert clamped.expert_layers == honest.expert_layers
        assert clamped.tg_tps == pytest.approx(honest.tg_tps, abs=1e-7)

    def test_it_is_said_in_one_line(self):
        out = E.estimate(E.Config(ctx=4096, ubatch=8192))
        assert len(out.notes) == 1
        assert "larger than the window" in out.notes[0]
        assert "4,096" in out.notes[0]

    def test_nothing_is_said_when_the_micro_batch_fits_the_window(self):
        assert E.estimate(E.Config(ctx=163840, ubatch=512)).notes == []


class TestThreeStates:
    """Three states and no fourth, and the words agree with each of them."""

    def test_the_three_states_are_the_only_ones(self):
        assert {E.OK, E.DEGRADED, E.FAIL} == {
            "fits",
            "spills to processor",
            "does not start",
        }

    def test_a_run_that_loads_with_layers_off_the_card_is_a_spill(self):
        out = E.estimate(_grid_config(262144, 2048))
        assert out.status == E.DEGRADED
        # It loads, so the headline may not say it does not fit.
        assert "Starts" in out.headline
        assert "does not fit" not in out.headline.lower()

    def test_the_layer_counts_in_the_headline_do_not_contradict_each_other(self):
        """0 of 48 expert layers beside 48 of 49 layers used to read as both.

        The expert-layer count and the count llama.cpp prints are different
        counts. In a spill the expert count is zero and the headline says so
        in words, never as a number beside a nearly full 49.
        """
        for ctx, ubatch in [(262144, 2048), (196608, 4096), (163840, 4096)]:
            out = E.estimate(_grid_config(ctx, ubatch))
            assert out.status == E.DEGRADED, f"ctx={ctx}, ubatch={ubatch}"
            assert out.expert_layers == 0, f"ctx={ctx}, ubatch={ubatch}"
            assert "No expert layer" in out.headline, f"ctx={ctx}, ubatch={ubatch}"
            assert f"0 of {M.LLAMA_LAYER_COUNT}" not in out.headline, (
                f"ctx={ctx}, ubatch={ubatch}"
            )

    def test_a_run_that_does_not_start_says_only_that(self):
        out = E.estimate(E.Config(ctx=262144, ubatch=8192))
        assert out.status == E.FAIL
        assert "Does not start" in out.headline


class TestFreeAndOverflowAreNeverBothShown:
    def test_one_of_the_two_is_always_zero(self):
        for ctx in M.CTX_CHOICES[::7]:
            for ubatch in M.UBATCH_CHOICES:
                out = E.estimate(E.Config(ctx=ctx, ubatch=ubatch))
                assert out.free_mib == 0.0 or out.overflow_mib == 0.0, (
                    f"free {out.free_mib} and overflow {out.overflow_mib}, "
                    f"ctx={ctx}, ubatch={ubatch}"
                )

    def test_a_spill_reports_no_free_memory(self):
        out = E.estimate(_grid_config(262144, 2048))
        assert out.status == E.DEGRADED
        assert out.overflow_mib > 0.0
        assert out.free_mib == 0.0


class TestAdviceOrder:
    """Largest saving first, and no hard-coded claim about which is strongest."""

    @staticmethod
    def freed(line: str) -> float | None:
        match = re.search(r"frees ([\d,]+) MiB", line)
        return float(match.group(1).replace(",", "")) if match else None

    def test_the_lines_come_in_order_of_the_memory_they_free(self):
        cases = [
            E.Config(ctx=262144, ubatch=4096, kv_type="f16"),
            E.Config(ctx=262144, ubatch=8192, kv_type="f16", vision=True),
            E.Config(ctx=196608, ubatch=4096, kv_type="f16", fit_margin=2048),
        ]
        for cfg in cases:
            out = E.estimate(cfg)
            amounts = [self.freed(line) for line in out.advice]
            numbered = [a for a in amounts if a is not None]
            assert len(numbered) > 1, f"ctx={cfg.ctx}, ubatch={cfg.ubatch}"
            assert numbered == sorted(numbered, reverse=True), (
                f"ctx={cfg.ctx}, ubatch={cfg.ubatch}"
            )
            # Lines without a figure sit at the end.
            assert amounts == numbered + [None] * (len(amounts) - len(numbered)), (
                f"ctx={cfg.ctx}, ubatch={cfg.ubatch}"
            )

    def test_no_line_claims_to_be_the_strongest_lever(self):
        out = E.estimate(E.Config(ctx=262144, ubatch=4096, kv_type="f16"))
        for line in out.advice:
            assert "strongest" not in line.lower()


class TestAnotherExpertLayerWouldFit:
    """The old line claimed a 49th expert layer on a card already holding 48."""

    def test_a_full_card_is_never_told_another_layer_would_fit(self):
        out = E.estimate(E.Config(ctx=8192, ubatch=512, vram_total_mib=98304))
        assert out.expert_layers == M.BLOCK_COUNT
        assert out.free_mib > M.PER_EXPERT_LAYER_MIB
        for line in out.advice:
            assert "another expert layer would fit" not in line
        assert any("already on the card" in line for line in out.advice)

    def test_below_the_block_count_a_whole_layer_of_free_memory_cannot_happen(self):
        """Which is why the offer only ever made sense on a full card.

        The layer count is a floor division of the free memory by the size of
        one layer, so whatever is left over is always less than one layer.
        """
        for vram in (12288, 16376, 24576, 32768, 49152):
            for ctx in (8192, 65536, 163840, 262144):
                out = E.estimate(E.Config(ctx=ctx, ubatch=512, vram_total_mib=vram))
                if out.status != E.OK or out.expert_layers >= M.BLOCK_COUNT:
                    continue
                assert out.free_mib < M.PER_EXPERT_LAYER_MIB, f"vram={vram}, ctx={ctx}"


class TestImageInputAdvice:
    """Offered only when switching it off changes the outcome."""

    def test_offered_when_it_turns_a_spill_into_a_fit(self):
        cfg = E.Config(ctx=196608, ubatch=2048, vision=True)
        out = E.estimate(cfg)
        assert out.status == E.DEGRADED
        assert E.outcome(E.Config(ctx=196608, ubatch=2048)) == E.OK
        assert any(E.IMAGE_INPUT_LABEL in line for line in out.advice), out.advice

    def test_not_offered_when_the_run_spills_either_way(self):
        cfg = E.Config(ctx=262144, ubatch=2048, kv_type="f16", vision=True)
        out = E.estimate(cfg)
        assert out.status == E.DEGRADED
        assert E.outcome(E.Config(ctx=262144, ubatch=2048, kv_type="f16")) == E.DEGRADED
        assert not any(E.IMAGE_INPUT_LABEL in line for line in out.advice)


class TestReferenceShortfallGuard:
    """A corrected coefficient may not divide by zero or invert the curve."""

    def test_the_real_anchor_still_overflows(self):
        assert E.reference_shortfall_mib() > 0.0

    def test_a_coefficient_that_stops_the_anchor_overflowing_is_caught(self):
        with mock.patch.object(M, "DENSE_AND_ATTENTION_MIB", 100):
            guarded = E.reference_shortfall_mib()
        assert guarded > 0.0

    def test_the_degraded_rules_stay_the_right_way_round(self):
        assert E.degraded_generation(E.REFERENCE_SHORTFALL_MIB * 2) < (
            E.degraded_generation(E.REFERENCE_SHORTFALL_MIB * 0.5)
        )
        assert E.degraded_llama_layers(E.REFERENCE_SHORTFALL_MIB * 2) <= (
            E.degraded_llama_layers(E.REFERENCE_SHORTFALL_MIB * 0.5)
        )


class TestValidationAgainstDirectMeasurement:
    """measured.py VALIDATION_POINTS: five whole runs measured directly on
    this machine, after the model itself was built. Nothing here is fitted
    to them; this only checks how far the existing model lands from them,
    reproducing the figures rather than trusting the task notes that quoted
    them.
    """

    #: Each row's own tolerance: a hair above the error it actually has
    #: today, so a real regression in the model still fails even though the
    #: five rows do not share one bound. The comment names the error this
    #: reproduces, independently recomputed by this test, not copied on
    #: trust from wherever the five measurements were reported.
    ROW_TOLERANCE_PCT: typing.ClassVar[
        dict[tuple[int, int, int], tuple[float, float]]
    ] = {
        (65536, 4096, 24): (6.0, 1.0),  # prefill -5.5%, generation -0.6%
        (131072, 2048, 24): (
            9.5,
            14.5,
        ),  # prefill +9.1%, generation +14.0% (worst pair)
        (131072, 1024, 24): (7.5, 12.5),  # prefill +7.4%, generation +12.3%
        (262144, 512, 24): (1.0, 5.0),  # prefill -0.6%, generation -4.7%
        (262144, 512, 10): (0.5, 8.0),  # prefill 0.0%, generation -7.5%
    }

    @staticmethod
    def _estimate_for(ctx: int, ubatch: int, threads: int) -> E.Estimate:
        cfg = E.Config(
            ctx=ctx,
            ubatch=ubatch,
            threads=threads,
            kv_type="q8_0",
            fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
            vision=False,
            speculation=False,
        )
        return E.estimate(cfg)

    def test_every_row_is_reproduced_within_its_own_tolerance(self):
        for point in M.VALIDATION_POINTS:
            ctx, ubatch, threads, measured_pp, measured_tg, folder = point
            pp_tol, tg_tol = self.ROW_TOLERANCE_PCT[(ctx, ubatch, threads)]
            out = self._estimate_for(ctx, ubatch, threads)
            msg = f"ctx={ctx}, ubatch={ubatch}, threads={threads}, folder={folder}"
            assert relative_error(out.pp_tps, measured_pp) * 100 <= pp_tol, msg
            assert relative_error(out.tg_tps, measured_tg) * 100 <= tg_tol, msg

    def test_the_worst_row_is_131072_ubatch_2048(self):
        """This is the pair the header line and the README call out."""
        out = self._estimate_for(131072, 2048, 24)
        # measured 525 prefill, 20.42 generation (results.md folder ctx130)
        assert out.pp_tps == pytest.approx(573.0, abs=1e-1)
        assert out.tg_tps == pytest.approx(23.2787, abs=1e-3)
        assert relative_error(out.pp_tps, 525) * 100 == pytest.approx(9.14, abs=1e-1)
        assert relative_error(out.tg_tps, 20.42) * 100 == pytest.approx(14.00, abs=1e-1)

    def test_worst_validation_error_rounds_to_9_and_14(self):
        """Both the header line and the README state these two numbers."""
        worst_pp, worst_tg = E.worst_validation_error_pct()
        assert round(worst_pp) == 9
        assert round(worst_tg) == 14


class TestNearestValidationPointTest:
    def test_every_row_is_matched_exactly_by_its_own_configuration(self):
        for point in M.VALIDATION_POINTS:
            ctx, ubatch, threads, _pp, _tg, folder = point
            cfg = E.Config(ctx=ctx, ubatch=ubatch, threads=threads)
            found = E.nearest_validation_point(cfg)
            msg = f"ctx={ctx}, ubatch={ubatch}, threads={threads}"
            assert found is not None, msg
            assert found == point, msg
            assert found[5] == folder, msg

    def test_a_near_miss_on_threads_matches_nothing(self):
        # 131072/2048/24 is a real row; one thread off is not a measurement.
        cfg = E.Config(ctx=131072, ubatch=2048, threads=23)
        assert E.nearest_validation_point(cfg) is None

    def test_a_near_miss_on_the_window_matches_nothing(self):
        # 262144/512/24 is a real row; a neighbouring window is not.
        cfg = E.Config(ctx=253952, ubatch=512, threads=24)
        assert E.nearest_validation_point(cfg) is None

    def test_the_comparison_names_the_measurement_and_the_miss(self):
        cfg = E.Config(
            ctx=131072,
            ubatch=2048,
            threads=24,
            kv_type="q8_0",
            fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
            vision=False,
            speculation=False,
        )
        lines = E.validation_comparison_lines(cfg)
        # No differences from the calibration settings, so there is no
        # third line naming any.
        assert len(lines) == 2
        assert "Measured here" in lines[0]
        assert "prefill 525" in lines[0]
        assert "generation 20.42" in lines[0]
        assert "ctx130" in lines[0]
        assert "prefill 573" in lines[1]
        assert "generation 23.28" in lines[1]
        assert "9% high on prefill" in lines[1]
        assert "14% high on generation" in lines[1]

    def test_the_comparison_says_so_plainly_when_there_is_no_match(self):
        cfg = E.Config(ctx=4096, ubatch=256, threads=7)
        lines = E.validation_comparison_lines(cfg)
        assert len(lines) == 1
        assert "No direct measurement" in lines[0]

    def test_the_first_two_lines_ignore_levers_that_do_not_enter_the_match(self):
        """Match is window, micro-batch and thread count only (see the
        docstring on nearest_validation_point). Image input, speculation,
        fit margin and cache type must not move the measured or modelled
        figures on lines 1 and 2; they can only ever add the third line.
        """
        baseline = E.Config(ctx=131072, ubatch=2048, threads=24)
        with_extras = E.Config(
            ctx=131072,
            ubatch=2048,
            threads=24,
            vision=True,
            speculation=True,
            fit_margin=2048,
            kv_type="f16",
        )
        baseline_lines = E.validation_comparison_lines(baseline)
        extras_lines = E.validation_comparison_lines(with_extras)
        assert baseline_lines == extras_lines[:2]
        assert len(baseline_lines) == 2
        assert len(extras_lines) == 3


class TestValidationSettingDifferences:
    """The third comparison line: only appears with something to name, and
    names exactly the settings that differ from the calibration ones."""

    def test_no_differences_at_the_calibration_settings(self):
        cfg = E.Config(
            ctx=131072,
            ubatch=2048,
            threads=24,
            kv_type="q8_0",
            fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
            vision=False,
            speculation=False,
        )
        assert E.validation_setting_differences(cfg) == []
        assert len(E.validation_comparison_lines(cfg)) == 2

    def test_the_middle_presets_own_settings_are_named_exactly(self):
        # The Middle preset sets image input on and speculation on, and
        # leaves cache and fit margin at the calibration values.
        cfg = E.Config(
            ctx=131072,
            ubatch=2048,
            threads=24,
            kv_type="q8_0",
            fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
            vision=True,
            speculation=True,
        )
        diffs = E.validation_setting_differences(cfg)
        assert diffs == ["image input on", "speculation on"]

        lines = E.validation_comparison_lines(cfg)
        assert len(lines) == 3
        assert "Your settings differ from the measured run" in lines[2]
        assert "image input on" in lines[2]
        assert "speculation on" in lines[2]
        assert "cache" not in lines[2]
        assert "fit margin" not in lines[2]

    def test_cache_and_fit_margin_differences_are_also_named(self):
        cfg = E.Config(
            ctx=131072,
            ubatch=2048,
            threads=24,
            kv_type="f16",
            fit_margin=2048,
            vision=False,
            speculation=False,
        )
        diffs = E.validation_setting_differences(cfg)
        assert diffs == [
            "cache f16 instead of q8_0",
            f"fit margin 2,048 MiB instead of {M.DEFAULT_FIT_MARGIN_MIB:,} MiB",
        ]


class TestConfigPresetsData:
    """measured.py CONFIG_PRESETS: the three shipped profiles."""

    def test_the_three_profiles_match_the_shipped_windows(self):
        by_label = {
            label: (ctx, ubatch, threads)
            for label, ctx, ubatch, threads in M.CONFIG_PRESETS
        }
        assert by_label["Long context"] == (262144, 512, 24)
        assert by_label["Middle"] == (131072, 2048, 24)
        assert by_label["Fast prompts"] == (65536, 4096, 24)

    def test_every_profile_lands_on_a_validation_point(self):
        """A shipped profile that did not correspond to a measurement would
        make the "measured on this machine" line lie by omission."""
        validated = {
            (ctx, ubatch, threads) for ctx, ubatch, threads, *_ in M.VALIDATION_POINTS
        }
        for label, ctx, ubatch, threads in M.CONFIG_PRESETS:
            assert (ctx, ubatch, threads) in validated, f"profile={label}"
