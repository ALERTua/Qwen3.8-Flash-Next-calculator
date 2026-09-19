"""The estimation model. No interface code lives here.

Given a configuration (window, micro-batch, cache type, threads, fit margin,
vision, speculation, and the machine's memory), this module returns a video
memory breakdown, an expert-layer count, a system memory figure, two speed
estimates and a plain-language line saying what to change.

All coefficients come from ``measured.py``. This module only combines them.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from . import measured as M

#: The three states, and the only three. Every line in the panel has to
#: agree with the one that is current.
OK = "fits"
DEGRADED = "spills to processor"
FAIL = "does not start"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """One point in the lever space. Every field is a control in the window."""

    ctx: int = 163840
    ubatch: int = 512
    kv_type: str = "q8_0"
    threads: int = 10
    fit_margin: int = M.DEFAULT_FIT_MARGIN_MIB
    vision: bool = False
    speculation: bool = False
    repetitiveness: float = 0.0  # percent, 0 to 100
    vram_total_mib: int = M.REFERENCE_VRAM_MIB
    ram_total_gb: float = M.REFERENCE_RAM_GB

    #: Video memory other programs already hold, MiB. Zero unless the
    #: window is reading the card live. It is kept apart from
    #: ``vram_total_mib`` so that the card keeps its real size everywhere
    #: the size is printed, while every capacity decision uses what is
    #: actually left.
    vram_busy_mib: int = 0

    @property
    def vram_for_us_mib(self) -> int:
        """Card memory this run may actually take."""
        return max(0, self.vram_total_mib - self.vram_busy_mib)

    @property
    def effective_ubatch(self) -> int:
        """The micro-batch llama.cpp can actually use.

        A micro-batch larger than the window is not a configuration: the run
        never has more tokens in flight than the window holds, so llama.cpp
        works on at most ``ctx`` tokens at a time. Every memory figure is
        computed from this clamped value, not from the slider.
        """
        return min(self.ubatch, self.ctx)

    @property
    def ubatch_clamped(self) -> bool:
        return self.ubatch > self.ctx

    @property
    def batch(self) -> int:
        """llama.cpp is always driven here with a batch of twice the micro-batch."""
        return self.effective_ubatch * 2


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class Estimate:
    """Everything the window shows for one configuration."""

    status: str = OK

    # Video memory, MiB. These are the amounts actually resident on the card.
    dense_mib: float = 0.0
    expert_mib: float = 0.0
    kv_mib: float = 0.0
    compute_mib: float = 0.0
    vision_mib: float = 0.0
    margin_mib: float = 0.0
    free_mib: float = 0.0

    # How much the configuration asks for beyond what the card can give.
    # Zero unless the status is degraded or fail.
    overflow_mib: float = 0.0

    # What the ideal (nothing pushed off) breakdown would have been. The bar
    # draws this when the card cannot hold it, so the shortfall is visible.
    wanted_dense_mib: float = 0.0
    wanted_kv_mib: float = 0.0

    # Layers. Two different counts, and they are never mixed in one
    # sentence: expert_layers counts the blocks whose expert weights are on
    # the card, out of 48; llama_layers is what llama.cpp itself prints,
    # the 48 blocks plus the output layer, out of 49.
    expert_layers: int = 0
    expert_layers_max: int = M.BLOCK_COUNT
    llama_layers: int = M.LLAMA_LAYER_COUNT

    # Speed, tokens per second
    pp_tps: float = 0.0
    tg_tps: float = 0.0
    tg_before_threads: float = 0.0
    thread_factor: float = 1.0
    speculation_factor: float = 1.0

    # True when the figure above is read off a straight line outside the
    # range that line was fitted on. The number is still the best one
    # available, but it is not backed by a measurement.
    tg_extrapolated: bool = False
    pp_extrapolated: bool = False
    tg_extrapolation_reason: str = ""
    pp_extrapolation_reason: str = ""

    # The micro-batch actually used, after clamping it to the window.
    effective_ubatch: int = 0
    ubatch_clamped: bool = False

    # System memory, GB. system_gb is the sum of the two below it.
    system_gb: float = 0.0
    weights_system_gb: float = 0.0
    cache_system_gb: float = 0.0
    ram_ok: bool = True

    # True when the card holds every weight, so there is no weight left to
    # keep in system memory at all.
    all_weights_on_card: bool = False

    # Plain language
    headline: str = ""
    #: Statements of fact about the configuration itself, not advice.
    notes: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory pieces
# ---------------------------------------------------------------------------


def kv_cache_mib(ctx: int, kv_type: str) -> float:
    """Key-value cache on the card, MiB, for a full window."""
    return M.KV_MIB_PER_TOKEN[kv_type] * ctx


def compute_buffer_mib(ctx: int, ubatch: int) -> float:
    """Compute buffers grow with both the micro-batch and the window."""
    return M.COMPUTE_MIB_PER_UBATCH_TOKEN * ubatch * ctx


def vision_mib(vision: bool) -> float:
    return M.VISION_PROJECTOR_MIB if vision else 0.0


def free_for_experts(cfg: Config) -> float:
    """MiB left for expert weights. Negative means the run is in trouble."""
    return (
        cfg.vram_for_us_mib
        - cfg.fit_margin
        - M.DENSE_AND_ATTENTION_MIB
        - kv_cache_mib(cfg.ctx, cfg.kv_type)
        - compute_buffer_mib(cfg.ctx, cfg.effective_ubatch)
        - vision_mib(cfg.vision)
    )


def expert_layers_on_gpu(cfg: Config) -> int:
    free = free_for_experts(cfg)
    return max(0, min(M.BLOCK_COUNT, math.floor(free / M.PER_EXPERT_LAYER_MIB)))


def can_load(cfg: Config) -> bool:
    """False when the compute buffers alone leave no room for weights or cache.

    With ``--fit`` llama.cpp will push whole layers onto the processor rather
    than fail, so a run only truly refuses to start when there is nothing left
    on the card after the margin, the projector and the compute buffers.
    """
    left = (
        cfg.vram_for_us_mib
        - cfg.fit_margin
        - vision_mib(cfg.vision)
        - compute_buffer_mib(cfg.ctx, cfg.effective_ubatch)
    )
    return left > 0


# ---------------------------------------------------------------------------
# The degraded branch, anchored on one measured run
# ---------------------------------------------------------------------------


def reference_shortfall_mib() -> float:
    """The shortfall of the one measured degraded run, MiB.

    Window 262144, micro-batch 2048, q8_0, no vision, fit margin 512, on the
    16376 MiB card. That run kept 33 of 49 layers and generated 8.1 t/s.
    Every degraded estimate is scaled against this point.
    """
    anchor = Config(
        ctx=M.DEGRADED_ANCHOR_CTX,
        ubatch=M.DEGRADED_ANCHOR_UBATCH,
        kv_type="q8_0",
        fit_margin=M.REFERENCE_FIT_MARGIN_MIB,
        vision=False,
        vram_total_mib=M.REFERENCE_VRAM_MIB,
    )
    shortfall = -free_for_experts(anchor)
    # Every degraded rule divides by this number and reads its sign as the
    # direction of the curve. If a coefficient is ever corrected so far that
    # the anchor run stops overflowing, the division would fail or the curve
    # would run backwards, so the value is floored at one whole MiB.
    return max(1.0, shortfall)


#: Computed once; it depends only on the coefficients, never on user input.
REFERENCE_SHORTFALL_MIB = reference_shortfall_mib()


def degraded_llama_layers(shortfall: float) -> int:
    """How many of the 49 layers llama.cpp keeps on the card.

    Linear from 49 at no shortfall down to the measured 33 at the reference
    shortfall, and on down to zero past it.
    """
    lost_at_anchor = M.LLAMA_LAYER_COUNT - M.DEGRADED_ANCHOR_LLAMA_LAYERS
    lost = lost_at_anchor * shortfall / REFERENCE_SHORTFALL_MIB
    return max(0, min(M.LLAMA_LAYER_COUNT, round(M.LLAMA_LAYER_COUNT - lost)))


def degraded_generation(shortfall: float) -> float:
    """Generation speed once whole layers are pushed onto the processor.

    The slowdown factor grows in proportion to the shortfall: it is 1.0 when
    nothing overflows and reaches 16.87 / 8.1 at the reference shortfall.
    """
    base = M.TG_BASE_INTERCEPT
    anchor_slowdown = base / M.DEGRADED_ANCHOR_TG
    slowdown = 1.0 + (anchor_slowdown - 1.0) * shortfall / REFERENCE_SHORTFALL_MIB
    return base / max(slowdown, 1.0)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def base_generation(layers: int) -> float:
    """base(L) = 16.87 t/s + 0.74 t/s per expert layer * L, 10 threads."""
    return M.TG_BASE_INTERCEPT + M.TG_PER_LAYER * layers


def base_generation_is_extrapolated(layers: int) -> bool:
    """True above the largest layer count the straight line was fitted on.

    The line was fitted on 0, 1, 2 and 4 expert layers only. Nothing was
    measured above 4, so the line is kept exactly as it is and the answer is
    marked instead of being replaced by an invented curve.
    """
    return layers > M.TG_MAX_FITTED_LAYERS


def thread_factor(threads: int) -> float:
    """Measured thread curve, normalised so that 10 threads gives 1.00.

    Interpolates between the measured points. The curve starts at 4 threads;
    below that nothing was measured, so the value falls in a straight line
    towards zero threads instead of holding the 4-thread value, which would
    claim that one thread is as fast as four. Above the last measured point
    the end value is held.

    The curve is known to overstate the gain at a working depth, and
    damping it is written up beside THREAD_DEPTH_DAMPING in measured.py.
    It is not applied: the whole model, including the five validation
    runs, rests on the same measurements, and damping this one factor
    alone moved the worst validation error from 14 to 19 per cent.
    """
    samples = M.THREAD_SAMPLES
    reference = _interpolate_samples(samples, M.THREAD_REFERENCE)
    return _interpolate_samples(samples, threads) / reference


def thread_factor_is_extrapolated(threads: int) -> bool:
    """True below the first measured thread count."""
    return threads < M.THREAD_SAMPLES[0][0]


def _interpolate_samples(samples: list[tuple[int, float]], x: float) -> float:
    first_x, first_y = samples[0]
    if x <= first_x:
        # Straight line from the origin to the first measured point.
        if first_x <= 0:
            return first_y
        return first_y * max(0.0, x) / first_x
    if x >= samples[-1][0]:
        return samples[-1][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(samples):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return samples[-1][1]


def speculation_factor(on: bool, repetitiveness: float) -> float:
    """1.00 on text the model has not seen, up to 2.9 on repeated text."""
    if not on:
        return 1.0
    share = max(0.0, min(100.0, repetitiveness)) / 100.0
    return M.SPECULATION_MIN + (M.SPECULATION_MAX - M.SPECULATION_MIN) * share


#: Below this the extrapolated prompt-processing line is not allowed to go.
#: A straight line carried far enough outside the grid would reach zero and
#: then turn negative, which is not a speed.
PP_FLOOR_TPS = 1.0


def prompt_processing(ctx: int, ubatch: int) -> float:
    """Tokens per second, read off the measured grid.

    Two-way interpolation: linear in the window, linear in log2 of the
    micro-batch. Outside the grid the line carries on with the slope of the
    nearest measured edge, so a micro-batch of 256 is not reported at the
    512 value. Anything outside is marked; see
    ``prompt_processing_is_extrapolated``.
    """
    return prompt_processing_detail(ctx, ubatch)[0]


def prompt_processing_is_extrapolated(ctx: int, ubatch: int) -> bool:
    """True when the window or the micro-batch sits outside the measured grid."""
    return prompt_processing_detail(ctx, ubatch)[1]


def prompt_processing_detail(ctx: int, ubatch: int) -> tuple[float, str]:
    """The speed, and a plain reason string that is empty when nothing is
    extrapolated."""
    rows = M.PP_CTX_ROWS
    cols = M.PP_UBATCH_COLUMNS
    ub_log = math.log2(max(1, ubatch))

    def row_value(row_ctx: int) -> float:
        values = M.PP_GRID[row_ctx]
        pairs = [(math.log2(c), v) for c, v in zip(cols, values)]
        return _line_through(pairs, ub_log)

    row_pairs = [(float(row_ctx), row_value(row_ctx)) for row_ctx in rows]
    value = max(PP_FLOOR_TPS, _line_through(row_pairs, float(ctx)))

    outside = []
    if ubatch < cols[0] or ubatch > cols[-1]:
        outside.append(
            f"micro-batch {ubatch:,} is outside the measured "
            f"{cols[0]:,} to {cols[-1]:,}"
        )
    if ctx < rows[0] or ctx > rows[-1]:
        outside.append(
            f"window {ctx:,} is outside the measured {rows[0]:,} to {rows[-1]:,}"
        )
    return value, "; ".join(outside)


def _line_through(pairs: list[tuple[float, float]], x: float) -> float:
    """Linear between the points, and along the end slope outside them.

    Holding the end value outside the grid was the old behaviour and it was
    wrong by 54% at micro-batch 256: the table's own slope from 512 to 1024
    says 222 tokens/s there, not the 341 of the 512 column.
    """
    if len(pairs) == 1:
        return pairs[0][1]
    if x < pairs[0][0]:
        (x0, y0), (x1, y1) = pairs[0], pairs[1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if x > pairs[-1][0]:
        (x0, y0), (x1, y1) = pairs[-2], pairs[-1]
        return y1 + (y1 - y0) * (x - x1) / (x1 - x0)
    for (x0, y0), (x1, y1) in itertools.pairwise(pairs):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pairs[-1][1]


# ---------------------------------------------------------------------------
# The whole estimate
# ---------------------------------------------------------------------------


def estimate(cfg: Config) -> Estimate:
    out = Estimate()

    ubatch = cfg.effective_ubatch
    out.effective_ubatch = ubatch
    out.ubatch_clamped = cfg.ubatch_clamped

    wanted_kv = kv_cache_mib(cfg.ctx, cfg.kv_type)
    compute = compute_buffer_mib(cfg.ctx, ubatch)
    vis = vision_mib(cfg.vision)
    free = free_for_experts(cfg)

    out.margin_mib = float(cfg.fit_margin)
    out.compute_mib = compute
    out.vision_mib = vis
    out.wanted_dense_mib = float(M.DENSE_AND_ATTENTION_MIB)
    out.wanted_kv_mib = wanted_kv
    out.pp_tps, out.pp_extrapolation_reason = prompt_processing_detail(cfg.ctx, ubatch)
    out.pp_extrapolated = bool(out.pp_extrapolation_reason)
    out.thread_factor = thread_factor(cfg.threads)
    out.speculation_factor = speculation_factor(cfg.speculation, cfg.repetitiveness)

    thread_reason = ""
    if thread_factor_is_extrapolated(cfg.threads):
        thread_reason = (
            f"the thread curve starts at {M.THREAD_SAMPLES[0][0]} threads, "
            f"so {cfg.threads} is carried down towards zero"
        )

    if not can_load(cfg):
        out.status = FAIL
        out.dense_mib = 0.0
        out.expert_mib = 0.0
        out.kv_mib = 0.0
        out.expert_layers = 0
        out.llama_layers = 0
        # The compute buffers alone do not fit beside the margin.
        out.overflow_mib = -(cfg.vram_for_us_mib - cfg.fit_margin - vis - compute)
        out.free_mib = 0.0
        out.tg_before_threads = 0.0
        out.tg_tps = 0.0
        out.pp_tps = 0.0
        out.pp_extrapolated = False
        out.pp_extrapolation_reason = ""
        _write_system_memory(out, wanted_kv)
        out.ram_ok = _ram_ok(out.system_gb, cfg.ram_total_gb)
        _write_words(cfg, out)
        return out

    if free < 0:
        # llama.cpp starts pushing whole layers, attention included, onto the
        # processor. Everything below is anchored on the single measured run.
        out.status = DEGRADED
        shortfall = -free
        out.overflow_mib = shortfall
        out.llama_layers = degraded_llama_layers(shortfall)
        resident = out.llama_layers / M.LLAMA_LAYER_COUNT
        out.dense_mib = M.DENSE_AND_ATTENTION_MIB * resident
        out.kv_mib = wanted_kv * resident
        out.expert_layers = 0
        out.expert_mib = 0.0
        # Free memory and an overflow are never shown together. The card is
        # already turning work away, so whatever the anchored split leaves
        # unplaced is not room the user can spend.
        out.free_mib = 0.0
        out.tg_before_threads = degraded_generation(shortfall)
        out.tg_extrapolation_reason = thread_reason
    else:
        out.status = OK
        out.llama_layers = M.LLAMA_LAYER_COUNT
        out.expert_layers = expert_layers_on_gpu(cfg)
        out.dense_mib = float(M.DENSE_AND_ATTENTION_MIB)
        out.kv_mib = wanted_kv
        out.expert_mib = out.expert_layers * M.PER_EXPERT_LAYER_MIB
        out.free_mib = free - out.expert_mib
        out.tg_before_threads = base_generation(out.expert_layers)
        reasons = []
        if base_generation_is_extrapolated(out.expert_layers):
            reasons.append(
                f"the generation line was fitted on at most "
                f"{M.TG_MAX_FITTED_LAYERS} expert layers, and this run puts "
                f"{out.expert_layers} on the card"
            )
        if thread_reason:
            reasons.append(thread_reason)
        out.tg_extrapolation_reason = "; ".join(reasons)

    out.tg_extrapolated = bool(out.tg_extrapolation_reason)
    out.tg_tps = out.tg_before_threads * out.thread_factor * out.speculation_factor

    _write_system_memory(out, wanted_kv)
    out.ram_ok = _ram_ok(out.system_gb, cfg.ram_total_gb)

    _write_words(cfg, out)
    return out


def _write_system_memory(out: Estimate, wanted_kv: float) -> None:
    """Weights that do not fit on the card, plus any cache pushed off it.

    The subtraction is floored at zero. The published total weight size and
    the video-memory coefficients come from different sources and disagree
    by about five gigabytes, so a large enough card can hold more weight
    than the published total. Without the floor the panel would print a
    negative amount of weights in system memory, which cannot happen.
    """
    weights_on_card_gb = (out.dense_mib + out.expert_mib) / 1024.0
    out.weights_system_gb = max(0.0, M.TOTAL_WEIGHTS_GB - weights_on_card_gb)
    out.all_weights_on_card = out.weights_system_gb <= 0.0
    out.cache_system_gb = max(0.0, wanted_kv - out.kv_mib) / 1024.0
    out.system_gb = out.weights_system_gb + out.cache_system_gb


def _ram_ok(system_gb: float, ram_total_gb: float) -> bool:
    return system_gb + M.RAM_HEADROOM_GB <= ram_total_gb


# ---------------------------------------------------------------------------
# Validation against direct measurements
# ---------------------------------------------------------------------------
# M.VALIDATION_POINTS holds five whole configurations measured directly,
# after the model above was already built. Nothing here is fitted to them
# and nothing interpolates between them: a configuration either lands
# exactly on one, or there is nothing to compare it to.


def nearest_validation_point(cfg: Config) -> tuple | None:
    """The validation row measured at this exact configuration, or ``None``.

    "Match" means the same window, the same micro-batch and the same thread
    count, nothing else. Five single points are not a surface, so a near
    miss is not interpolated; it is treated the same as no measurement.
    """
    for point in M.VALIDATION_POINTS:
        ctx, ubatch, threads, _pp, _tg, _folder = point
        if ctx == cfg.ctx and ubatch == cfg.ubatch and threads == cfg.threads:
            return point
    return None


def _validation_settings(ctx: int, ubatch: int, threads: int) -> Config:
    """The configuration a validation row was measured under: q8_0 cache,
    the default 1,024 MiB fit margin, speculation off, vision off, on the
    reference machine.

    "Match" only ever looks at window, micro-batch and thread count, so
    cache type, fit margin, speculation and vision play no part in it
    either. The comparison is always read off this fixed configuration,
    never off whatever the live estimate happens to be under those other
    levers -- otherwise switching image input on, which the shipped
    profiles do, would silently change a number that is supposed to answer
    one fixed question: how good is the model at this window, this
    micro-batch and this thread count.
    """
    return Config(
        ctx=ctx,
        ubatch=ubatch,
        threads=threads,
        kv_type="q8_0",
        fit_margin=M.DEFAULT_FIT_MARGIN_MIB,
        vision=False,
        speculation=False,
    )


def validation_error_pct(point: tuple) -> tuple[float, float]:
    """(prefill error %, generation error %) of the model against one
    measured validation row, positive when the estimate reads high.

    Reads the row's window, micro-batch and thread count back through the
    estimator at the settings the validation session held fixed throughout.
    """
    ctx, ubatch, threads, measured_pp, measured_tg, _folder = point
    out = estimate(_validation_settings(ctx, ubatch, threads))
    pp_error = (out.pp_tps - measured_pp) / measured_pp * 100.0
    tg_error = (out.tg_tps - measured_tg) / measured_tg * 100.0
    return pp_error, tg_error


def worst_validation_error_pct() -> tuple[float, float]:
    """The worst absolute prefill and generation error across all five
    validation rows, each read back at the settings it was measured with."""
    worst_pp = 0.0
    worst_tg = 0.0
    for point in M.VALIDATION_POINTS:
        pp_error, tg_error = validation_error_pct(point)
        worst_pp = max(worst_pp, abs(pp_error))
        worst_tg = max(worst_tg, abs(tg_error))
    return worst_pp, worst_tg


def _error_words(error_pct: float) -> str:
    """ "9% high", "14% low", or "spot on" once the rounded size is zero."""
    rounded = round(abs(error_pct))
    if rounded == 0:
        return "spot on"
    direction = "high" if error_pct > 0 else "low"
    return f"{rounded:.0f}% {direction}"


def validation_setting_differences(cfg: Config) -> list[str]:
    """Plain-language names of the settings where ``cfg`` differs from the
    fixed calibration configuration a validation row is read back at: q8_0
    cache, the default 1,024 MiB fit margin, no image input, no speculation.

    Used only to say, next to a comparison, that the number just above the
    bar belongs to a different configuration than the one being compared.
    """
    calibration = _validation_settings(cfg.ctx, cfg.ubatch, cfg.threads)
    diffs = []
    if cfg.kv_type != calibration.kv_type:
        diffs.append(f"cache {cfg.kv_type} instead of {calibration.kv_type}")
    if cfg.fit_margin != calibration.fit_margin:
        diffs.append(
            f"fit margin {cfg.fit_margin:,} MiB instead of "
            f"{calibration.fit_margin:,} MiB"
        )
    if cfg.vision != calibration.vision:
        diffs.append("image input on" if cfg.vision else "image input off")
    if cfg.speculation != calibration.speculation:
        diffs.append("speculation on" if cfg.speculation else "speculation off")
    return diffs


def validation_comparison_lines(cfg: Config) -> list[str]:
    """Up to three lines shown next to the two estimated speeds.

    1. What was measured, and under exactly which settings.
    2. What the model says for that same configuration, and by how much it
       misses.
    3. Only when the live configuration differs from the measured one: which
       settings differ, so the number shown above the bar is never mistaken
       for the one being compared.

    When the configuration does not land exactly on a measured point, this
    is one line saying so in plain words instead of disappearing, so "no
    measurement" and "the estimate matches" are never the same silence.
    """
    point = nearest_validation_point(cfg)
    if point is None:
        return [
            (
                "No direct measurement was taken at this exact window, "
                "micro-batch and thread count."
            )
        ]
    ctx, ubatch, threads, measured_pp, measured_tg, folder = point
    calibration = _validation_settings(ctx, ubatch, threads)
    out = estimate(calibration)
    pp_error = (out.pp_tps - measured_pp) / measured_pp * 100.0
    tg_error = (out.tg_tps - measured_tg) / measured_tg * 100.0

    lines = [
        (
            f"Measured here ({calibration.kv_type} cache, fit margin "
            f"{calibration.fit_margin:,}, no image input, no speculation): "
            f"prefill {measured_pp:,.0f}, generation {measured_tg:.2f} "
            f"— folder {folder}."
        ),
        (
            f"The model puts that configuration at prefill {out.pp_tps:,.0f} "
            f"and generation {out.tg_tps:.2f}, which is "
            f"{_error_words(pp_error)} on prefill and "
            f"{_error_words(tg_error)} on generation."
        ),
    ]

    diffs = validation_setting_differences(cfg)
    if diffs:
        lines.append(
            "Your settings differ from the measured run: "
            + ", ".join(diffs)
            + ". The estimate above the bar is for your "
            "settings, not for the measured run."
        )
    return lines


# ---------------------------------------------------------------------------
# Which lever to reduce
# ---------------------------------------------------------------------------


def outcome(cfg: Config) -> str:
    """The state a configuration lands in, without the words around it."""
    if not can_load(cfg):
        return FAIL
    return OK if free_for_experts(cfg) >= 0 else DEGRADED


#: Best first. Used only to ask whether a change improves the outcome.
_OUTCOME_ORDER = [OK, DEGRADED, FAIL]


def outcome_improves(before: str, after: str) -> bool:
    return _OUTCOME_ORDER.index(after) < _OUTCOME_ORDER.index(before)


def _with(cfg: Config, **changes) -> Config:
    return Config(**{**cfg.__dict__, **changes})


def largest_ubatch_that_fits(cfg: Config) -> int | None:
    """The biggest micro-batch from the ladder that still keeps every layer on."""
    best = None
    for ub in M.UBATCH_CHOICES:
        if outcome(_with(cfg, ubatch=ub)) == OK:
            best = ub
    return best


def largest_ctx_that_fits(cfg: Config) -> int | None:
    """The biggest window from the slider stops that still fits this micro-batch."""
    best = None
    for ctx in M.CTX_CHOICES:
        if outcome(_with(cfg, ctx=ctx)) == OK:
            best = ctx
    return best


#: One wording for the image-input switch, shared by the checkbox and the
#: advice line so that the two can never drift apart.
IMAGE_INPUT_LABEL = "image input (the vision projector)"


def _write_words(cfg: Config, out: Estimate) -> None:
    """Fill in the headline, the notes and the advice lines.

    Every line here agrees with ``out.status``. There are three states and no
    fourth: it fits, it spills to the processor, or it does not start.
    """
    notes: list[str] = []
    # (MiB this line would free, text). A line with no figure carries -1 and
    # keeps its place at the end.
    scored: list[tuple[float, str]] = []

    if out.ubatch_clamped:
        notes.append(
            f"Micro-batch {cfg.ubatch:,} is larger than the window "
            f"{cfg.ctx:,} tokens. llama.cpp never has more tokens in flight "
            f"than the window holds, so everything below is the estimate for "
            f"a micro-batch of {out.effective_ubatch:,} tokens."
        )

    if out.status == FAIL:
        out.headline = (
            f"Does not start. The compute buffers alone need "
            f"{out.compute_mib:,.0f} MiB, which is more than the card has "
            f"left after the {cfg.fit_margin:,} MiB fit margin."
        )
    elif out.status == DEGRADED:
        if out.llama_layers == 0:
            # --fit does not refuse: it moves everything onto the processor
            # and the server still starts.
            kept = (
                f"llama.cpp keeps none of the {M.LLAMA_LAYER_COUNT} layers "
                f"it counts on the card, so every layer runs on the processor"
            )
        else:
            kept = (
                f"llama.cpp keeps only about {out.llama_layers} of the "
                f"{M.LLAMA_LAYER_COUNT} layers it counts on the card"
            )
        out.headline = (
            f"Starts, but spills to the processor: short by "
            f"{out.overflow_mib:,.0f} MiB. No expert layer keeps its expert "
            f"weights on the card, and {kept}. Generation collapses."
        )
    else:
        layer_word = (
            "layer keeps its" if out.expert_layers == 1 else "layers keep their"
        )
        out.headline = (
            f"Fits. {out.expert_layers} of {out.expert_layers_max} expert "
            f"{layer_word} expert weights on the card, "
            f"{out.free_mib:,.0f} MiB free."
        )

    if out.status in (FAIL, DEGRADED):
        ub = largest_ubatch_that_fits(cfg)
        if ub is not None and ub < out.effective_ubatch:
            saved = compute_buffer_mib(
                cfg.ctx, out.effective_ubatch
            ) - compute_buffer_mib(cfg.ctx, ub)
            scored.append(
                (
                    saved,
                    (
                        f"Micro-batch {out.effective_ubatch:,} -> {ub:,} tokens "
                        f"frees {saved:,.0f} MiB of compute buffers."
                    ),
                )
            )
        ctx = largest_ctx_that_fits(cfg)
        if ctx is not None and ctx < cfg.ctx:
            saved = (
                compute_buffer_mib(cfg.ctx, out.effective_ubatch)
                + kv_cache_mib(cfg.ctx, cfg.kv_type)
            ) - (
                compute_buffer_mib(ctx, min(cfg.ubatch, ctx))
                + kv_cache_mib(ctx, cfg.kv_type)
            )
            scored.append(
                (
                    saved,
                    (
                        f"Window {cfg.ctx:,} -> {ctx:,} tokens frees {saved:,.0f} MiB "
                        f"of key-value cache and compute buffers at this micro-batch."
                    ),
                )
            )
        if cfg.kv_type == "f16":
            saved = kv_cache_mib(cfg.ctx, "f16") - kv_cache_mib(cfg.ctx, "q8_0")
            scored.append(
                (
                    saved,
                    (
                        f"Key-value cache 16-bit f16 -> 8-bit q8_0, half the memory, "
                        f"frees {saved:,.0f} MiB."
                    ),
                )
            )
        if cfg.vision and outcome_improves(
            out.status, outcome(_with(cfg, vision=False))
        ):
            # Offered only when switching it off changes the outcome. A line
            # that frees memory and leaves the run in the same state is noise.
            scored.append(
                (
                    float(M.VISION_PROJECTOR_MIB),
                    (
                        f"Switch {IMAGE_INPUT_LABEL} off, which frees "
                        f"{M.VISION_PROJECTOR_MIB:,} MiB."
                    ),
                )
            )
        if cfg.fit_margin > M.FIT_MARGIN_MIN:
            saved = float(cfg.fit_margin - M.FIT_MARGIN_MIN)
            scored.append(
                (
                    saved,
                    (
                        f"Fit margin {cfg.fit_margin:,} -> {M.FIT_MARGIN_MIN:,} MiB "
                        f"frees {saved:,.0f} MiB, at the risk of a CUDA failure when "
                        f"the desktop takes more memory."
                    ),
                )
            )
        if ub is None and ctx is None:
            scored.append(
                (
                    -1.0,
                    (
                        "Nothing on this card fits this configuration. Use a card "
                        "with more video memory."
                    ),
                )
            )
    elif out.expert_layers >= M.BLOCK_COUNT and out.free_mib > 0:
        # "Another expert layer would fit" is only ever true below the block
        # count, and below the block count it cannot happen: the layer count
        # is a floor division, so anything left over is already less than one
        # layer. All that survives is this line, for a card holding all 48.
        scored.append(
            (
                -1.0,
                (
                    f"All {M.BLOCK_COUNT} expert layers are already on the card. "
                    f"The {out.free_mib:,.0f} MiB free buys a larger micro-batch or "
                    f"a larger window, nothing more."
                ),
            )
        )

    if not out.ram_ok:
        scored.append(
            (
                -1.0,
                (
                    f"System memory: {out.system_gb:,.1f} GB of weights plus "
                    f"{M.RAM_HEADROOM_GB:,.0f} GB for everything else is more than "
                    f"the {cfg.ram_total_gb:,.0f} GB this machine has. Put more "
                    f"layers on the card, or use a smaller model file."
                ),
            )
        )

    # Largest saving first. A line with no figure keeps its place at the end.
    ordered = [
        text
        for _saved, text in sorted(
            scored, key=lambda pair: -pair[0] if pair[0] >= 0 else float("inf")
        )
    ]
    out.notes = notes
    out.advice = ordered


# ---------------------------------------------------------------------------
# The .cmd launcher
# ---------------------------------------------------------------------------
# This is the only place that builds the launcher text; the interface only
# displays whatever ``launcher_script`` returns. The split between a
# ``set`` line and a flag on the ``llama-server.exe`` line follows
# ``llama-server.exe --help`` on the measured machine: a setting is a
# ``set`` line only when its help entry prints an "(env: ...)" line.

#: Flags with no environment variable, confirmed against
#: ``llama-server.exe --help``: none of the six print an "(env: ...)" line.

#: Measured on 2026-09-19 and it makes no difference: 16, 24 and 32 all
#: gave 616 to 628 tokens per second of prefill once the machine was
#: warm, and generation stayed between 21 and 23 throughout. A first
#: sweep appeared to show 32 winning by a quarter; that was the first
#: server after an idle machine being slow, and running the same three
#: backwards produced the opposite order. The earlier sweep in
#: ``bench-results\batch-threads`` reads the same way: 546, 540 and 570
#: for 12, 16 and 24 is one noise band. The value stays at 16 because it
#: is what the shipped launchers already carry.
_FLAG_THREADS_BATCH = 16
_FLAG_PRIO = 2
_FLAG_TEMP = "1.0"
_FLAG_TOP_P = "0.95"
_FLAG_MIN_P = "0.0"

#: The published n-gram-mod recipe. Not measured here.
_FLAG_SPEC_N_MATCH = 60
_FLAG_SPEC_N_MIN = 12
_FLAG_SPEC_N_MAX = 24

#: Environment-variable settings that are fixed, not driven by a control.
_ENV_JINJA = "1"
_ENV_N_PARALLEL = "1"
_ENV_LOAD_MODE = "mmap"
_ENV_LAZY_MODE = "on"
_ENV_FLASH_ATTN = "on"
_ENV_FIT = "on"
_ENV_TOP_K = "20"
_ENV_ENDPOINT_METRICS = "1"

_WARNING_RULE = ":: " + "=" * 70

#: One plain sentence per non-fitting state, for the warning block only.
_STATUS_SENTENCE = {
    DEGRADED: "this configuration spills to the processor",
    FAIL: "this configuration does not start",
}


def _warning_block(out: Estimate) -> list[str]:
    """A loud comment block, placed before every other comment.

    The person must not save a file that will not run, or that collapses
    generation speed, without seeing why in the first lines they read.
    """
    sentence = _STATUS_SENTENCE.get(out.status, out.status)
    lever = out.advice[0] if out.advice else "See the advice panel in the calculator."
    return [
        _WARNING_RULE,
        f":: WARNING: {sentence}.",
        f":: Single lever to fix this: {lever}",
        _WARNING_RULE,
        "",
    ]


def _chat_template_fetch_block() -> list[str]:
    """Batch lines that make sure a chat template file is on disk.

    A person who copies only this launcher gets the model, because
    llama-server downloads it from the repository named above, but not
    the template file, which is local. The template inside the model file
    refuses any system message that is not the first one, so the server
    answers HTTP 500 to every client that injects one mid-conversation.
    These lines fetch a published template with that rule removed, and
    stop the launcher when the fetch fails, because starting anyway hides
    the cause behind a retry loop.

    Comments inside the parenthesised block use ``rem``. A ``::`` label
    inside parentheses makes cmd.exe print "The system cannot find the
    drive specified".
    """
    folder = M.LAUNCHER_CHAT_TEMPLATE_RELATIVE_PATH.rsplit("\\", 1)[0]
    url = M.LAUNCHER_CHAT_TEMPLATE_FALLBACK_URL
    return [
        ":: The template file named above is not part of this launcher, and a",
        ":: machine that copied only this file does not have it. On a machine",
        ":: that has it, the next line jumps over everything here.",
        ":: Measured: the template inside the model file raises an exception on",
        ":: any system message that is not the first one, so Claude Code and",
        ":: other clients that inject one get HTTP 500 in a retry loop. The file",
        ":: fetched here is a published copy with that rule removed. It is the",
        ":: newest published version, so it carries whatever its author has",
        ":: repaired since. The version that arrived is printed below, because",
        ":: a template that changes upstream must not change here in silence.",
        ":: Labels, not nested parentheses: measured, cmd.exe reports 0 instead",
        ":: of 1 when exit /b sits inside two blocks and a command follows it.",
        'if exist "%LLAMA_ARG_CHAT_TEMPLATE_FILE%" goto template_ready',
        "echo Chat template not found. Fetching "
        + M.LAUNCHER_CHAT_TEMPLATE_FALLBACK_REPO
        + " ...",
        'if not exist "%~dp0' + folder + '" mkdir "%~dp0' + folder + '"',
        'curl.exe -L -f -s -S -o "%LLAMA_ARG_CHAT_TEMPLATE_FILE%" "' + url + '"',
        "if errorlevel 1 goto template_failed",
        "echo Chat template saved.",
        """for /f tokens^=2^ delims^=^" %%V in ('findstr /c:"template_version" "%LLAMA_ARG_CHAT_TEMPLATE_FILE%"') do echo Template version: %%V""",
        "goto template_ready",
        "",
        ":template_failed",
        "echo.",
        "echo Could not fetch the chat template. The template inside the model",
        "echo file answers HTTP 500 to a system message that arrives",
        "echo mid-conversation, so this launcher stops instead of starting a",
        "echo server that fails on the first request.",
        "echo Fetch this file by hand:",
        "echo   " + url,
        "echo and save it as %LLAMA_ARG_CHAT_TEMPLATE_FILE%",
        "popd",
        "exit /b 1",
        "",
        ":template_ready",
        "",
    ]


def _mmproj_fetch_block() -> list[str]:
    """Batch lines that make sure the image projector file is on disk.

    llama-server downloads the model itself from the repository named
    above, but the projector is passed to it as a file path (see
    launcher_script), and LLAMA_ARG_MMPROJ_AUTO is set to 0 because
    --mmproj-auto was measured to do nothing with a file path anyway. So
    nothing fetches the projector for a machine that copied only this
    launcher, the same gap _chat_template_fetch_block closes for the
    template.

    The download lands on a temporary name first, never straight on the
    target: a run interrupted partway through a direct download would
    leave a half file that looks finished, and the server would then fail
    on some unrelated error instead of on a plainly missing file.

    Comments inside the parenthesised block use ``rem``. A ``::`` label
    inside parentheses makes cmd.exe print "The system cannot find the
    drive specified". Cleaning up the partial file also has to happen off
    its own label, not chained after another ``if errorlevel 1``:
    measured, a ``del`` of a file that turns out not to be there resets
    errorlevel, so a second ``if errorlevel 1`` right after it never fires.
    """
    folder = M.LAUNCHER_MMPROJ_RELATIVE_PATH.rsplit("\\", 1)[0]
    url = M.LAUNCHER_MMPROJ_URL
    part = "%LLAMA_ARG_MMPROJ%.part"
    measured_mib = M.LAUNCHER_MMPROJ_MEASURED_BYTES / (1024 * 1024)
    return [
        ":: The projector file named above is not part of this launcher, and a",
        ":: machine that copied only this file does not have it. On a machine",
        ":: that has it, the next line jumps over everything here. It comes",
        ":: from the same repository as the model, because that is where it",
        ":: is measured to live.",
        'if exist "%LLAMA_ARG_MMPROJ%" goto mmproj_ready',
        (
            f"echo Projector not found. This is a {measured_mib:.0f} MiB download "
            "and takes a while."
        ),
        'if not exist "%~dp0' + folder + '" mkdir "%~dp0' + folder + '"',
        'curl.exe -L -f --progress-bar -o "' + part + '" "' + url + '"',
        "if errorlevel 1 goto mmproj_download_failed",
        'move /y "' + part + '" "%LLAMA_ARG_MMPROJ%"',
        "if errorlevel 1 goto mmproj_failed",
        'for %%A in ("%LLAMA_ARG_MMPROJ%") do echo Projector saved, %%~zA bytes.',
        "goto mmproj_ready",
        "",
        ":mmproj_download_failed",
        'del "' + part + '"',
        "goto mmproj_failed",
        "",
        ":mmproj_failed",
        "echo.",
        "echo Could not fetch the image projector. Image input needs this",
        "echo file, and llama-server does not fetch it by itself.",
        "echo Fetch this file by hand:",
        "echo   " + url,
        "echo and save it as %LLAMA_ARG_MMPROJ%",
        "popd",
        "exit /b 1",
        "",
        ":mmproj_ready",
        "",
    ]


def _used_video_memory_mib(out: Estimate) -> float:
    """Everything actually resident on the card, MiB.

    Adds up to ``vram_total_mib`` exactly when the run fits (``free_mib``
    is the remainder); when it does not, ``free_mib`` is zero and this is
    what the card holds before the overflow.
    """
    return (
        out.dense_mib
        + out.expert_mib
        + out.kv_mib
        + out.compute_mib
        + out.vision_mib
        + out.margin_mib
    )


def _header_comments(cfg: Config, out: Estimate) -> list[str]:
    """The comment block every launcher gets, fitting or not."""
    lines = [
        f":: {M.MODEL_NAME} launcher, written by the tuning calculator.",
        (
            f":: Estimated: prefill {out.pp_tps:,.0f} tokens/s, "
            f"generation {out.tg_tps:,.2f} tokens/s."
        ),
        (
            f":: Estimated video memory: {_used_video_memory_mib(out):,.0f} MiB "
            f"of {cfg.vram_total_mib:,}, with {out.expert_layers} of "
            f"{out.expert_layers_max} expert layers on the card."
        ),
        (
            ":: These are estimates from one measured machine. See "
            "bench-results\\results.md."
        ),
    ]
    if cfg.vram_busy_mib > 0:
        lines.append(
            f":: Other programs held {cfg.vram_busy_mib:,} MiB of the card "
            f"when this was written, so the numbers above are for the "
            f"{cfg.vram_for_us_mib:,} MiB that were left. Free the card and "
            f"the same settings do better."
        )
    if out.pp_extrapolated:
        lines.append(
            f":: Prompt processing is extrapolated: {out.pp_extrapolation_reason}."
        )
    if out.tg_extrapolated:
        lines.append(f":: Generation is extrapolated: {out.tg_extrapolation_reason}.")
    if (
        cfg.vram_total_mib != M.REFERENCE_VRAM_MIB
        or cfg.ram_total_gb != M.REFERENCE_RAM_GB
    ):
        lines.append(
            f":: Estimates were fitted on an RTX 4090 Laptop with "
            f"{M.REFERENCE_VRAM_MIB:,} MiB and {M.REFERENCE_RAM_GB:g} GB. "
            f"The paths in this file are for this machine."
        )
    return lines


def launcher_script(cfg: Config, out: Estimate) -> str:
    """A complete, runnable Windows .cmd launcher for this configuration.

    Every setting with an environment variable (checked against
    ``llama-server.exe --help`` on the measured machine) is written as a
    ``set`` line; every setting without one stays a flag on the
    ``llama-server.exe`` line. See measured.py for the launcher's fixed
    repository, projector path, host and port.
    """
    lines: list[str] = ["@echo off", "setlocal", "pushd %~dp0", ""]

    if out.status != OK:
        lines += _warning_block(out)

    lines += _header_comments(cfg, out)
    lines.append("")

    lines.append(f'set "LLAMA_ARG_HF_REPO={M.LAUNCHER_HF_REPO}"')

    lines.append(
        ":: The name any client sees at /v1/models. Measured: llama.cpp "
        "answers with the loaded model whatever name a request carries, so "
        "this is a label, not a key."
    )
    lines.append(f'set "LLAMA_ARG_ALIAS={M.LAUNCHER_ALIAS}"')

    lines.append(
        ":: The chat template this server must use. Why the model's own will not do,"
    )
    lines.append(
        ":: and where this file comes from when it is missing, is written out once,"
    )
    lines.append(":: further down beside the line that fetches it.")
    lines.append(
        'set "LLAMA_ARG_CHAT_TEMPLATE_FILE=%~dp0'
        f'{M.LAUNCHER_CHAT_TEMPLATE_RELATIVE_PATH}"'
    )

    lines.append(f'set "LLAMA_ARG_HOST={M.LAUNCHER_HOST}"')
    lines.append(f'set "LLAMA_ARG_PORT={M.LAUNCHER_PORT}"')
    lines.append(f'set "LLAMA_ARG_JINJA={_ENV_JINJA}"')
    lines.append(f'set "LLAMA_ARG_N_PARALLEL={_ENV_N_PARALLEL}"')
    lines.append("")

    lines.append(
        ":: The n-gram table must stay mapped on the disk. Never --load-mode none."
    )
    lines.append(f'set "LLAMA_ARG_LOAD_MODE={_ENV_LOAD_MODE}"')
    lines.append(f'set "LLAMA_ARG_LAZY_MODE={_ENV_LAZY_MODE}"')
    lines.append("")

    lines.append(":: Flash attention is required for a quantized value cache.")
    lines.append(f'set "LLAMA_ARG_FLASH_ATTN={_ENV_FLASH_ATTN}"')
    lines.append("")

    lines.append(
        ":: --fit places the layers itself. The margin is video memory it leaves free."
    )
    lines.append(f'set "LLAMA_ARG_FIT={_ENV_FIT}"')
    lines.append(f'set "LLAMA_ARG_FIT_TARGET={cfg.fit_margin}"')
    lines.append("")

    if out.ubatch_clamped:
        lines.append(
            f":: The requested micro-batch {cfg.ubatch} is larger than the "
            f"window {cfg.ctx}: llama.cpp never has more tokens in flight "
            f"than the window holds, so it would have clamped this anyway. "
            f"The value below is already clamped to {out.effective_ubatch}."
        )

    lines.append(f'set "LLAMA_ARG_CTX_SIZE={cfg.ctx}"')
    lines.append(f'set "LLAMA_ARG_BATCH={cfg.batch}"')
    lines.append(f'set "LLAMA_ARG_UBATCH={out.effective_ubatch}"')
    lines.append(f'set "LLAMA_ARG_CACHE_TYPE_K={cfg.kv_type}"')
    lines.append(f'set "LLAMA_ARG_CACHE_TYPE_V={cfg.kv_type}"')
    lines.append(f'set "LLAMA_ARG_THREADS={cfg.threads}"')
    lines.append(
        f'set "LLAMA_ARG_SPEC_TYPE={"ngram-mod" if cfg.speculation else "none"}"'
    )

    if cfg.vision:
        lines.append(f'set "LLAMA_ARG_MMPROJ=%~dp0{M.LAUNCHER_MMPROJ_RELATIVE_PATH}"')
        lines.append(
            ":: --mmproj-auto was measured to do nothing when the model is "
            "given as a file path, and it was never tested with a "
            "repository name, so the file is named outright."
        )
    else:
        lines.append(":: Image input is off, so no projector file is loaded.")
    lines.append('set "LLAMA_ARG_MMPROJ_AUTO=0"')

    lines.append("")
    lines.append(f'set "LLAMA_ARG_TOP_K={_ENV_TOP_K}"')
    lines.append(f'set "LLAMA_ARG_ENDPOINT_METRICS={_ENV_ENDPOINT_METRICS}"')
    lines.append("")
    lines += _chat_template_fetch_block()
    if cfg.vision:
        lines += _mmproj_fetch_block()

    # Every comment about the flag line below has to come before it: once
    # the caret continuation starts, a "::" line would be read as an
    # argument, not a comment.
    lines.append(
        ":: No environment variable exists for the settings on the next line, so they"
    )
    lines.append(
        ":: have to be flags. Checked against all 139 variables in the server's help."
    )
    if cfg.speculation:
        lines.append(
            ":: The three --spec-ngram-mod-* values below come from the "
            "published recipe; they were not measured here."
        )
    lines.append("llama-server.exe ^")
    lines.append(f" --threads-batch {_FLAG_THREADS_BATCH} --prio {_FLAG_PRIO} ^")
    lines.append(f" --temp {_FLAG_TEMP} --top-p {_FLAG_TOP_P} --min-p {_FLAG_MIN_P} ^")
    lines.append(" --no-warmup ^")
    if cfg.speculation:
        lines.append(
            f" --spec-ngram-mod-n-match {_FLAG_SPEC_N_MATCH} "
            f"--spec-ngram-mod-n-min {_FLAG_SPEC_N_MIN} "
            f"--spec-ngram-mod-n-max {_FLAG_SPEC_N_MAX} ^"
        )
    lines.append(" %*")
    lines.append("")
    lines.append("popd")
    lines.append("exit /b %ERRORLEVEL%")

    return "\n".join(lines)
