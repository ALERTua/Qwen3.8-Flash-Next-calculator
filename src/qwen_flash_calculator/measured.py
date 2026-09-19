"""Measured data for Qwen3.8-Flash-Next on an RTX 4090 Laptop.

Everything in this file is a measurement, a coefficient fitted to a
measurement, or a figure published by the people who built the model file
and marked as published where it appears. Nothing here is invented. The
source is the benchmark session recorded in ``bench-results/results.md``
(2026-09-17 / 2026-09-18).

No interface code, no logic: tables only.
"""

# --------------------------------------------------------------------------
# The model under test
# --------------------------------------------------------------------------

MODEL_NAME = "Qwen3.8-Flash-Next"

#: The model file variant, the quant. "Build" is not used for this anywhere
#: in this project, because llama.cpp itself also has builds; a llama.cpp
#: build is always called a llama.cpp version here.
MODEL_QUANT = "AD-4.27bpw-Q4_K_M-M64"

#: Transformer blocks in the GGUF header (``qwen4exp.block_count``).
BLOCK_COUNT = 48

#: llama.cpp counts the 48 blocks plus the output layer when it prints
#: "49 layers (N overflowing)".
LLAMA_LAYER_COUNT = 49

#: Native context length from the GGUF header.
NATIVE_CONTEXT = 262144

#: Total fast memory the weights need, dense plus experts, in GB.
#: Published by the people who built the model file (the quant), not
#: measured here.
#:
#: This figure and the video-memory coefficients below disagree, and the
#: disagreement is on purpose. 6200 + 48 * 1138 = 60,824 MiB = 59.4 GiB of
#: weights if every expert layer sat on the card, against the 54.5 GB
#: published for the same model file. The two numbers come from different
#: sources: the coefficients were fitted to what llama.cpp actually reports
#: as resident on the graphics card, the 54.5 GB is the model file's own
#: published size. For the system-memory line the published number wins,
#: and the subtraction is floored at zero so the panel can never print a
#: negative amount of weights in system memory.
#:
#: The 38.4 GB n-gram table inside the model file is a different thing
#: again: it stays mapped on the solid-state drive, about 2.7 KB is read
#: per token, and it is not counted anywhere here. It has nothing to do
#: with the n-gram speculation switch in the interface.
TOTAL_WEIGHTS_GB = 54.5

# --------------------------------------------------------------------------
# The machine the numbers were measured on
# --------------------------------------------------------------------------

REFERENCE_VRAM_MIB = 16376
REFERENCE_RAM_GB = 96.0

#: The main grid was measured with ``--fit on -fitt 512``. The degraded
#: anchor below therefore belongs to a margin of 512, not to whatever the
#: user has selected in the interface.
REFERENCE_FIT_MARGIN_MIB = 512

#: results.md section 7.4 recommends 1024 for a machine that also runs a
#: browser and an editor. That is the interface default.
DEFAULT_FIT_MARGIN_MIB = 1024

# --------------------------------------------------------------------------
# Video memory coefficients (MiB)
# --------------------------------------------------------------------------
# These four coefficients describe what lands on the graphics card: they
# were fitted to the CUDA0 breakdowns llama.cpp prints, and together they
# reproduce all four measured whole-expert-layer counts at the default fit
# margin of 1024 MiB. They deliberately do not agree with the published
# total weight size of 54.5 GB above; see the note there for why, and why
# the system-memory line uses the published figure instead of these.

#: Dense weights plus the attention tensors that always stay on the card.
DENSE_AND_ATTENTION_MIB = 6200

#: One block's expert weights, resident on the card.
PER_EXPERT_LAYER_MIB = 1138

#: Key-value cache, MiB per context token, by cache type.
KV_MIB_PER_TOKEN = {
    "q8_0": 0.0149,
    "f16": 0.0298,
}

#: Compute buffers grow with both the micro-batch and the window.
COMPUTE_MIB_PER_UBATCH_TOKEN = 0.000014

#: The vision projector file, mmproj-Qwen3.8-Flash-Next-F16.gguf. This is
#: what the image-input switch turns on and off.
VISION_PROJECTOR_MIB = 900

#: The n-gram table inside the model file, from the quant's published table
#: of sizes. It stays mapped on the solid-state drive, about 2.7 KB of it is
#: read per generated token, and it takes no video memory and no system
#: memory. It has nothing to do with the n-gram speculation switch.
NGRAM_TABLE_GB = 38.4
NGRAM_TABLE_KB_PER_TOKEN = 2.7

# --------------------------------------------------------------------------
# The four measured video-memory breakdowns (results.md section 7.2)
# --------------------------------------------------------------------------
# (ctx, ubatch, model MiB, kv MiB, compute MiB,
#  blocks with their expert weights on the card out of 48,
#  layers llama.cpp reports on the card out of 49)

MEASURED_BREAKDOWNS = [
    (163840, 2048, 7531, 2407, 4752, 1, 49),
    (262144, 2048, 4281, 2828, 7581, 0, 33),  # degraded run
    (163840, 512, 10593, 2407, 1188, 4, 49),
    (262144, 512, 8668, 3784, 1813, 2, 49),
]

# --------------------------------------------------------------------------
# Generation speed
# --------------------------------------------------------------------------

#: base(L) = TG_BASE_INTERCEPT + TG_PER_LAYER * L, measured at 10 threads,
#: window 163840, from the hand-made -ncmoe splits in results.md 7.3.
TG_BASE_INTERCEPT = 16.87
TG_PER_LAYER = 0.74

#: The straight line above was fitted on these whole-expert-layer counts
#: only. Above the largest of them the line is an extrapolation, and every
#: estimate that uses it must say so.
TG_FITTED_LAYER_COUNTS = (0, 1, 2, 4)
TG_MAX_FITTED_LAYERS = max(TG_FITTED_LAYER_COUNTS)

#: Generation, tokens per second, against thread count. Measured at ctx
#: 163840 with a short prompt. Section 6.3.
THREAD_SAMPLES = [
    (4, 14.84),
    (6, 19.97),
    (8, 20.27),
    (10, 20.60),
    (12, 21.09),
    (14, 22.22),
    (16, 23.98),
    (18, 23.89),
    (20, 23.92),
    (22, 23.13),
    (24, 25.12),
    (32, 22.19),
]

#: Thread count the base curve was measured at. The thread factor is
#: normalised so that this many threads gives exactly 1.00.
THREAD_REFERENCE = 10

#: The curve above was measured with a short, 370-token prompt at window
#: 65,536. It is the only thread curve this project has, and it does not
#: hold at a working depth: at about 20,000 tokens, going from 10 to 24
#: threads was measured at only 2% to 4%, not the roughly 22% the curve
#: implies for that same jump. The gain above the reference is therefore
#: damped to what was measured at depth.
THREAD_DEPTH_CAVEAT = (
    "This curve was measured with a short prompt at window 65,536. At a "
    "working depth of about 20,000 tokens, 10 to 24 threads was measured "
    "at only 2% to 4%, not the ~22% this curve implies, so the gain above "
    "10 threads is damped to what was measured at depth."
)

#: What damping this curve to the depth measurement would cost, written
#: down because it was tried on 2026-09-19 and put back.
#:
#: Three per cent measured at depth over the twenty-two the curve claims
#: is a factor of about 0.14 on the gain above the reference. Applied, it
#: brings the estimate for the launcher in use from 24.50 tokens per
#: second to 18.90 against a measured 17.7, which is better. It also
#: moves the worst error against the five directly measured runs in
#: VALIDATION_POINTS from 14 per cent to 19, because those five were
#: taken with the same harness and carry the same inflation. One factor
#: cannot be corrected while the grid it sits in stays as it is. Re-
#: measure the grid with a new text per request, then refit, then damp.
THREAD_DEPTH_DAMPING_IF_REFITTED = 3.0 / 22.0

#: n-gram speculation (``--spec-type ngram-mod``): the speed-up on repeated
#: text. 1.00 on text the model has not seen before, up to 2.9 on text it
#: has seen. Section 6.5. The pool it speculates from is built out of the
#: conversation itself and costs no video memory. This is NOT the 38.4 GB
#: n-gram table inside the model file; the two only share a word.
SPECULATION_MIN = 1.00
SPECULATION_MAX = 2.9

# --------------------------------------------------------------------------
# The degraded anchor
# --------------------------------------------------------------------------
# At the full window with micro-batch 2048 and fit margin 512 the fitter
# pushed whole layers, attention included, onto the processor: 33 of 49
# layers stayed on the card and generation fell from 19.8 to 8.1 t/s.
# Every degraded estimate is anchored on this single measured point.

DEGRADED_ANCHOR_CTX = 262144
DEGRADED_ANCHOR_UBATCH = 2048
DEGRADED_ANCHOR_TG = 8.1
DEGRADED_ANCHOR_LLAMA_LAYERS = 33

# --------------------------------------------------------------------------
# Prompt processing grid (results.md section 8)
# --------------------------------------------------------------------------
# Rows are the window, columns the micro-batch, values tokens per second.

PP_UBATCH_COLUMNS = [512, 1024, 2048, 4096]

PP_GRID = {
    65536: [323, 453, 575, 700],
    98304: [318, 427, 604, 718],
    131072: [346, 463, 573, 696],
    163840: [341, 460, 570, 630],
    196608: [322, 442, 574, 608],
    262144: [326, 447, 530, 582],
}

PP_CTX_ROWS = sorted(PP_GRID)

# --------------------------------------------------------------------------
# Validation: five whole configurations, measured directly
# --------------------------------------------------------------------------
# These are direct measurements of five whole runs, taken after the model
# above was already built. They are never fitted to and never interpolated:
# their only job is to show, next to the estimate, how far it lands from a
# real run. All five share cache q8_0, ``--fit on``, fit margin 1,024 MiB,
# speculation off and vision off.
#
# (window, micro-batch, threads, measured prefill tokens/s,
#  measured generation tokens/s, results folder)

VALIDATION_POINTS = [
    (65536, 4096, 24, 741, 23.43, "final"),
    (131072, 2048, 24, 525, 20.42, "ctx130"),
    (131072, 1024, 24, 431, 21.53, "ctx130"),
    (262144, 512, 24, 328, 23.47, "final"),
    (262144, 512, 10, 326, 19.84, "grid"),
]

# --------------------------------------------------------------------------
# Interface ranges
# --------------------------------------------------------------------------

CTX_MIN = 4096
CTX_STEP = 8192
CTX_MAX = 262144

#: 4096 is the floor asked for; every other stop is a multiple of 8192.
CTX_CHOICES = [CTX_MIN] + [CTX_STEP * k for k in range(1, CTX_MAX // CTX_STEP + 1)]

UBATCH_CHOICES = [256, 512, 1024, 2048, 4096, 8192]

KV_TYPES = ["q8_0", "f16"]

THREAD_MIN = 1
THREAD_MAX = 32

FIT_MARGIN_MIN = 256
FIT_MARGIN_MAX = 4096
FIT_MARGIN_STEP = 64

#: Presets so other machines can be modelled. 16376 is the measured card.
VRAM_PRESETS = [
    ("12 GB", 12288),
    ("16 GB", 16376),
    ("24 GB", 24576),
    ("48 GB", 49152),
]

RAM_PRESETS = [
    ("64 GB", 64.0),
    ("96 GB", 96.0),
    ("128 GB", 128.0),
]

#: Shipped profiles: whole configurations that were also measured directly
#: (see VALIDATION_POINTS above), offered as one-click presets so a person
#: can land on a measured point instead of hunting the sliders for one.
#: Loading any of the three also sets cache q8_0, fit margin 1,024 MiB,
#: speculation on and image input on; those four are fixed for every
#: profile, not part of what tells the three apart.
#:
#: (label, window, micro-batch, threads)
CONFIG_PRESETS = [
    ("Long context", 262144, 512, 24),
    ("Middle", 131072, 2048, 24),
    ("Fast prompts", 65536, 4096, 24),
]

#: System memory the machine must keep free for everything else.
RAM_HEADROOM_GB = 8.0

DISCLAIMER = (
    "Every figure is an estimate fitted to one measured machine "
    "(RTX 4090 Laptop 16,376 MiB, i9-13900HX, 96 GB) and to the one "
    "llama.cpp version that session was run with. A different processor, "
    "a different memory speed or a different llama.cpp version will move "
    "them."
)

# --------------------------------------------------------------------------
# The .cmd launcher (the panel at the bottom of the window)
# --------------------------------------------------------------------------
# Fixed identifiers the launcher needs that are not a control anywhere in
# the interface: which model file to fetch, where its projector lives
# relative to the launcher, and which host and port to bind the server to.

LAUNCHER_HF_REPO = "AtomicChat/Qwen3.8-Flash-Next-GGUF:AD-4.27bpw-Q4_K_M-M64"

#: The name this server answers to at /v1/models. Purely a label: llama.cpp
#: serves the loaded model whatever name a request carries.
LAUNCHER_ALIAS = "qwen3.8-flash-next"

#: Relative to the launcher's own folder, joined onto %~dp0. A patched copy
#: of the model's own chat template that lets a system message appear
#: mid-conversation; see estimator.py's launcher_script for why.
LAUNCHER_CHAT_TEMPLATE_RELATIVE_PATH = (
    "chat-templates\\qwen3.8-flash-next-multi-system.jinja"
)

#: Where the launcher gets a working template when that file is missing,
#: which is the normal state on any machine that copied only the launcher.
#: The launcher follows the newest published version, because the author
#: keeps repairing the template and a person setting up a machine wants
#: those repairs. The launcher prints the version it fetched, so a change
#: upstream shows up in the window instead of passing unnoticed.
LAUNCHER_CHAT_TEMPLATE_FALLBACK_REPO = "froggeric/Qwen-Fixed-Chat-Templates"
LAUNCHER_CHAT_TEMPLATE_FALLBACK_REVISION = "main"

#: The revision measured here on 2026-09-18, kept for the record. It
#: answered with code 200 and 28,234 bytes, carried no "System message
#: must be at the beginning." rule, and rendered a system message in the
#: middle of a conversation as its own system turn. It called itself
#: "qwen3.8-froggeric-v22.5", and on that day it was also what "main"
#: served. Anything newer than this is not measured here.
LAUNCHER_CHAT_TEMPLATE_MEASURED_REVISION = "855bffc49448e299789730ff92c9b8d834d6cc14"
LAUNCHER_CHAT_TEMPLATE_MEASURED_VERSION = "qwen3.8-froggeric-v22.5"
LAUNCHER_CHAT_TEMPLATE_MEASURED_BYTES = 28234
LAUNCHER_CHAT_TEMPLATE_FALLBACK_FILE = "chat_template.jinja"
LAUNCHER_CHAT_TEMPLATE_FALLBACK_URL = (
    "https://huggingface.co/"
    + LAUNCHER_CHAT_TEMPLATE_FALLBACK_REPO
    + "/resolve/"
    + LAUNCHER_CHAT_TEMPLATE_FALLBACK_REVISION
    + "/"
    + LAUNCHER_CHAT_TEMPLATE_FALLBACK_FILE
)

#: Relative to the launcher's own folder, joined onto %~dp0.
LAUNCHER_MMPROJ_RELATIVE_PATH = "models\\mmproj-Qwen3.8-Flash-Next-F16.gguf"

#: The projector lives in the same repository as the model, without the
#: quantization suffix after the colon. Built from LAUNCHER_HF_REPO instead
#: of a second copy of the repository name, so the two never drift apart.
LAUNCHER_MMPROJ_REPO = LAUNCHER_HF_REPO.split(":", 1)[0]

#: The file name only, taken from the path above.
LAUNCHER_MMPROJ_FILE = LAUNCHER_MMPROJ_RELATIVE_PATH.rsplit("\\", 1)[-1]

LAUNCHER_MMPROJ_URL = (
    "https://huggingface.co/"
    + LAUNCHER_MMPROJ_REPO
    + "/resolve/main/"
    + LAUNCHER_MMPROJ_FILE
)

#: Measured on 2026-09-19 against LAUNCHER_MMPROJ_URL: answered HTTP 200
#: with a file of exactly this size, which also matches the copy already
#: on this machine at %~dp0LAUNCHER_MMPROJ_RELATIVE_PATH.
LAUNCHER_MMPROJ_MEASURED_BYTES = 904003840

LAUNCHER_HOST = "127.0.0.1"
LAUNCHER_PORT = 8080
