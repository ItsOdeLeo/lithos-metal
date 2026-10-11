# Apple M2 Max, 30 GPU cores

The exact `Apple M2 Max` / `Apple8` / 30-core device uses the cooperative matrix
path for prefill and multi-token projections. Its operand and destination
coordinates match the shared Apple9 layout. M2 selects TN=16, with TK=64 for
BF16/FP8 and TK=128 for block-scaled formats; the measured GEMV-to-matrix
crossovers are T=8 for BF16/NVFP4 and T=4 for FP8/INT8/affine INT4. The pack
remains `interleaved16` with block scales. Crossover calibration uses 17408×5120
matrices, min-of-3, and the best measured shader geometry at each T, following
the shared calibration convention; full-model gains are measured separately
below.

## Local GDN geometry

Enabling the matrix path also enables local GDN preparation and producer-side
permutation. In the reference model on Apple8, the shared 1024-thread GDN
dispatch exceeds the actual pipeline limit: the fused/permuted one-token kernel
permits 896 threads normally and 384 with shader validation; the eight-token
local kernel permits 512 with validation. Such dispatches can leave the output
unwritten.

The M2 finalizer keeps local preparation in 256-thread groups. The fused norm
uses sixteen-column state slices so each group still owns a complete head;
multi-token kernels retain their state slices and split each head across smaller
groups. Grid sizes, parameter records and specialized constants agree. Resource
bindings, barriers, state layout and rounding order are preserved. Output and
both state slots are checked against separate preparation/norm, including
partial steps, inactive steps, shared key heads and separate gate projections.

The shared changes are limited to recognizing Apple8's measured cooperative
layout and an optional `gemm_tile_shape` backend hook. The central layout
selector supplies both input permutation and fused producer ordering, which must
agree. Tile geometry is selected before dispatch emission, so a finalizer alone
cannot safely change it. Other backends return `None` and retain their previous
default macros; explicit tile arguments still take precedence. Other devices'
hardware performance has not been remeasured.

## Qualification

Measured on a 30-core M2 Max, 64 GB unified memory, macOS 26.7 (`25G229`), SDK
26.2, Apple clang 17.0.0 and Python 3.13.7, on 2026-10-11. Oracle dependencies:
PyTorch 2.14.0 and Transformers 5.17.0.

The BF16 reference is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, safetensors SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. Kernel
qualification covers BF16, FP8 E4M3, NVFP4, INT8 and affine INT4; full-model
measurements use the BF16 reference. INT8/affine INT4 matrix operands round the
dequantized weights to BF16, as documented by the shared matrix oracle.

All timings below are min-of-5 after warmup, alternating off/on order on the
same machine and day. Only the accelerator option changes. Both paths use direct
serial encoding and the same BF16 pack; autotuning and shader validation are
off. All paired generated tokens/text match.

| Prompt tokens | Prefill GPU off (ms) | On (ms) | Speedup | Decode off / on (ms/token) |
| --- | ---: | ---: | ---: | ---: |
| 128 | 942.28 | 64.58 | 14.59× | 8.684 / 6.446 |
| 512 | 3293.96 | 223.61 | 14.73× | 8.696 / 6.485 |
| 1024 | 6478.25 | 464.27 | 13.95× | 8.793 / 6.595 |

Session runs generate 32 tokens with 128-token chunks and capacity 1184; prefix
caching is disabled. Every timed run has zero setup time and zero cached prompt
tokens. Raw samples also retain Session prefill wall time; that is separate from
HTTP/SSE TTFT.

| Actual chat prompt tokens | SSE TTFT off (ms) | On (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 146 | 1522.27 | 120.60 | 12.62× |
| 495 | 3247.84 | 238.83 | 13.60× |
| 961 | 6433.43 | 490.27 | 13.12× |

SSE runs use the normal serving CLI, capacity 2048, 128-token chunks, no
drafter, and `--keep-warm 0`. Prefix caching is enabled, with distinct prefixes
per pair: server metrics confirm zero cache hits and zero setup time in every
timed request. TTFT starts before HTTP submission and ends at the first nonempty
text event. Warmup rows are retained and excluded from the table.

| Correctness check | Result |
| --- | --- |
| Contract tier | 1171 passed, 1 optional MLX check skipped |
| Shared matrix kernels, shader validation | 178 passed, 9 inapplicable variants skipped |
| BF16/NVFP4 row kernels and GEMV fusions, shader validation | 201 passed |
| M2 tiles/GDN/dispatch limits, runtime and mixed workloads, shader validation | 65 passed |
| Layers and Qwen3.5 reference model, shader validation | 15 passed, 1 optional MLX check skipped |
| Fixed-layout calibration | All 75 format/geometry points passed their oracles |
| Normal serving CLI | Health/models, repeated greedy, prefix reuse, SSE text/usage, three API protocols and disconnect recovery passed |

Raw evidence and serving reproduction are in [the benchmark
record](../../../../tools/bench/results/m2-max-30c-matrix-20261011/README.md).

## Command encoding and limits

The backend retains direct serial command encoding by default. Mixed ICB replay
hangs remain unresolved; ICB replay is available only as an explicit diagnostic
override. Both benchmark paths use the same encoding policy. Cross-worker mixer
fusion remains disabled. The [hardware probe
run](../../../../probes/results/Apple-M2-Max_30c_macOS26.7_20261009-105128.txt)
records the unresolved claim-protocol stall and cross-threadgroup handoff
timeouts; neither is counted as a pass.

The 38-core M2 Max, large public MoE/DSpark pairs and every experimental kernel
geometry are outside this qualification. In particular, the shared four-column
fused GDN harness and the BF16 four-row commuted-norm experiment are not enabled
by this backend's selected geometry/crossover. A bounded validation run is not
an exhaustive stability guarantee.

## Reproduce

Follow [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for the native build and
`.[dev,oracle,serve]` environment. Place the pinned checkpoint at
`$MONOLITH_MODELS/Qwen3.5-0.8B`; set `PACK` to a local output directory.

```bash
python tools/pack_weights.py --model "$MONOLITH_MODELS/Qwen3.5-0.8B" \
  --out "$PACK" --lane-order interleaved16 --scale-placement block --max-context 4096
python -m pytest tests/contract
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_gemm_tile.py tests/kernels/test_m2_matrix.py \
  tests/runtime tests/layers tests/models/qwen3_5
python tools/bench/matrix_profile.py \
  --config monolith/backends/metal/m2_max_30c/config.json --out /tmp/m2-profile.json
python tools/bench/matrix_ab.py \
  --profile monolith/backends/metal/m2_max_30c/config.json \
  --model "$MONOLITH_MODELS/Qwen3.5-0.8B" --pack "$PACK" \
  --lengths 128 512 1024 --tokens 32 --repeats 5 --out /tmp/m2-model-ab.jsonl
```

Keep shader validation off for timings. `matrix_profile.py` only produces a
candidate profile: retain the source pack layout and qualify it with model and
serving checks before promoting its matrix policy. Session prefill wall time is
reported separately from real HTTP/SSE time to first visible text.