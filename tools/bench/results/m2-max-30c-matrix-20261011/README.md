# M2 Max matrix qualification — 2026-10-11

Apple M2 Max, 30 GPU cores, Apple8, 64 GB, macOS 26.7. The
[backend report](../../../../monolith/backends/metal/m2_max_30c/README.md)
contains the A/B tables, pinned BF16 checkpoint, software versions and limits.

This is the work for [#40](https://github.com/lithos-ai/lithos-metal/issues/40),
after [#9](https://github.com/lithos-ai/lithos-metal/pull/9) and
[#35](https://github.com/lithos-ai/lithos-metal/pull/35).
Development base: `dc550f4be90e41394f53c09954c8a1439b893ff7`.
The model benchmark records the implementation's source hash and profile/pack
manifest hashes, including changes beyond that base.

- `qualification.jsonl`: fixed-layout measurement metadata, layout-readback summary,
  and 75 oracle-checked GEMV/matrix points. Per-point timings are min-of-3 over
  approximately 2 GiB. Reproduce the retained matrix decision with
  `tools/bench/matrix_profile.py`; attention and cross-worker fusion are retained.
  Its header also retains gate counts and serving checks; optional/inapplicable
  skips are reported separately from passes.
- `model-ab.jsonl`: five alternating warmed pairs at each prompt length, 32
  generated tokens, no prefix caching, no timed compilation. All tokens match.
- `serve-ab.jsonl`: five alternating real SSE pairs at each chat prompt length,
  plus retained warmups. All text and usage match. Every timed request has zero
  setup time and zero cache hits. Use `usage.prompt_tokens` for actual lengths;
  the source token-slice length identifies the case before chat templating.

The off/on knob controls the shared matrix route and its producer fusions. These
are full-path gains against the correct shader route, rather than an isolated
tile-shape comparison. Both paths keep the M2 direct-encoding policy. Every timed
pair improves prefill GPU time and SSE TTFT; tables report min-of-5 as required by
CONTRIBUTING.md. Decode timings are reported independently.

## Reproduce serving A/B

Build the runtime and reference pack using the backend report. Unset
`MTL_SHADER_VALIDATION` and `MTL_SHADER_VALIDATION_REPORT_TO_STDERR` for timings.
Start each server in its own terminal and wait for its ready message, completing
the first server's warmup before starting the second:

```bash
MONOLITH_ACCELERATOR=off OMP_NUM_THREADS=8 python -m monolith.serve \
  --model "$MONOLITH_MODELS/Qwen3.5-0.8B" --no-draft --local-files-only \
  --pack "$PACK" --max-context 2048 --prefill-chunk-size 128 --keep-warm 0 \
  --port 18091 > /tmp/m2-off.log 2>&1
```

```bash
MONOLITH_ACCELERATOR=on OMP_NUM_THREADS=8 python -m monolith.serve \
  --model "$MONOLITH_MODELS/Qwen3.5-0.8B" --no-draft --local-files-only \
  --pack "$PACK" --max-context 2048 --prefill-chunk-size 128 --keep-warm 0 \
  --port 18092 > /tmp/m2-on.log 2>&1
```

In a third terminal, run the retained paired harness. It sends requests
sequentially, excludes warmups, checks server metrics, and fails on text/usage
changes. The servers stay alive until stopped explicitly.

```bash
python tools/bench/results/m2-max-30c-matrix-20261011/serve_ab.py \
  --model "$MONOLITH_MODELS/Qwen3.5-0.8B" \
  --off-log /tmp/m2-off.log --on-log /tmp/m2-on.log --out /tmp/m2-sse-ab.json
```

The additional mixed-workload gate used the backend's encoding default:

```bash
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_m2_matrix.py tests/runtime \
  tests/kernels/test_lm_drafter.py tests/kernels/test_spec_sampling.py \
  tests/kernels/test_spec_rollback.py \
  'tests/kernels/test_large_prefill.py::test_large_prefill_shares_state_with_small_decode[False-False-True-on-lm]' \
  'tests/kernels/test_large_prefill.py::test_large_prefill_shares_state_with_small_decode[False-True-True-on-dspark]'
```
