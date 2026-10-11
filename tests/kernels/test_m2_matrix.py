"""The measured M2 tile geometry and legal whole-head GDN dispatch."""
import copy
import math

import numpy as np
import pytest

from monolith.backends.metal import using_backend
from monolith.bench import check_against_oracle
from tests.kernels.test_gemm_tile import _run, dev


@pytest.fixture(scope="module", autouse=True)
def m2_device(dev):
    info = dev.info()
    if (info.name, info.gpu_cores, info.apple_family) != ("Apple M2 Max", 30, 8):
        pytest.skip("qualification for the 30-core M2 Max")


@pytest.mark.parametrize("fmt", ["bf16", "fp8_e4m3", "nvfp4", "int8", "int4_affine"])
@pytest.mark.parametrize("tm,active", [(8, 5), (16, 13), (32, 29)])
def test_m2_selected_tiles_preserve_matrix_and_token_tails(dev, fmt, tm, active):
    with using_backend("m2_max_30c"):
        got, ref = _run(dev, fmt, 269, 4096, tm, active, "interleaved16", placement="block")
    assert check_against_oracle(got[:active], ref).ok()
    assert not got[active:].any()


@pytest.mark.parametrize("hk,hv,separate", [(8, 16, False), (16, 48, True)])
@pytest.mark.parametrize("perm", [None, (8, 64)])
def test_m2_whole_head_gdn_matches_separate_norm_and_state(dev, hk, hv, separate, perm):
    torch = pytest.importorskip("torch")
    from tests.kernels.test_gdn_mixer import Harness, _module, _proj, CW

    m, rng = _module(64, hk, hv, 128, 128, seed=91)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (hv, 128, 128)).astype(np.float32))}
    with using_backend("m2_max_30c"):
        harnesses = [Harness(dev, m, 1, prepared=True, slice_cols=16, slices_per_block=1,
                             tokens_per_pass=1, perm_out=perm, fused_norm=fused, ab_separate=separate,
                             local_prepare=fused, local_groups=8, single_pass=fused, specialize=True)
                     for fused in (False, True)]
        for h in harnesses:
            h.set_state(state)
        for active, done in [(1, False), (1, False), (0, False), (1, False), (1, True)]:
            proj = _proj(rng, torch, 1, m.in_proj.n)
            np.testing.assert_array_equal(*[h.step(proj, t_active=active, done=done) for h in harnesses])
            assert harnesses[1].pso.max_threads_per_threadgroup >= 256
            for attr, size in (("conv_state", 2 * harnesses[0].conv_bytes), ("rec_state", 2 * harnesses[0].rec_bytes)):
                assert getattr(harnesses[0], attr).read(0, size) == getattr(harnesses[1], attr).read(0, size)


@pytest.mark.parametrize("t,slices", [(4, 4), (6, 2), (8, 2)])
@pytest.mark.parametrize("hk,hv,separate", [(8, 16, False), (16, 48, True)])
def test_m2_local_gdn_matches_separate_prepare_with_partial_tokens(dev, t, slices, hk, hv, separate):
    torch = pytest.importorskip("torch")
    from tests.kernels.test_gdn_mixer import Harness, _module, _proj, CW

    m, rng = _module(64, hk, hv, 128, 128, seed=101 + t)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (hv, 128, 128)).astype(np.float32))}
    with using_backend("m2_max_30c"):
        harnesses = [Harness(dev, m, t, prepared=True, slice_cols=slices, slices_per_block=1,
                             tokens_per_pass=t, local_prepare=local, local_groups=8,
                             single_pass=local, specialize=True, ab_separate=separate) for local in (False, True)]
        for h in harnesses:
            h.set_state(state)
        for active, done in [(t, False), (t - 1, False), (1, False), (0, False), (t, True)]:
            proj = _proj(rng, torch, t, m.in_proj.n)
            np.testing.assert_array_equal(*[h.step(proj, t_active=active, done=done) for h in harnesses])
            assert harnesses[1].pso.max_threads_per_threadgroup >= 256
            for attr, size in (("conv_state", 2 * harnesses[0].conv_bytes), ("rec_state", 2 * harnesses[0].rec_bytes)):
                assert getattr(harnesses[0], attr).read(0, size) == getattr(harnesses[1], attr).read(0, size)


@pytest.fixture(scope="module")
def hybrid_pack(tmp_path_factory, m2_device):
    from tests.contract import test_nn_lowering as synth
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.nn.pack_plan import pack_model
    from monolith.formats import PackLayout
    from monolith.packs import PackFile

    path = tmp_path_factory.mktemp("m2-matrix-hybrid")
    config = copy.deepcopy(synth.CFG)
    config["text_config"].update(linear_num_key_heads=8, linear_num_value_heads=16,
                               linear_key_head_dim=128, linear_value_head_dim=128)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(synth, "CFG", config)
        synth._checkpoint(path)
    model = Qwen3_5Model.from_checkpoint(str(path), max_context=256)
    pack_model(model, str(path), str(path / "pack"), PackLayout(lane_order="interleaved16", scale_placement="block"))
    return model, PackFile(path / "pack")


@pytest.mark.parametrize("t", [1, 4, 6, 8, 128])
@pytest.mark.parametrize("accelerator", ["off", "on"])
def test_m2_compiled_dispatches_fit_actual_pipeline_limits(dev, hybrid_pack, t, accelerator):
    from monolith.backends.metal import get_backend, load_configs
    from monolith.compiler import compile_program
    from monolith.core import StepStateLayout
    from monolith.runtime.engine import compile_pipelines
    from monolith.runtime.program import Program

    model, pack = hybrid_pack
    config = load_configs()["apple-m2-max-30c"]
    program = compile_program(model, pack, config,
                              t=t, dynamic_t=t > 1, accelerator=accelerator,
                              layout=StepStateLayout(t_max=max(8, t)))
    # The retiled parameters and constants must survive the runtime ABI round-trip.
    before = program.to_json()
    get_backend(config.backend).finalize(program, None, {}, config)
    assert program.to_json() == before
    program = Program.from_json(program.to_json())
    pipelines = compile_pipelines(program, dev)
    for op in program.ops:
        assert math.prod(op.threadgroup) <= pipelines[op.kernel].max_threads_per_threadgroup, op.kernel
