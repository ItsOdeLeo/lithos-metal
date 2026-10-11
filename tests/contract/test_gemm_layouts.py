"""gemm_tile's two cooperative-tensor layouts (kernels/common/gemm_tile.metal): the backend being compiled for picks
the layout, the macros carry it, and the reduction order x_permute writes is a permutation whose inverse is the
producers' perm_dest (PERM_OUT), for both layouts. No GPU: the formulas are mirrored in numpy."""

import numpy as np
import pytest

from monolith import kernels
from monolith.backends.metal.context import using_backend
from monolith.bench import pack_spec, random_spec
from monolith.formats import PackLayout


def _info(fmt="nvfp4", n=256, k=4096):
    return pack_spec(random_spec(fmt, n, k, np.random.default_rng(0)), PackLayout(rows=16))[1]


@pytest.mark.parametrize("backend,layout", [("m2_max_30c", "apple9"), ("m3_pro", "apple9"), ("m3_max_30c", "apple9"), ("m4_pro", "apple9"),
                                            ("m5_pro", "apple10"), ("m5_max_40c", "apple10")])
def test_the_backend_picks_the_layout(backend, layout, monkeypatch):
    # Cross-compilation must follow the selected backend, even on a different host GPU.
    monkeypatch.setattr(kernels, "_device_family", lambda: 10 if layout == "apple9" else 8)
    with using_backend(backend):
        assert kernels.coop_layout() == layout
        macros = kernels.gemm_macros(_info(), tm=16)
        assert (macros.get("COOP_A9") == "1") == (layout == "apple9")
        assert ("PERM_A9" in kernels.perm_out_macros(4096, 32, 256)) == (layout == "apple9")


@pytest.mark.parametrize("family,layout", [(8, "apple9"), (9, "apple9"), (10, "apple10"), (None, "apple10")])
def test_standalone_harness_picks_the_host_layout(monkeypatch, family, layout):
    monkeypatch.setattr(kernels, "_device_family", lambda: family)
    with using_backend("common"):
        assert kernels.coop_layout() == layout


@pytest.mark.parametrize("fmt,tk", [("bf16", 64), ("fp8_e4m3", 64), ("nvfp4", 128), ("int8", 128), ("int4_affine", 128)])
@pytest.mark.parametrize("tm", [8, 16, 32])
def test_m2_tiles_and_explicit_diagnostic_override(fmt, tk, tm):
    info = _info(fmt, k=5120)
    with using_backend("m2_max_30c"):
        macros = kernels.gemm_macros(info, tm=tm)
        assert (macros["TN"], macros["TK"]) == ("16u", f"{tk}u")
        explicit = kernels.gemm_macros(info, tm=tm, tn=32, tk=128)
        assert (explicit["TN"], explicit["TK"]) == ("32u", "128u")
    # M2's measured override must not change other backends' defaults.
    with using_backend("m4_pro"):
        assert kernels.gemm_macros(info, tm=32)["TN"] == "32u"


def _perm_dest(n, k, wpw, tk, layout):
    """numpy mirror of PERM_OUT_MSL's perm_dest: natural column n -> slot in x'."""
    kl, span = k // 32, 32 * wpw
    l, o = n // kl, n % kl
    j, e = o // wpw, o % wpw
    p = j * span + l * wpw + e
    kt, r = p // tk, p % tk
    if layout == "apple9":
        slot = r // (tk // 8) + 8 * (r % (tk // 8))
    else:
        mq, r2 = r // (tk // 4), r % (tk // 4)
        slot = (r2 & 3) | ((mq & 1) << 2) | ((mq >> 1) << 3) | ((r2 >> 2) << 4)
    return kt * tk + slot


@pytest.mark.parametrize("layout", ["apple9", "apple10"])
@pytest.mark.parametrize("k,wpw,tk", [(4096, 32, 256), (5120, 32, 128), (2048, 16, 256), (1024, 8, 64)])
def test_x_permute_is_the_inverse_of_perm_dest(layout, k, wpw, tk):
    perm = kernels.x_permute_columns(k, wpw, tk, layout=layout)
    assert sorted(perm.tolist()) == list(range(k))
    dest = _perm_dest(np.arange(k), k, wpw, tk, layout)
    assert np.array_equal(dest[perm], np.arange(k))


def test_apple9_rejects_quarter_word_tiles():
    with using_backend("m4_pro"):
        info = _info()
        with pytest.raises(ValueError, match="on Apple9"):
            kernels.gemm_macros(info, tm=8, tn=64, tk=64)
        assert kernels.gemm_macros(info, tm=8)["TK"] == "128u"
