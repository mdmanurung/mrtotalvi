"""Opt-in posterior readout E[native | y, z] of get_protein_noise_decomposition (EXPERIMENTAL)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from mrtotalvi import MrTotalVI
from mrtotalvi._module import _posterior_native
from tests._simulate_adt import simulate_citeseq


def _nb_pmf(y, mu, theta):
    # Independent of the package: scipy's (n, p) form of the NB with mean mu.
    return nbinom.pmf(y, theta, theta / (theta + mu))


def _t(value):
    return torch.tensor([[float(value)]])


def _case(y, split="excess"):
    # Design note: theta=5, m1=2, f=10, logit=0 (pi=0.5), eff=1 (already inside m1).
    return _posterior_native(_t(y), _t(2.0), _t(10.0), _t(5.0), _t(0.0), split)


def test_design_note_responsibility_matches_scipy():
    fg, bg = _nb_pmf(10, 20.0, 5.0), _nb_pmf(10, 2.0, 5.0)
    expected = 0.5 * fg / (0.5 * bg + 0.5 * fg)
    assert expected == pytest.approx(0.98076, abs=1e-5)
    _, resp, _ = _case(10)
    assert resp.item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(("split", "value"), [("excess", 8.8268), ("whole", 9.8076)])
def test_design_note_readout(split, value):
    res, _, _ = _case(10, split)
    assert res.item() == pytest.approx(value, abs=1e-3)


def test_zero_count_responsibility():
    _, resp, _ = _case(0)
    fg, bg = _nb_pmf(0, 20.0, 5.0), _nb_pmf(0, 2.0, 5.0)
    assert resp.item() == pytest.approx(fg / (fg + bg), abs=1e-6)
    assert resp.item() == pytest.approx(0.001718, abs=1e-6)


@pytest.fixture(scope="module")
def fitted():
    adata, _, _ = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=200)
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    model = MrTotalVI(adata, sample_key="donor", n_latent=4)
    model.train(max_epochs=2, accelerator="cpu", batch_size=64)
    return model, adata


COMBOS = [(s, b) for s in ("excess", "whole") for b in ("sampled", "lognormal_mean")]


def _read(model, adata, **kwargs):
    torch.manual_seed(0)
    return model.get_protein_noise_decomposition(adata, n_samples=5, **kwargs)


@pytest.mark.parametrize(("split", "background"), COMBOS)
def test_readout_bounded_by_counts(fitted, split, background):
    model, adata = fitted
    out = _read(model, adata, readout="posterior", split=split, background=background)
    assert set(out) == {"native"}
    native = out["native"].to_numpy()
    counts = np.asarray(adata.obsm["protein"], dtype=float)[:, : native.shape[1]]
    assert np.isfinite(native).all()
    assert (native >= 0).all()
    assert (native <= counts * (1 + 1e-5) + 1e-4).all()  # float32 rounding


def test_default_unchanged(fitted):
    model, adata = fitted
    default = _read(model, adata)
    explicit = _read(model, adata, readout="decoder")
    assert set(default) == {"native", "ambient", "nonspecific", "observed", "foreground_probability"}
    for key in default:
        assert default[key].equals(explicit[key])


@pytest.mark.parametrize(
    "kwargs",
    [{"readout": "mean"}, {"split": "half"}, {"background": "alpha"}],
)
def test_invalid_options_raise(fitted, kwargs):
    model, adata = fitted
    with pytest.raises(ValueError):
        model.get_protein_noise_decomposition(adata, n_samples=2, **kwargs)


def test_non_totalvi_mode_raises(fitted):
    model, adata = fitted
    old = model.module.protein_noise_model
    model.module.protein_noise_model = "ambient"
    try:
        with pytest.raises(NotImplementedError, match="totalvi"):
            model.get_protein_noise_decomposition(adata, n_samples=2, readout="posterior")
    finally:
        model.module.protein_noise_model = old


def test_posterior_readout_with_size_factor_scales_native_only(fitted):
    # Codex A6: the posterior readout returns native only; size_factor must not index the other components.
    model, adata = fitted
    plain = _read(model, adata, readout="posterior")
    scaled = _read(model, adata, readout="posterior", size_factor="total_adt")
    assert set(scaled) == {"native"}
    factor = model._protein_size_factor(adata)
    np.testing.assert_allclose(scaled["native"].values, plain["native"].values / factor[:, None], rtol=1e-6)
