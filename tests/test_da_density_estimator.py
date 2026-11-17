"""Opt-in DA density estimators: default parity, known answers without training, validation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from mrtotalvi._stats import differential_abundance
from tests.test_mrtotalvi import _make_adata, _setup_and_train

POSTERIOR_SCALE = 0.05  # per-cell posterior scale, far below neighbour spacing (~0.3)
FLOOR = 0.4
K = 50
SAMPLES = ["a1", "a2", "b1", "b2"]


class StubModel:
    """Latent ``u`` and per-cell posterior scales are set directly; nothing is trained."""

    def __init__(self, u, samples, scale=POSTERIOR_SCALE):
        self.u = u.astype(np.float32)
        self.samples = np.asarray(samples)
        self.scale = scale
        self.sample_order = list(dict.fromkeys(samples))

    def _validate_anndata(self, adata):
        return adata

    def get_latent_representation(self, adata, **kwargs):
        return self.u

    def get_aggregated_posterior(self, adata, sample, batch_size):
        loc = torch.as_tensor(self.u[self.samples == sample])
        scale = torch.full_like(loc, self.scale)
        probs = torch.ones(loc.shape[0]) / loc.shape[0]
        return MixtureSameFamily(Categorical(probs=probs), Independent(Normal(loc, scale), 1))


def _stub_data(depleted: bool, n_per_sample=600, seed=0):
    """Samples a*, b* ~ N(0, I) in 2-D. If ``depleted``, b* lose 75% of cells with u0 > 0.

    Truth for log(p_a / p_b): +log(1 / 0.4) = +0.916 where u0 > 0, and
    log(0.625) = -0.470 where u0 < 0 (b is renormalised by 0.625).
    """
    rng = np.random.default_rng(seed)
    us, names = [], []
    for name in SAMPLES:
        x = rng.normal(size=(8 * n_per_sample, 2))
        if depleted and name.startswith("b"):
            x = x[(x[:, 0] < 0) | (rng.random(len(x)) < 0.25)]
        us.append(x[:n_per_sample])
        names += [name] * n_per_sample
    u = np.vstack(us)
    obs = pd.DataFrame(
        {"sample": names, "cond": [n[0] for n in names]},
        index=[f"c{i}" for i in range(len(names))],
    )
    adata = ad.AnnData(np.zeros((len(names), 1), dtype=np.float32), obs=obs)
    return StubModel(u, names), adata, u


def _enrich(estimator, depleted, **kwargs):
    model, adata, u = _stub_data(depleted)
    da = differential_abundance(
        model,
        adata=adata,
        sample_key="sample",
        sample_cov_keys=["cond"],
        compute_log_enrichment=True,
        density_estimator=estimator,
        batch_size=400,
        **kwargs,
    )
    return da["cond_log_enrichs"].sel(cond_enrichment="a").values, u


ESTIMATORS = {
    "knn": {"knn_k": K},
    "scale_floor": {"scale_floor": FLOOR},
}


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_null_gives_zero_enrichment(estimator):
    enrich, _ = _enrich(estimator, depleted=False, **ESTIMATORS[estimator])
    assert abs(np.nanmean(enrich)) < 0.1
    assert np.nanstd(enrich) < 0.6


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_depletion_has_right_sign_and_size(estimator):
    enrich, u = _enrich(estimator, depleted=True, **ESTIMATORS[estimator])
    # b is depleted for u0 > 0, so a is enriched there; interior regions avoid the step edge
    assert np.mean(enrich[u[:, 0] > 1]) == pytest.approx(np.log(1 / 0.4), abs=0.3)
    assert np.mean(enrich[u[:, 0] < -1]) == pytest.approx(np.log(0.625), abs=0.3)


def test_posterior_is_noise_dominated_on_the_null():
    """Documented behaviour: small posterior scales give a null spread far above the others."""
    posterior, _ = _enrich("posterior", depleted=False)
    knn, _ = _enrich("knn", depleted=False)
    floor, _ = _enrich("scale_floor", depleted=False, scale_floor=FLOOR)
    assert np.std(posterior) > 5 * np.std(knn)
    assert np.std(posterior) > 5 * np.std(floor)


def test_knn_excludes_self_and_own_sample():
    """1-D layout. A = {0, 1, 10, 11}, B = {5, 6, 20, 21}, k = 2 (omit False first)."""
    x = np.array([0, 1, 10, 11, 5, 6, 20, 21], dtype=float)[:, None]
    names = ["A"] * 4 + ["B"] * 4
    obs = pd.DataFrame({"sample": names}, index=[f"c{i}" for i in range(8)])
    adata = ad.AnnData(np.zeros((8, 1), dtype=np.float32), obs=obs)
    model = StubModel(x, names)

    def run(omit):
        return differential_abundance(
            model,
            adata=adata,
            sample_key="sample",
            density_estimator="knn",
            knn_k=2,
            omit_original_sample=omit,
        )["log_probs"].values

    lp = run(False)
    # cell 0 (A at 0): neighbours excluding itself are A@1 and B@5, so k_A = k_B = 1
    # (if it counted itself the neighbours would be itself and A@1, k_A = 2)
    np.testing.assert_allclose(lp[0], [np.log(1.5) - np.log(3), np.log(1.5) - np.log(4)])
    lp = run(True)
    # own sample is omitted: no A column for A cells; both neighbours come from B
    assert np.isnan(lp[0, 0])
    np.testing.assert_allclose(lp[0, 1], np.log(2.5) - np.log(4))


def test_default_equals_explicit_posterior_on_trained_model():
    adata = _make_adata()
    adata.obs["condition"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_2"]), "a", "b")
    model = _setup_and_train(adata, max_epochs=1)
    kwargs = {"sample_cov_keys": ["condition"], "compute_log_enrichment": True}
    with pytest.warns(UserWarning, match="descriptive"):
        default = model.differential_abundance(**kwargs)
    with pytest.warns(UserWarning, match="descriptive"):
        explicit = model.differential_abundance(density_estimator="posterior", **kwargs)
    assert list(default.data_vars) == list(explicit.data_vars)
    for name in default.data_vars:
        assert np.array_equal(default[name].values, explicit[name].values)
    assert default.attrs == explicit.attrs
    assert "density_estimator" not in default.attrs
    with pytest.warns(UserWarning, match="descriptive"):
        knn = model.differential_abundance(density_estimator="knn", knn_k=5, **kwargs)
    assert knn.attrs["density_estimator"] == "knn"
    assert set(knn.data_vars) == set(default.data_vars)
    assert not np.array_equal(knn["log_probs"].values, default["log_probs"].values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"density_estimator": "kde"}, "must be 'posterior'"),
        ({"density_estimator": "scale_floor"}, "needs scale_floor > 0"),
        ({"density_estimator": "scale_floor", "scale_floor": -1.0}, "needs scale_floor > 0"),
        ({"density_estimator": "posterior", "scale_floor": 0.4}, "only used with"),
        ({"density_estimator": "knn", "knn_k": 0}, "knn_k must be"),
        ({"density_estimator": "knn", "n_mc_samples": 2}, "n_mc_samples=1"),
        (
            {"density_estimator": "knn", "donor_key": "sample", "omit_original_sample": False},
            "donor_key is not supported",
        ),
        ({"density_estimator": "knn", "knn_k": 10_000}, "reference cells"),
    ],
)
def test_argument_validation(kwargs, match):
    model, adata, _ = _stub_data(depleted=False, n_per_sample=20)
    with pytest.raises(ValueError, match=match):
        differential_abundance(model, adata=adata, sample_key="sample", **kwargs)
