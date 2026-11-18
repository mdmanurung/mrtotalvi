"""Opt-in DA density estimators: default parity, known answers without training, validation."""

from __future__ import annotations

from pathlib import Path

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
        ({"density_estimator": "posterior", "knn_k": -3}, "knn_k must be"),
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


# --- leave-one-cell-out for omit_original_sample=False, NaN for empty groups -------------------

DEFAULT_GOLDEN = Path(__file__).parent / "data" / "da_default_golden.npz"


def test_default_outputs_are_byte_identical_to_pre_fix_golden():
    """Golden outputs were written by the code before the leave-one-cell-out change."""
    model, adata, _ = _stub_data(depleted=True, n_per_sample=60)
    golden = np.load(DEFAULT_GOLDEN)
    n_checked = 0
    for est, kw in [("posterior", {}), ("scale_floor", {"scale_floor": FLOOR}), ("knn", {"knn_k": 20})]:
        da = differential_abundance(
            model,
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["cond"],
            compute_log_enrichment=True,
            density_estimator=est,
            batch_size=100,
            **kw,
        )
        for name in da.data_vars:
            assert np.array_equal(da[name].values, golden[f"{est}__{name}"], equal_nan=True)
            n_checked += 1
    assert n_checked == len(golden.files)


def _own_group_enrichment(estimator=None, donor_key=None, **kwargs):
    """Null data: every sample ~ N(0, I). Mean enrichment of each cell for its own group."""
    # 10-D, posterior scale 0.15: the kernel is narrow against the sample spacing, as in
    # real models, so a cell's own kernel dominates its own sample's density.
    rng = np.random.default_rng(1)
    names = [n for n in SAMPLES for _ in range(250)]
    u = rng.normal(size=(len(names), 10))
    obs = pd.DataFrame(
        {"sample": names, "cond": [n[0] for n in names]},
        index=[f"c{i}" for i in range(len(names))],
    )
    adata = ad.AnnData(np.zeros((len(names), 1), dtype=np.float32), obs=obs)
    model = StubModel(u, names, scale=0.15)
    adata.obs["donor"] = [n[1] for n in adata.obs["sample"]]  # a1/b1 -> "1", a2/b2 -> "2"
    da = differential_abundance(
        model,
        adata=adata,
        sample_key="sample",
        sample_cov_keys=["cond"],
        compute_log_enrichment=True,
        omit_original_sample=False,
        density_estimator=estimator or "posterior",
        donor_key="donor" if donor_key else None,
        batch_size=400,
        **kwargs,
    )
    enrich = da["cond_log_enrichs"]
    own = adata.obs["cond"].to_numpy()
    return np.where(own == "a", enrich.sel(cond_enrichment="a"), enrich.sel(cond_enrichment="b"))


@pytest.mark.parametrize(
    ("kwargs", "before_fix_nats"),
    [
        ({}, 89.4),
        ({"estimator": "scale_floor", "scale_floor": FLOOR}, 12.3),
        ({"donor_key": True}, 51.7),
    ],
    ids=["posterior", "scale_floor", "donor_key"],
)
def test_null_own_group_enrichment_is_centred_without_own_kernel(kwargs, before_fix_nats):
    own = _own_group_enrichment(**kwargs)
    assert abs(np.mean(own)) < 5.0  # before the fix: +before_fix_nats
    assert np.mean(own > 0) < 0.7  # before the fix: ~100% of cells above 0


def test_loo_removes_exactly_the_own_component():
    """Three cells in sample A, two in B, 1-D. Check against the closed form."""
    x = np.array([0.0, 0.5, 2.0, 1.0, 3.0])[:, None]
    names = ["A", "A", "A", "B", "B"]
    obs = pd.DataFrame({"sample": names}, index=[f"c{i}" for i in range(5)])
    adata = ad.AnnData(np.zeros((5, 1), dtype=np.float32), obs=obs)
    model = StubModel(x, names, scale=0.7)
    lp = differential_abundance(
        model, adata=adata, sample_key="sample", omit_original_sample=False, batch_size=2
    )["log_probs"].values.astype(float)

    def density(at, locs):
        return np.mean(np.exp(-0.5 * ((at - locs) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi)))

    a, b = x[:3, 0], x[3:, 0]
    for i in range(5):
        others_a = np.delete(a, i) if i < 3 else a
        others_b = np.delete(b, i - 3) if i >= 3 else b
        np.testing.assert_allclose(lp[i, 0], np.log(density(x[i, 0], others_a)), rtol=1e-5)
        np.testing.assert_allclose(lp[i, 1], np.log(density(x[i, 0], others_b)), rtol=1e-5)


def test_group_of_only_the_own_sample_is_nan_not_inf():
    """One sample per condition: under omit_original_sample=True a cell's own group is empty."""
    model, adata, _ = _stub_data(depleted=False, n_per_sample=30)
    adata = adata[adata.obs["sample"].isin(["a1", "b1"])].copy()
    keep = np.isin(model.samples, ["a1", "b1"])
    model = StubModel(model.u[keep], model.samples[keep])
    with pytest.warns(UserWarning, match="no reference sample"):
        da = differential_abundance(
            model,
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["cond"],
            compute_log_enrichment=True,
            batch_size=100,
        )
    assert not np.isinf(da["cond_log_probs"].values).any()
    assert not np.isinf(da["cond_log_enrichs"].values).any()
    own = adata.obs["cond"].to_numpy() == "a"
    assert np.isnan(da["cond_log_probs"].sel(cond="a").values[own]).all()
    assert np.isfinite(da["cond_log_probs"].sel(cond="a").values[~own]).all()
    assert np.isnan(da["cond_log_enrichs"].values).all()  # every cell has one empty group


def test_one_cell_own_sample_is_nan_under_loo():
    x = np.array([0.0, 1.0, 2.0])[:, None]
    names = ["A", "B", "B"]
    obs = pd.DataFrame({"sample": names, "cond": ["a", "b", "b"]}, index=["c0", "c1", "c2"])
    adata = ad.AnnData(np.zeros((3, 1), dtype=np.float32), obs=obs)
    with pytest.warns(UserWarning, match="no reference sample"):
        da = differential_abundance(
            StubModel(x, names),
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["cond"],
            omit_original_sample=False,
            batch_size=3,
        )
    assert np.isnan(da["log_probs"].values[0, 0])
    assert np.isnan(da["cond_log_probs"].sel(cond="a").values[0])
    assert np.isfinite(da["cond_log_probs"].sel(cond="a").values[1:]).all()
