"""Regression tests for the 2026-11-19 code-review fixes."""

from __future__ import annotations

import math
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import chi2
from scvi import REGISTRY_KEYS
from torch.distributions import Normal

import mrtotalvi._counterfactual as cf
from mrtotalvi import MrTotalVI
from mrtotalvi._counterfactual import _max_component_log_prob
from mrtotalvi._stats import _wald_pvalue, get_outlier_cell_sample_pairs
from tests.test_minor_batch import _OutlierStub
from tests.test_mrtotalvi import MAX_EPOCHS_QUICK, N_LATENT, _make_adata
from tests.test_mrtotalvi_v2 import _make_context_v2_model, _make_enrichment_v2_model


# Enrichment group and donor summaries ----------------------------------------


def test_group_and_donor_summaries_skip_nan_sample_densities():
    """A factual-singleton NaN sample carries no weight in group or donor summaries."""
    model, adata = _make_enrichment_v2_model()
    # Cell 0 is the only retained sample_0 reference, so sample_0 is NaN for it.
    out = model.local_sample_enrichment(
        adata=adata,
        indices=[0],
        reference_indices=[0, 1, 2, 3],
        group_key="group",
        contrast=("numerator", "denominator"),
        donor_key="donor",
        inference_mode="posterior_mc",
        n_draws=3,
        random_state=7,
    )
    density = out["log_density"]
    assert np.isnan(density.sel(target_sample="sample_0")).all()
    d1, d2, d3 = (density.sel(target_sample=f"sample_{i}") for i in (1, 2, 3))
    assert np.isfinite(d1).all() and np.isfinite(d2).all() and np.isfinite(d3).all()
    numerator = out["group_log_density"].sel(group="numerator")
    denominator = out["group_log_density"].sel(group="denominator")
    np.testing.assert_allclose(numerator, d2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        denominator, np.logaddexp(d1, d3) - np.log(2.0), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(out["log_ratio"], numerator - denominator, rtol=1e-6, atol=1e-6)
    assert np.isnan(out["donor_log_ratio"].sel(donor="donor_0")).all()
    for name in ("donor_log_ratio_mean", "donor_log_ratio_median"):
        np.testing.assert_allclose(out[name], d2 - d3, rtol=1e-6, atol=1e-6)


# Marginal context cap ---------------------------------------------------------


def test_counterfactual_marginal_context_cap(monkeypatch):
    """Too many distinct marginal contexts raise; at the cap the call still runs."""
    model, adata = _make_context_v2_model()
    protein_names = [
        str(name)
        for name in model.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        ).column_names
    ]
    request = {
        "adata": adata,
        "indices": [0, 1],
        "target_samples": ["sample_0"],
        "gene_list": [str(adata.var_names[0])],
        "protein_list": protein_names[:2],
        "library_policy": "sample_balanced_marginal",
        "marginal_reference_indices": np.arange(adata.n_obs),
    }
    # size_factor is 1..12, so the library marginal has 12 distinct contexts.
    monkeypatch.setattr(cf, "MAX_CONTEXT_SCENARIOS", 11)
    with pytest.raises(ValueError, match="Subset marginal_reference_indices"):
        model.get_counterfactual_expression(**request)
    monkeypatch.setattr(cf, "MAX_CONTEXT_SCENARIOS", 12)
    assert "rna_rate" in model.get_counterfactual_expression(**request).data_vars


# DA donor centering -----------------------------------------------------------


def test_da_donor_centering_ignores_one_cell_sample():
    """A one-cell own sample (NaN) must not NaN the other samples of its donor."""
    adata = _make_adata()
    is_d0 = adata.obs["sample"].eq("donor_0").to_numpy()
    keep = ~is_d0
    keep[np.flatnonzero(is_d0)[0]] = True  # donor_0 keeps one cell
    adata = adata[keep].copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    adata.obs["patient"] = adata.obs["condition"]
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
    model.is_trained_ = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = model.differential_abundance(
            sample_cov_keys=["condition"],
            donor_key="patient",
            omit_original_sample=False,
            batch_size=64,
        )
    lone = adata.obs_names[adata.obs["sample"] == "donor_0"][0]
    assert np.isnan(out["log_probs"].sel(cell_name=lone, sample="donor_0"))
    assert np.isfinite(out["condition_log_probs"].values).all()

    # Expected aggregates from the uncentered output: centre within donor with nanmean,
    # then logmeanexp over the usable samples of each condition.
    sample_info = adata.obs[["sample", "condition", "patient"]].drop_duplicates("sample")
    log_probs = out["log_probs"]
    centered = log_probs.copy()
    for patient in sample_info["patient"].unique():
        samples = sample_info.loc[sample_info["patient"] == patient, "sample"].tolist()
        block = log_probs.sel(sample=samples)
        centered.loc[{"sample": samples}] = block - block.mean("sample", skipna=True)
    for condition in ("a", "b"):
        samples = sample_info.loc[sample_info["condition"] == condition, "sample"].tolist()
        block = centered.sel(sample=samples).values
        usable = ~np.isnan(block)
        expected = logsumexp(np.where(usable, block, -np.inf), axis=1) - np.log(usable.sum(1))
        np.testing.assert_allclose(
            out["condition_log_probs"].sel(condition=condition).values,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )


# Outlier cell-sample pairs ----------------------------------------------------


class _SortingLoaderStub(_OutlierStub):
    """A loader that returns cells in sorted index order."""

    def get_aggregated_posterior(self, adata, indices, batch_size):
        return super().get_aggregated_posterior(adata, np.sort(indices), batch_size)


def _outlier_data():
    rng = np.random.default_rng(0)
    labels = np.repeat(["s0", "s1", "s2"], 40)
    u = rng.normal(size=(labels.size, 2))
    obs = pd.DataFrame({"sample": labels}, index=[f"c{i}" for i in range(labels.size)])
    return u, labels, ad.AnnData(np.zeros((labels.size, 1), dtype=np.float32), obs=obs)


def test_outlier_subsample_is_independent_of_loader_order():
    u, labels, adata = _outlier_data()
    request = dict(
        adata=adata, sample_key="sample", subsample_size=10, random_state=0, batch_size=50
    )
    as_given = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), **request)
    sorted_loader = get_outlier_cell_sample_pairs(_SortingLoaderStub(u, labels), **request)
    np.testing.assert_array_equal(as_given["log_probs"].values, sorted_loader["log_probs"].values)


def test_outlier_pairs_absent_registered_sample_is_nan_inadmissible():
    u, labels, adata = _outlier_data()
    stub = _OutlierStub(u, labels)
    stub.sample_order = ["s0", "s1", "s2", "s_missing"]
    request = dict(adata=adata, sample_key="sample", subsample_size=None, batch_size=50)
    out = get_outlier_cell_sample_pairs(stub, **request)
    assert out["log_probs"].sel(sample="s_missing").isnull().all()
    assert not out["is_admissible"].sel(sample="s_missing").any()
    reference = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), **request)
    np.testing.assert_array_equal(
        out["log_probs"].sel(sample=["s0", "s1", "s2"]).values, reference["log_probs"].values
    )


# DE p-values ------------------------------------------------------------------


def test_de_pvalue_is_float64_chi2_survival():
    """p-values are chi2.sf of the Wald statistic in float64, df = analysed samples."""
    adata = _make_adata()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    de = model._legacy_differential_expression_for_reproducibility(
        sample_cov_keys=["condition"], mc_samples=1, batch_size=32
    )
    assert de["pvalue"].dtype == np.float64
    n_samples = adata.obs["sample"].nunique()
    expected = chi2.sf(de["effect_size"].values.astype(np.float64), n_samples)
    np.testing.assert_allclose(de["pvalue"].values, expected, rtol=1e-12, atol=0)


def test_wald_pvalue_keeps_the_upper_tail():
    """At statistic 100, df 4, 1 - cdf rounds to 0; the survival function does not."""
    pvalue = _wald_pvalue(np.array([100.0], dtype=np.float32), np.array([4]))
    np.testing.assert_allclose(pvalue, [9.836624224615988e-21], rtol=1e-12)


# Vectorised support scoring ---------------------------------------------------


def _pairwise_score(point, loc, scale, keep):
    """The per-pair loop the vectorised scorer replaced."""
    if not np.any(keep):
        return math.nan
    values = (
        Normal(
            torch.as_tensor(loc[keep], dtype=torch.float32),
            torch.as_tensor(scale[keep], dtype=torch.float32),
        )
        .log_prob(torch.as_tensor(point, dtype=torch.float32))
        .sum(dim=-1)
    )
    return float(values.max().item())


def test_max_component_log_prob_matches_pairwise_loop_across_chunks():
    rng = np.random.default_rng(0)
    n_components, n_latent, n_query = 1500, 10, 1505  # chunk is 1118 rows
    loc = rng.normal(size=(n_components, n_latent)).astype("f4")
    scale = np.exp(0.3 * rng.normal(size=(n_components, n_latent))).astype("f4")
    points = rng.normal(size=(n_query, n_latent)).astype("f4")
    names = np.array([f"c{i % 1400}" for i in range(n_components)])  # duplicate names
    query_names = np.array([f"c{i % 1400}" for i in range(n_query)])

    got = _max_component_log_prob(
        points, loc, scale, lambda a, b: names[None, :] == query_names[a:b, None]
    )
    expected = np.array(
        [_pairwise_score(points[i], loc, scale, names != query_names[i]) for i in range(n_query)]
    )
    np.testing.assert_array_equal(got, expected)

    leave_one_out = _max_component_log_prob(
        loc, loc, scale, lambda a, b: np.arange(a, b)[:, None] == np.arange(n_components)[None, :]
    )
    expected_loo = np.array(
        [_pairwise_score(loc[i], loc, scale, np.arange(n_components) != i)
         for i in range(n_components)]
    )
    np.testing.assert_array_equal(leave_one_out, expected_loo)


def test_max_component_log_prob_all_excluded_is_neg_inf():
    loc = np.zeros((3, 4), dtype="f4")
    scale = np.ones((3, 4), dtype="f4")
    got = _max_component_log_prob(
        np.zeros((2, 4), dtype="f4"), loc, scale, lambda a, b: np.ones((b - a, 3), dtype=bool)
    )
    assert np.all(np.isneginf(got))
