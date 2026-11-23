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
    np.testing.assert_allclose(denominator, np.logaddexp(d1, d3) - np.log(2.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["log_ratio"], numerator - denominator, rtol=1e-6, atol=1e-6)
    assert np.isnan(out["donor_log_ratio"].sel(donor="donor_0")).all()
    for name in ("donor_log_ratio_mean", "donor_log_ratio_median"):
        np.testing.assert_allclose(out[name], d2 - d3, rtol=1e-6, atol=1e-6)


# Marginal context cap ---------------------------------------------------------


def test_counterfactual_marginal_context_cap(monkeypatch):
    """Too many distinct marginal contexts raise; at the cap the call still runs."""
    model, adata = _make_context_v2_model()
    protein_names = [
        str(name) for name in model.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names
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
    adata.obs["condition"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b")
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
    request = {"adata": adata, "sample_key": "sample", "subsample_size": 10, "random_state": 0, "batch_size": 50}
    as_given = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), **request)
    sorted_loader = get_outlier_cell_sample_pairs(_SortingLoaderStub(u, labels), **request)
    np.testing.assert_array_equal(as_given["log_probs"].values, sorted_loader["log_probs"].values)


def test_outlier_pairs_absent_registered_sample_is_nan_inadmissible():
    u, labels, adata = _outlier_data()
    stub = _OutlierStub(u, labels)
    stub.sample_order = ["s0", "s1", "s2", "s_missing"]
    request = {"adata": adata, "sample_key": "sample", "subsample_size": None, "batch_size": 50}
    out = get_outlier_cell_sample_pairs(stub, **request)
    assert out["log_probs"].sel(sample="s_missing").isnull().all()
    assert not out["is_admissible"].sel(sample="s_missing").any()
    reference = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), **request)
    np.testing.assert_array_equal(out["log_probs"].sel(sample=["s0", "s1", "s2"]).values, reference["log_probs"].values)


# DE p-values ------------------------------------------------------------------


def test_de_pvalue_is_float64_chi2_survival():
    """p-values are chi2.sf of the Wald statistic in float64, df = analysed samples."""
    adata = _make_adata()
    adata.obs["condition"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b")
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

    got = _max_component_log_prob(points, loc, scale, lambda a, b: names[None, :] == query_names[a:b, None])
    expected = np.array([_pairwise_score(points[i], loc, scale, names != query_names[i]) for i in range(n_query)])
    np.testing.assert_array_equal(got, expected)

    leave_one_out = _max_component_log_prob(
        loc, loc, scale, lambda a, b: np.arange(a, b)[:, None] == np.arange(n_components)[None, :]
    )
    expected_loo = np.array(
        [_pairwise_score(loc[i], loc, scale, np.arange(n_components) != i) for i in range(n_components)]
    )
    np.testing.assert_array_equal(leave_one_out, expected_loo)


def test_max_component_log_prob_all_excluded_is_neg_inf():
    loc = np.zeros((3, 4), dtype="f4")
    scale = np.ones((3, 4), dtype="f4")
    got = _max_component_log_prob(
        np.zeros((2, 4), dtype="f4"), loc, scale, lambda a, b: np.ones((b - a, 3), dtype=bool)
    )
    assert np.all(np.isneginf(got))


# Backed AnnData row order -----------------------------------------------------

# n_obs is 8 in the v2 fixture; the request has a duplicate and is not ascending.
BACKED_REQUEST = np.array([7, 2, 5, 0, 6, 1, 3, 4, 2])


@pytest.mark.parametrize("batch_size", [4, 256])  # 4 crosses minibatch boundaries
def test_backed_adata_returns_rows_in_request_order(tmp_path, batch_size):
    """scvi sorts backed minibatches; outputs must still follow the requested order."""
    from mrtotalvi._counterfactual import _collect_observed_context, _encode_u_params
    from mrtotalvi._stats import collect_u_posterior
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)
    ascending = np.arange(backed.n_obs)

    # Expected rows come from the ascending request, where the backed sort is a no-op.
    ref_loc, ref_scale, ref_sample = _encode_u_params(model, backed, ascending, batch_size)
    loc, scale, sample = _encode_u_params(model, backed, BACKED_REQUEST, batch_size)
    np.testing.assert_allclose(loc, ref_loc[BACKED_REQUEST], atol=1e-6)
    np.testing.assert_allclose(scale, ref_scale[BACKED_REQUEST], atol=1e-6)
    np.testing.assert_array_equal(sample, ref_sample[BACKED_REQUEST])

    ref_context = _collect_observed_context(model, backed, ascending, batch_size)
    context = _collect_observed_context(model, backed, BACKED_REQUEST, batch_size)
    for key in ("batch", "sample", "library"):
        np.testing.assert_array_equal(context[key], ref_context[key][BACKED_REQUEST])

    for give_z in (False, True):
        latent = model.get_latent_representation(
            adata=backed, indices=BACKED_REQUEST, give_z=give_z, batch_size=batch_size
        )
        ref_latent = model.get_latent_representation(
            adata=backed, indices=ascending, give_z=give_z, batch_size=batch_size
        )
        np.testing.assert_allclose(latent, ref_latent[BACKED_REQUEST], atol=1e-6)

    posterior_loc, _ = collect_u_posterior(model, adata=backed, indices=BACKED_REQUEST, batch_size=batch_size)
    np.testing.assert_allclose(posterior_loc.numpy(), ref_loc[BACKED_REQUEST], atol=1e-6)


@pytest.mark.parametrize("batch_size", [4, 256])
def test_backed_counterfactual_outputs_match_in_memory(tmp_path, batch_size):
    """Backed requests follow the request order, and a repeated cell counts per request."""
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    adata.obs["group"] = np.where(np.arange(adata.n_obs) % 2 == 0, "even", "odd")
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)

    # In-memory data loads rows as requested, so it is the reference for backed data.
    rep = model.get_local_sample_representation(backed, indices=BACKED_REQUEST, batch_size=batch_size)
    ref_rep = model.get_local_sample_representation(adata, indices=BACKED_REQUEST, batch_size=batch_size)
    assert list(rep.cell_name.values) == list(adata.obs_names[BACKED_REQUEST])
    np.testing.assert_allclose(rep.values, ref_rep.values, atol=1e-6)

    dist = model.get_local_sample_distances(backed, indices=BACKED_REQUEST, batch_size=batch_size, groupby="group")
    ref_dist = model.get_local_sample_distances(adata, indices=BACKED_REQUEST, batch_size=batch_size, groupby="group")
    assert list(dist.cell_name.values) == list(adata.obs_names[BACKED_REQUEST])
    np.testing.assert_allclose(dist["cell"].values, ref_dist["cell"].values, atol=1e-6)
    np.testing.assert_allclose(
        dist["group_distances"].sel(group=["even", "odd"]).values,
        ref_dist["group_distances"].sel(group=["even", "odd"]).values,
        atol=1e-6,
    )

    corr = model.get_feature_correlation_matrix(backed, indices=BACKED_REQUEST, n_samples=2, batch_size=batch_size)
    assert np.isfinite(corr.to_numpy()).sum() > 0


# Marginal scenario memory and context hash -----------------------------------


class _ContextStubModel:
    module = type("Module", (), {"panel_key": "panel"})()

    class adata_manager:
        @staticmethod
        def get_state_registry(key):
            return type("Registry", (), {"categorical_mapping": ["a", "b", "c"]})()


def _marginal_request(n_query, n_ref, *, library_policy="sample_balanced_marginal"):
    """Reference cells with distinct library sizes, so each is its own context."""
    rng = np.random.default_rng(3)
    reference = {
        "batch": np.arange(n_ref) % 3,
        "panel": np.arange(n_ref) % 3,
        "library": (100.0 + rng.random(n_ref) * 50).astype(np.float32),
        "sample": np.arange(n_ref) % 2,
    }
    query = {
        "batch": np.arange(n_query) % 3,
        "panel": np.arange(n_query) % 3,
        "library": np.linspace(100, 200, n_query).astype(np.float32),
    }
    return {
        "model": _ContextStubModel,
        "observed_context": query,
        "batch_policy": "sample_balanced_marginal",
        "specified_batch": None,
        "panel_policy": "sample_balanced_marginal",
        "specified_panel": None,
        "library_policy": library_policy,
        "specified_library_size": None,
        "reference_context": reference,
    }


@pytest.mark.parametrize("library_policy", ["sample_balanced_marginal", "observed"])
def test_context_hash_matches_full_json_payload(library_policy):
    """The streamed hash equals sha256 of the full JSON payload the old code built."""
    import hashlib
    import json

    scenarios, digest = cf._build_context_scenarios(**_marginal_request(7, 5, library_policy=library_policy))
    payload = json.dumps(
        [
            {
                "batch": np.asarray(s["batch"]).tolist(),
                "library": [float(v) for v in s["library"]],
                "panel": np.asarray(s["panel"]).tolist(),
                "weight": [float(v) for v in s["weight"]],
            }
            for s in scenarios
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert digest == hashlib.sha256(payload.encode()).hexdigest()


def test_marginal_scenarios_do_not_copy_per_cell_arrays():
    """S contexts over n cells hold O(n + S) memory, not n x S copies."""
    n_query, n_ref = 20_000, 64
    scenarios, _ = cf._build_context_scenarios(**_marginal_request(n_query, n_ref))
    assert len(scenarios) == n_ref
    owned = sum(
        values.nbytes
        for scenario in scenarios
        for values in scenario.values()
        if isinstance(values, np.ndarray) and values.base is None
    )
    # Before the fix every scenario owned four n-cell arrays: n * S * 24 bytes.
    assert owned < n_query * 24


# Counterfactual index validation ----------------------------------------------


@pytest.mark.parametrize("bad", [[1.9], [-0.5], ["1"], [np.nan]])
def test_counterfactual_indices_reject_non_integers(bad):
    adata = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="integer observation indices"):
        cf._as_indices(adata, bad, name="indices")


def test_counterfactual_indices_keep_whole_number_floats():
    adata = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    np.testing.assert_array_equal(cf._as_indices(adata, [1.0, 3.0], name="indices"), [1, 3])
    np.testing.assert_array_equal(cf._as_indices(adata, [2, 0], name="indices"), [2, 0])


# Protein axis order on transfer ------------------------------------------------


def _numpy_protein_model():
    """A model whose protein matrix is numpy, with names in adata.uns."""
    import scvi

    adata = scvi.data.synthetic_iid(n_proteins=4)
    adata.obsm["protein_expression"] = np.asarray(adata.obsm["protein_expression"])
    adata.uns["prot_names"] = ["CD3", "CD4", "CD8", "CD19"]
    adata.obs["sample"] = [f"s{i % 3}" for i in range(adata.n_obs)]
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="prot_names",
        sample_key="sample",
        batch_key="batch",
    )
    return MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2), adata


def test_reordered_protein_names_are_rejected_on_transfer():
    model, adata = _numpy_protein_model()
    other = adata.copy()
    other.obsm["protein_expression"] = adata.obsm["protein_expression"][:, ::-1].copy()
    other.uns["prot_names"] = ["CD19", "CD8", "CD4", "CD3"]
    with pytest.raises(ValueError, match="registered protein"):
        model._validate_anndata(other)


def test_same_protein_order_is_accepted_on_transfer():
    model, adata = _numpy_protein_model()
    model._validate_anndata(adata.copy())


def test_reordered_protein_names_are_rejected_on_load(tmp_path):
    model, adata = _numpy_protein_model()
    model.is_trained_ = True
    model.save(tmp_path / "model", overwrite=True)
    other = adata.copy()
    other.obsm["protein_expression"] = adata.obsm["protein_expression"][:, ::-1].copy()
    other.uns["prot_names"] = ["CD19", "CD8", "CD4", "CD3"]
    with pytest.raises(ValueError, match="registered protein"):
        MrTotalVI.load(tmp_path / "model", adata=other)
    MrTotalVI.load(tmp_path / "model", adata=adata.copy())


# Inherited TOTALVI methods ----------------------------------------------------


@pytest.mark.parametrize("transform_batch", [["batch_0", "batch_1"], ["batch_0", "batch_1", "batch_0"]])
def test_normalized_expression_library_size_scales_once(transform_batch):
    """Scale rows sum to 1, so a batch mean scaled by L sums to L per cell."""
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, _ = _make_v2_model()
    torch.manual_seed(0)
    unit, _ = model.get_normalized_expression(transform_batch=transform_batch, library_size=1, return_numpy=True)
    torch.manual_seed(0)
    scaled, _ = model.get_normalized_expression(transform_batch=transform_batch, library_size=1000.0, return_numpy=True)
    np.testing.assert_allclose(scaled.sum(1), 1000.0, rtol=1e-4)
    np.testing.assert_allclose(scaled, 1000.0 * unit, rtol=1e-4)


def test_feature_correlation_uses_the_supplied_adata():
    """Correlations come from the passed adata, checked against np.corrcoef by hand."""
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    other = adata[:6].copy()
    n_features = model.summary_stats.n_vars + model.summary_stats.n_proteins

    def fake_denoised(adata=None, indices=None, n_samples=1, **kwargs):
        source = model.adata if adata is None else adata
        values = np.random.default_rng(source.n_obs).gamma(2.0, size=(source.n_obs, n_features))
        rows = values if indices is None else values[np.asarray(indices)]
        return rows[:, :, None]

    model._get_denoised_samples = fake_denoised
    got = model.get_feature_correlation_matrix(
        adata=other, indices=np.arange(1, 6), n_samples=1, correlation_type="pearson"
    )
    values = np.random.default_rng(other.n_obs).gamma(2.0, size=(other.n_obs, n_features))
    np.testing.assert_allclose(got.values, np.corrcoef(values[1:6], rowvar=False), atol=1e-12)


# One-cell samples in the outlier pairs ----------------------------------------

# log N(x; mu, 0.5) in one dimension = -log(0.5) - log(2 pi) / 2 - d^2 / (2 * 0.25).
_LOG_NORMAL_D0 = -math.log(0.5) - 0.5 * math.log(2 * math.pi)


def _one_cell_sample_data(u, labels):
    obs = pd.DataFrame({"sample": labels}, index=[f"c{i}" for i in range(len(labels))])
    return ad.AnnData(np.zeros((len(labels), 1), dtype=np.float32), obs=obs)


def test_one_cell_sample_has_no_own_support():
    u = np.array([[0.0], [1.0], [2.0], [5.0], [5.5], [7.0]])
    labels = np.array(["A", "B", "B", "C", "C", "C"])
    out = get_outlier_cell_sample_pairs(
        _OutlierStub(u, labels),
        adata=_one_cell_sample_data(u, labels),
        sample_key="sample",
        subsample_size=None,
        quantile_threshold=0.05,
        batch_size=8,
    )
    # c0 is the only cell of A: no independent component, so NaN and inadmissible.
    assert np.isnan(out["log_probs"].sel(sample="A", cell_name="c0"))
    assert not out["is_admissible"].sel(sample="A", cell_name="c0")
    # Another cell still scores against A's single component (d = 1 for c1).
    np.testing.assert_allclose(out["log_probs"].sel(sample="A", cell_name="c1"), _LOG_NORMAL_D0 - 2.0, atol=1e-5)
    # The pooled threshold holds only B's and C's leave-one-out scores, by hand:
    # B: d = 1, 1; C: d = 0.5, 0.5, 1.5 (c5 nearest c4).
    pool = _LOG_NORMAL_D0 - 2.0 * np.array([1.0, 1.0, 0.25, 0.25, 2.25])
    expected = np.quantile(pool, 0.05)
    got = float(out["log_probs"].sel(sample="A", cell_name="c1") - out["log_ratios"].sel(sample="A", cell_name="c1"))
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_outlier_pairs_need_one_sample_with_two_cells():
    u = np.array([[0.0], [1.0], [2.0]])
    labels = np.array(["A", "B", "C"])
    with pytest.raises(ValueError, match="at least 2 cells"):
        get_outlier_cell_sample_pairs(
            _OutlierStub(u, labels),
            adata=_one_cell_sample_data(u, labels),
            sample_key="sample",
            subsample_size=None,
            batch_size=8,
        )


# Warnings for inherited DE and DA properties -----------------------------------


def test_knn_warns_for_groups_with_expected_count_below_one():
    """knn_k * n_g / N = 10 * 6 / 406 < 1 for the small group, so a warning names it."""
    from mrtotalvi._stats import differential_abundance
    from tests.test_da_density_estimator import StubModel

    rng = np.random.default_rng(0)
    sizes = {"s0": 200, "s1": 200, "s2": 3, "s3": 3}
    labels = np.concatenate([[name] * n for name, n in sizes.items()])
    u = rng.normal(size=(labels.size, 2))
    obs = pd.DataFrame(
        {"sample": labels, "size": np.where(np.isin(labels, ["s0", "s1"]), "big", "small")},
        index=[f"c{i}" for i in range(labels.size)],
    )
    adata = ad.AnnData(np.zeros((labels.size, 1), dtype=np.float32), obs=obs)
    with pytest.warns(UserWarning, match=r"size=small"):
        differential_abundance(
            StubModel(u, labels),
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["size"],
            density_estimator="knn",
            knn_k=10,
            omit_original_sample=False,
        )


@pytest.fixture(scope="module")
def trained_de_model():
    adata = _make_adata()
    adata.obs["condition"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b")
    adata.obs["site"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_2"]), "x", "y")
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    return model


def test_de_warns_only_for_correlated_design_columns(trained_de_model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained_de_model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["condition"], mc_samples=1, batch_size=32
        )
    assert not any("Design columns are correlated" in str(w.message) for w in caught)
    # condition and site are 0/1 columns that share donor_0, so they are not orthogonal.
    with pytest.warns(UserWarning, match="Design columns are correlated"):
        trained_de_model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["condition", "site"], mc_samples=1, batch_size=32
        )


def test_de_warns_when_rejected_samples_enter_standardisation(trained_de_model):
    with pytest.warns(UserWarning, match="Rejected \\(inadmissible\\) samples"):
        trained_de_model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["condition"],
            mc_samples=1,
            batch_size=32,
            filter_inadmissible_samples=True,
        )


# Stochastic residual scale floor ----------------------------------------------


@pytest.mark.parametrize("log_scale", [-60.0, -110.0])
def test_stochastic_residual_scale_has_a_floor(log_scale):
    """A very negative log-scale gives scale 1e-3 and a finite KL, not inf or a crash."""
    from torch.distributions import kl_divergence

    from mrtotalvi._components import EncoderUZ

    encoder = EncoderUZ(n_latent=3, n_sample=2, n_latent_u=2, use_map=False)
    raw = torch.cat([torch.zeros(4, 3), torch.full((4, 3), log_scale)], dim=-1)
    encoder.attention_block.forward = lambda query_embed, kv_embed: raw
    _, _, eps_dist = encoder(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
    np.testing.assert_allclose(eps_dist.scale.numpy(), 1e-3, rtol=1e-6)
    # Closed form for KL(N(0, s) || N(0, 1)) per dimension: -log s + s^2 / 2 - 1/2.
    kl = kl_divergence(eps_dist, Normal(0.0, 1.0))
    np.testing.assert_allclose(kl.numpy(), -math.log(1e-3) + 0.5e-6 - 0.5, rtol=1e-5)


# Seed combiner settings -------------------------------------------------------


def _seed_pair(**attrs):
    import xarray as xr

    base = {"specified_panel": "p0", "group_key": "condition", "contrast": '["A","B"]', "donor_key": "patient", **attrs}
    first = xr.Dataset({"value": (("draw",), [1.0, 3.0])}, attrs=dict(base))
    second = xr.Dataset({"value": (("draw",), [5.0, 7.0])}, attrs={**base, "random_state": 99})
    return first, second


@pytest.mark.parametrize(
    ("key", "changed"),
    [
        ("specified_panel", "p1"),
        ("group_key", "other_condition"),
        ("contrast", '["B","A"]'),
        ("donor_key", "other_patient"),
    ],
)
def test_seed_combiner_rejects_different_settings(key, changed):
    from mrtotalvi import combine_mrtotalvi_seed_results

    first, second = _seed_pair()
    result = combine_mrtotalvi_seed_results({0: first, 1: second})
    # Hand calculation: seed means 2 and 6, so mean 4 and sample variance 8.
    assert result.value_between_seed_mean.item() == 4.0
    assert result.value_between_seed_sd.item() == pytest.approx(8**0.5)
    second.attrs[key] = changed
    with pytest.raises(ValueError, match=key):
        combine_mrtotalvi_seed_results({0: first, 1: second})


def test_seed_combiner_warns_for_results_without_setting_attrs():
    import xarray as xr

    from mrtotalvi import combine_mrtotalvi_seed_results

    old = [
        xr.Dataset({"log_ratio": (("draw",), values)}, attrs={"random_state": seed})
        for seed, values in enumerate(([1.0, 3.0], [5.0, 7.0]))
    ]
    with pytest.warns(UserWarning, match="contrast"):
        result = combine_mrtotalvi_seed_results({0: old[0], 1: old[1]})
    assert result.log_ratio_between_seed_mean.item() == 4.0


def test_enrichment_records_contrast_settings_for_the_combiner():
    from mrtotalvi import combine_mrtotalvi_seed_results

    model, adata = _make_enrichment_v2_model()
    request = {
        "adata": adata,
        "indices": [0, 1],
        "group_key": "group",
        "donor_key": "donor",
        "inference_mode": "posterior_mc",
        "n_draws": 2,
    }
    forward = model.local_sample_enrichment(contrast=("numerator", "denominator"), **request)
    assert forward.attrs["group_key"] == "group"
    assert forward.attrs["contrast"] == '["numerator", "denominator"]'
    assert forward.attrs["donor_key"] == "donor"
    reverse = model.local_sample_enrichment(contrast=("denominator", "numerator"), random_state=1, **request)
    with pytest.raises(ValueError, match="contrast"):
        combine_mrtotalvi_seed_results({0: forward, 1: reverse})


# centered_v2 residual prior modes and protein annealing ------------------------


def _residual_cost_delta(mode):
    """Residual-prior cost of cell 0 for raw residuals [1, -0.5, -0.5] on one axis."""
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    module = model.module.eval()
    module.residual_prior_mode = mode
    tensors = next(iter(model._make_data_loader(adata=adata, indices=np.arange(2), batch_size=2)))
    assert tensors[REGISTRY_KEYS.SAMPLE_KEY].flatten()[0] == 0  # cell 0 is in sample 0
    inference, generative = module(tensors, compute_loss=False)
    raw = torch.zeros_like(inference["eps_raw_all"])  # (cell, sample, latent)
    raw[0, :, 0] = torch.tensor([1.0, -0.5, -0.5])

    def cost(values):
        centered = values - values.mean(1, keepdim=True)
        factual = module._gather_sample(centered, tensors[REGISTRY_KEYS.SAMPLE_KEY])
        outputs = dict(
            inference,
            eps_raw_all=values,
            eps_centered_all=centered,
            eps=factual,
            z=inference["z_base"] + factual,
        )
        return module.loss(tensors, outputs, generative).kl_local["kl_div_z"]

    # u and the likelihood are held fixed, so the difference is the residual penalty.
    return (cost(raw) - cost(torch.zeros_like(raw)))[0].item()


def test_residual_prior_modes_have_closed_form_costs():
    # sigma = exp(pz_scale) = 1. raw_mean: mean over samples of r^2 / 2 per sample,
    # (0.5 + 0.125 + 0.125) / 3 = 0.25. factual: the centered factual eps is [1, 0, 0],
    # cost 1^2 / 2 = 0.5, the same as a legacy private residual.
    assert _residual_cost_delta("raw_mean") == pytest.approx(0.25, abs=1e-6)
    assert _residual_cost_delta("factual") == pytest.approx(0.5, abs=1e-6)


def test_factual_residual_prior_needs_centered_v2():
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    with pytest.raises(ValueError, match="requires hierarchy_mode='centered_v2'"):
        MrTotalVI(_make_v2_adata(n_samples=3), sample_key="sample", residual_prior_mode="factual")


@pytest.mark.parametrize("path", ["legacy", "centered_v2", "mc_samples", "scale_observations", "no_hierarchy"])
@pytest.mark.parametrize("anneal", [True, False])
def test_protein_likelihood_slope_at_zero_kl_weight(path, anneal):
    """d loss / d pro_recons_weight at kl_weight=0 is mean protein NLL, or 0 when annealed.

    Each ``path`` reaches a different branch of ``loss`` that applies the protein weight.
    """
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    adata = _make_v2_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2" if path == "centered_v2" else "legacy",
        scale_observations=path == "scale_observations",
        anneal_protein_likelihood=anneal,
    )
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=8)))
    inference_kwargs = {"n_samples": 2} if path == "mc_samples" else {}
    inference, generative = module(tensors, inference_kwargs=inference_kwargs, compute_loss=False)
    if path == "no_hierarchy":
        inference = {k: v for k, v in inference.items() if k != "eps"}

    def loss(weight):
        return module.loss(tensors, inference, generative, kl_weight=0.0, pro_recons_weight=weight)

    reference = loss(0.0)
    slope = (loss(0.8).loss - reference.loss).item() / 0.8
    protein_nll = reference.reconstruction_loss["reconst_loss_protein"]
    if path == "scale_observations":
        # Legacy scale_observations divides each cell's loss by its sample's cell count.
        counts = module.n_obs_per_sample[tensors["sample"].long().flatten()]
        protein_nll = protein_nll / counts
    expected = 0.0 if anneal else protein_nll.mean().item()
    assert slope == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_new_model_flags_survive_save_and_load(tmp_path):
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    adata = _make_v2_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        residual_prior_mode="factual",
        anneal_protein_likelihood=False,
    )
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=adata)
    assert loaded.module.residual_prior_mode == "factual"
    assert loaded.module.anneal_protein_likelihood is False


# Fable review regressions ------------------------------------------------------


def test_backed_boolean_mask_selects_rows(tmp_path):
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)
    mask = np.zeros(backed.n_obs, dtype=bool)
    mask[[1, 4, 6]] = True
    got = model.get_latent_representation(adata=backed, indices=mask)
    expected = model.get_latent_representation(adata=backed, indices=[1, 4, 6])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_legacy_override_of_factual_checkpoint_saves_and_reloads(tmp_path):
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    adata = _make_v2_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        residual_prior_mode="factual",
    )
    model.is_trained_ = True
    model.save(tmp_path / "f", overwrite=True)
    with pytest.warns(UserWarning, match="resets it to 'raw_mean'"):
        legacy = MrTotalVI.load(
            tmp_path / "f",
            adata=adata,
            hierarchy_mode_override="legacy",
            allow_semantic_override=True,
        )
    assert legacy.module.residual_prior_mode == "raw_mean"
    legacy.save(tmp_path / "g", overwrite=True)
    assert MrTotalVI.load(tmp_path / "g", adata=adata).hierarchy_mode == "legacy"


def test_seed_combiner_accepts_an_old_and_a_new_result_with_a_warning():
    import xarray as xr

    from mrtotalvi import combine_mrtotalvi_seed_results

    old = xr.Dataset({"v": (("draw",), [1.0, 3.0])})
    new = xr.Dataset(
        {"v": (("draw",), [5.0, 7.0])},
        attrs={"group_key": "", "contrast": "[]", "donor_key": "", "specified_library_size": ""},
    )
    with pytest.warns(UserWarning, match="cannot be compared"):
        result = combine_mrtotalvi_seed_results({0: old, 1: new})
    assert result.v_between_seed_mean.item() == 4.0


def test_seed_combiner_rejects_different_technical_context():
    from mrtotalvi import combine_mrtotalvi_seed_results
    from tests.test_mrtotalvi_v2 import _make_context_v2_model

    model, adata = _make_context_v2_model()
    request = {
        "adata": adata,
        "indices": [0, 1],
        "target_samples": ["sample_0"],
        "gene_list": [str(adata.var_names[0])],
        "protein_list": [str(model.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names[0])],
        "library_policy": "specified",
        "inference_mode": "posterior_mc",
        "n_draws": 2,
    }
    small = model.get_counterfactual_expression(specified_library_size=1.0, **request)
    large = model.get_counterfactual_expression(specified_library_size=1000.0, random_state=1, **request)
    assert small.attrs["specified_library_size"] != large.attrs["specified_library_size"]
    with pytest.raises(ValueError, match="specified_library_size"):
        combine_mrtotalvi_seed_results({0: small, 1: large})


@pytest.mark.parametrize(
    ("offsets", "effect"),
    [
        ((0.0, 0.0, 0.0, 0.0), 1.0),
        ((3.0, 0.0, 0.0, 0.0), 0.0),
        ((0.0, 3.0, 0.0, 0.0), 0.0),
        ((3.0, -1.0, 2.0, 0.0), 1.0),
    ],
)
def test_de_donor_offsets_do_not_leak_into_beta(trained_paired_model, offsets, effect):
    """eps = patient offset + effect * time, so the time beta must equal the planted effect."""
    model = trained_paired_model
    order = list(model.sample_order)
    level = {f"p{i}": value for i, value in enumerate(offsets)}
    table = torch.tensor([[level[s.split("_")[0]] + effect * s.endswith("t1")] for s in order], dtype=torch.float32)
    scale = torch.arange(1.0, model.module.n_latent + 1)

    def residual(u, sample, deterministic=False):
        eps = table[sample.flatten().long()] * scale
        return torch.zeros_like(eps), eps, None

    original = model.module.qz.forward
    model.module.qz.forward = residual
    try:
        de = model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["time"],
            donor_key="patient",
            mc_samples=1,
            batch_size=200,
            indices=np.arange(50),
        )
    finally:
        model.module.qz.forward = original
    expected = np.broadcast_to(effect * scale.numpy(), de["beta"].values[:, 0, :].shape)
    np.testing.assert_allclose(de["beta"].values[:, 0, :], expected, atol=1e-4)


@pytest.fixture(scope="module")
def trained_paired_model():
    import scvi

    scvi.settings.seed = 0
    adata = scvi.data.synthetic_iid()
    patient = np.arange(adata.n_obs) % 4
    time = (np.arange(adata.n_obs) // 4) % 2
    adata.obs["patient"] = [f"p{i}" for i in patient]
    adata.obs["time"] = [f"t{i}" for i in time]
    adata.obs["sample"] = [f"p{i}_t{j}" for i, j in zip(patient, time, strict=True)]
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=5)
    model.train(max_epochs=2, accelerator="cpu")
    return model


# Continued training mode ---------------------------------------------------------


def test_continued_training_runs_in_train_mode(tmp_path):
    """The first, a second and a post-load train() all compute the loss in train mode."""
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    adata = _make_v2_adata(n_samples=3)
    model = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2)

    def modes_during(fit_model):
        seen = []
        original = type(fit_model.module).loss

        def recording_loss(self, *args, **kwargs):
            seen.append(self.training)
            return original(self, *args, **kwargs)

        fit_model.module.loss = recording_loss.__get__(fit_model.module)
        try:
            fit_model.train(max_epochs=1, accelerator="cpu", train_size=1.0, batch_size=8, early_stopping=False)
        finally:
            del fit_model.module.loss
        return set(seen)

    assert True in modes_during(model)
    assert modes_during(model) == {True}  # second call
    model.save(tmp_path / "m", overwrite=True)
    assert modes_during(MrTotalVI.load(tmp_path / "m", adata=adata)) == {True}


# Protein availability when the panel is the batch -----------------------------


def _batch_only_context_model():
    """Protein 0 is never measured in batch_0, and no panel is registered."""
    import scvi

    adata = scvi.data.synthetic_iid(batch_size=6, n_genes=5, n_proteins=3, n_batches=2)
    adata.obs["sample"] = np.asarray(["sample_0"] * 6 + ["sample_1"] * 6)
    adata.obs["batch"] = np.asarray(["batch_0", "batch_1"] * 6)
    protein = np.asarray(adata.obsm["protein_expression"], dtype=np.float32) + 1.0
    protein[adata.obs["batch"].to_numpy() == "batch_0", 0] = 0.0
    adata.obsm["protein_expression"] = protein
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = MrTotalVI(
            adata,
            sample_key="sample",
            n_latent=3,
            n_latent_u=2,
            hierarchy_mode="centered_v2",
            u_encoder_mode="sample_blind",
        )
    model.is_trained_ = True
    return model, adata


def test_specified_batch_also_sets_the_protein_mask_without_a_panel():
    model, adata = _batch_only_context_model()
    proteins = [str(p) for p in model.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names]
    batch_1_cells = np.flatnonzero(adata.obs["batch"].to_numpy() == "batch_1")[:2]
    out = model.get_counterfactual_expression(
        adata=adata,
        indices=batch_1_cells,
        gene_list=[str(adata.var_names[0])],
        protein_list=proteins[:1],
        batch_policy="specified",
        specified_batch="batch_0",
    )
    # Decoded in batch_0, where protein 0 was never measured: unavailable, not a value
    # from decoder weights the masked likelihood never trained.
    assert not out["protein_available"].values.any()
    with pytest.raises(ValueError, match="the panel is the batch"):
        model.get_counterfactual_expression(
            adata=adata,
            indices=batch_1_cells,
            gene_list=[str(adata.var_names[0])],
            protein_list=proteins[:1],
            batch_policy="specified",
            specified_batch="batch_0",
            panel_policy="specified",
            specified_panel="batch_1",
        )


# Memory guard with many registered samples -------------------------------------


def test_memory_guard_counts_every_registered_sample():
    """One target, but centering builds residuals for all 300 registered samples."""
    import scvi

    adata = scvi.data.synthetic_iid(batch_size=300, n_genes=20, n_proteins=3, n_batches=2)
    adata.obs["sample"] = [f"s{i % 300}" for i in range(adata.n_obs)]
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = MrTotalVI(
            adata,
            sample_key="sample",
            n_latent=30,
            n_latent_u=10,
            hierarchy_mode="centered_v2",
            u_encoder_mode="sample_blind",
        )
    model.is_trained_ = True
    # The output alone is small: one target, 256 cells, 32 draws (measured peak ~10 GB).
    output_only = cf._estimate_latent_bytes(
        n_cells=256,
        n_targets=1,
        n_draws=32,
        n_latent_u=10,
        n_latent=30,
        n_quantiles=3,
        posterior_mc=True,
    )
    assert output_only < cf.MAX_IN_MEMORY_BYTES
    with pytest.raises(MemoryError, match="512"):
        model.get_counterfactual_latent(
            indices=np.arange(256),
            target_samples=["s0"],
            inference_mode="posterior_mc",
            n_draws=32,
        )


# Overrides: backed order and log transform -----------------------------------------


def test_normalized_expression_keeps_request_order_on_backed_data(tmp_path):
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)
    request = [7, 2, 5, 0]
    # The protein foreground mixture is deterministic per cell, so compare by cell.
    torch.manual_seed(0)
    reference_rna, reference_protein = model.get_normalized_expression(adata=backed, indices=[0, 2, 5, 7], n_samples=1)
    torch.manual_seed(0)
    rna, protein = model.get_normalized_expression(adata=backed, indices=request, n_samples=1)
    assert list(rna.index) == list(backed.obs_names[request])
    np.testing.assert_allclose(protein.values, reference_protein.loc[backed.obs_names[request]].values, rtol=1e-5)


def test_feature_correlation_log_transform_runs():
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    corr = model.get_feature_correlation_matrix(adata=adata, n_samples=2, log_transform=True)
    n_features = model.summary_stats.n_vars + model.summary_stats.n_proteins
    assert corr.shape == (n_features, n_features)


def test_learnable_prior_scale_with_map_warns():
    from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

    with pytest.warns(UserWarning, match="removes most of the donor residual"):
        MrTotalVI(_make_v2_adata(n_samples=3), sample_key="sample", learn_z_u_prior_scale=True)


# Step 9 nits --------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, -1, 2**32, 1.5, "1"])
def test_counterfactual_random_state_must_be_a_32_bit_integer(bad):
    with pytest.raises(ValueError, match="random_state"):
        cf._validate_random_state(bad)


def test_counterfactual_random_state_accepts_numpy_integers():
    cf._validate_random_state(np.int64(2**32 - 1))
    cf._validate_random_state(0)


def test_counterfactual_indices_reject_an_empty_selection():
    adata = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="at least one observation"):
        cf._as_indices(adata, np.zeros(4, dtype=bool), name="indices")


def test_missing_protein_names_warn_on_transfer():
    model, adata = _numpy_protein_model()
    other = adata.copy()
    del other.uns["prot_names"]
    with pytest.warns(UserWarning, match="protein axis order cannot be checked"):
        model._validate_anndata(other)


def test_local_sample_distances_reject_missing_group_labels():
    model, adata = _numpy_protein_model()
    adata.obs["group"] = pd.Series(["a", None] * (adata.n_obs // 2), index=adata.obs_names)
    model.is_trained_ = True  # the label check runs before any inference
    with pytest.raises(ValueError, match="missing labels"):
        model.get_local_sample_distances(adata=adata, groupby="group")


def test_seed_combiner_warns_for_entries_nan_in_some_seeds():
    import xarray as xr

    from mrtotalvi import combine_mrtotalvi_seed_results

    first = xr.Dataset({"value": (("draw", "cell"), [[1.0, np.nan], [3.0, np.nan]])})
    second = xr.Dataset({"value": (("draw", "cell"), [[5.0, 2.0], [7.0, 4.0]])})
    with pytest.warns(UserWarning, match=r"value \(1\)"):
        result = combine_mrtotalvi_seed_results({0: first, 1: second})
    # Cell 0 has both seeds (means 2 and 6); cell 1 only the second seed (mean 3).
    np.testing.assert_allclose(result.value_between_seed_mean.values, [4.0, 3.0])


def test_registry_adapter_keeps_pandas_category_order():
    from mrtotalvi._datamodule import _MrTotalVIRegistryAdapter
    from tests.test_datamodule import _make_adata as _make_dm_adata

    adata = _make_dm_adata()
    order = sorted(adata.obs["sample"].unique(), reverse=True) + ["unused"]
    adata.obs["sample"] = pd.Categorical(adata.obs["sample"], categories=order)
    dm = _MrTotalVIRegistryAdapter(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        parallel=False,
    )
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein_expression", sample_key="sample")
    registered = (
        MrTotalVI(adata, sample_key="sample")
        .adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY)
        .categorical_mapping
    )
    np.testing.assert_array_equal(registered, order)
    adapter_mapping = dm.registry["field_registries"][REGISTRY_KEYS.SAMPLE_KEY]["state_registry"]["categorical_mapping"]
    np.testing.assert_array_equal(adapter_mapping, order)


# Step 10 mutants ----------------------------------------------------------------


def test_expression_records_the_built_context_hash(monkeypatch):
    """The attr is the digest of the scenarios this request built, not a constant."""
    from tests.test_mrtotalvi_v2 import _make_v2_model

    model, adata = _make_v2_model()
    built = []
    original = cf._build_context_scenarios

    def spy(**kwargs):
        scenarios, digest = original(**kwargs)
        built.append(digest)
        return scenarios, digest

    monkeypatch.setattr(cf, "_build_context_scenarios", spy)
    hashes = []
    for library_size in (10.0, 20.0):
        result = model.get_counterfactual_expression(
            adata=adata,
            indices=[0, 2],
            target_samples=["sample_0"],
            gene_list=[str(adata.var_names[0])],
            inference_mode="latent_mean",
            n_draws=1,
            library_policy="specified",
            specified_library_size=library_size,
        )
        hashes.append(result.attrs["context_table_sha256"])
    assert hashes == built
    assert hashes[0] != hashes[1]


def test_admissibility_rows_align_by_name_not_position():
    import xarray as xr

    model, adata = _numpy_protein_model()
    samples = list(model.sample_order)
    cells = list(adata.obs_names[:2])
    # Expected rows in request order (cells) and registry order (samples).
    expected = np.array([[True, False, True], [False, True, True]])
    frame = pd.DataFrame(expected, index=cells, columns=samples)
    # Reverse both axes, add a cell the request does not use, and swap the dims.
    frame = frame.iloc[::-1, ::-1]
    frame.loc[str(adata.obs_names[5])] = False
    mask = xr.DataArray(
        frame.to_numpy().T,
        dims=("sample", "cell_name"),
        coords={"sample": list(frame.columns), "cell_name": list(frame.index)},
    )
    np.testing.assert_array_equal(model._admissibility_rows(mask, cells), expected)


def test_knn_is_silent_when_every_expected_count_is_above_one():
    """knn_k * n_g / N = 70 * 6 / 406 = 1.03 for the small group: no warning."""
    from mrtotalvi._stats import differential_abundance
    from tests.test_da_density_estimator import StubModel

    rng = np.random.default_rng(0)
    sizes = {"s0": 200, "s1": 200, "s2": 3, "s3": 3}
    labels = np.concatenate([[name] * n for name, n in sizes.items()])
    obs = pd.DataFrame(
        {"sample": labels, "size": np.where(np.isin(labels, ["s0", "s1"]), "big", "small")},
        index=[f"c{i}" for i in range(labels.size)],
    )
    adata = ad.AnnData(np.zeros((labels.size, 1), dtype=np.float32), obs=obs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        differential_abundance(
            StubModel(rng.normal(size=(labels.size, 2)), labels),
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["size"],
            density_estimator="knn",
            knn_k=70,
            omit_original_sample=False,
        )
    assert not [w for w in caught if "knn group scores" in str(w.message)]


def test_de_is_silent_when_no_sample_is_rejected(trained_de_model, monkeypatch):
    import mrtotalvi._stats as stats

    original = stats.get_outlier_cell_sample_pairs

    def all_admissible(*args, **kwargs):
        out = original(*args, **kwargs)
        return out.assign(is_admissible=out["is_admissible"] | True)

    monkeypatch.setattr(stats, "get_outlier_cell_sample_pairs", all_admissible)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained_de_model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["condition"],
            mc_samples=1,
            batch_size=32,
            filter_inadmissible_samples=True,
        )
    assert not [w for w in caught if "Rejected (inadmissible)" in str(w.message)]
