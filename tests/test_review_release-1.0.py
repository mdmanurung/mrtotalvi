"""Reviewed fixes for the 1.0 release (crashes, ambient outputs, seeds, input checks)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scvi
import torch
import xarray as xr
from scvi import REGISTRY_KEYS

from mrtotalvi import MrTotalVI, combine_mrtotalvi_seed_results
from mrtotalvi._ambient import AmbientProfile
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq
from tests.test_mrtotalvi_v2 import _make_v2_model

# ----------------------------------------------------------------------------
# Shared builders
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    adata, _, empties = simulate_citeseq(seed=0, cells_per_donor=20, n_empty=400)
    profile = estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    return adata, profile, empties


def _model(sim, adata=None, setup_kwargs=None, **kwargs):
    base, profile, _ = sim
    adata = (base if adata is None else adata).copy()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein",
        sample_key="donor",
        batch_key="batch",
        **(setup_kwargs or {}),
    )
    if kwargs.get("protein_noise_model", "totalvi") != "totalvi":
        kwargs.setdefault("ambient_profile", profile)
    model = MrTotalVI(adata, sample_key="donor", n_latent=4, **kwargs)
    model.is_trained_ = True  # the getters under test read a fresh model; fits are not needed
    return model, adata


def _protein_names(model):
    return list(model.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)


# ----------------------------------------------------------------------------
# A1 - z_base background head under batch embeddings
# ----------------------------------------------------------------------------


def test_a1_zbase_background_head_with_batch_embedding_trains(sim):
    model, _ = _model(
        sim,
        protein_noise_model="ambient",
        protein_background_source="z_base",
        batch_representation="embedding",
    )
    model.train(max_epochs=1, accelerator="cpu", batch_size=64)
    native = model.get_protein_noise_decomposition(n_samples=2)["native"]
    assert np.isfinite(native.to_numpy()).all()


# ----------------------------------------------------------------------------
# A2, A3 - protein_list with the DataFrame return
# ----------------------------------------------------------------------------


def test_a2_foreground_probability_protein_list_frame(sim):
    model, adata = _model(sim)
    names = _protein_names(model)
    requested = [names[5], names[1]]  # request order differs from registry order
    scvi.settings.seed = 0
    full = model.get_protein_foreground_probability(n_samples=1)
    scvi.settings.seed = 0
    subset = model.get_protein_foreground_probability(protein_list=requested, n_samples=1)
    assert list(subset.columns) == [names[1], names[5]]
    assert list(subset.index) == list(adata.obs_names)
    np.testing.assert_allclose(subset.to_numpy(), full[[names[1], names[5]]].to_numpy(), rtol=1e-6)


def test_a2_foreground_probability_protein_list_rows_and_numpy(sim):
    model, adata = _model(sim)
    names = _protein_names(model)
    rows = [9, 2, 5]
    scvi.settings.seed = 0
    frame = model.get_protein_foreground_probability(indices=rows, protein_list=names[:3])
    scvi.settings.seed = 0
    array = model.get_protein_foreground_probability(indices=rows, protein_list=names[:3], return_numpy=True)
    assert list(frame.index) == list(adata.obs_names[rows])
    assert list(frame.columns) == names[:3]
    np.testing.assert_allclose(frame.to_numpy(), array, rtol=1e-6)
    stack = model.get_protein_foreground_probability(
        indices=rows, protein_list=names[:3], n_samples=2, return_mean=False
    )
    assert stack.shape == (3, 3, 2)


def test_a3_normalized_expression_protein_list_with_n_samples_overall(sim):
    model, adata = _model(sim)
    names = _protein_names(model)
    rna, protein = model.get_normalized_expression(protein_list=[names[3], names[0]], n_samples_overall=7)
    assert protein.shape == (7, 2)
    assert list(protein.columns) == [names[0], names[3]]
    assert rna.shape[0] == 7
    assert set(protein.index) <= set(adata.obs_names)


# ----------------------------------------------------------------------------
# B1 - ambient modes return the native component
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "isotypes"), [("ambient", None), ("ambient_isotype", ISOTYPES)])
def test_b1_ambient_normalized_protein_is_native(sim, mode, isotypes):
    model, _ = _model(sim, protein_noise_model=mode, isotype_names=isotypes)
    scvi.settings.seed = 0
    _, protein = model.get_normalized_expression(n_samples=3, batch_size=64)
    scvi.settings.seed = 0
    native = model.get_protein_noise_decomposition(n_samples=3, batch_size=64)["native"]
    np.testing.assert_allclose(protein[native.columns].to_numpy(), native.to_numpy(), rtol=1e-5)
    assert list(protein.columns) == _protein_names(model)
    if isotypes:
        assert protein[isotypes].isna().all().all()


def test_b1_native_follows_indices_protein_list_and_numpy(sim):
    model, _ = _model(sim, protein_noise_model="ambient")
    names = _protein_names(model)
    rows = [7, 3, 11]
    requested = [names[5], names[1]]
    scvi.settings.seed = 0
    _, array = model.get_normalized_expression(indices=rows, protein_list=requested, n_samples=2, return_numpy=True)
    scvi.settings.seed = 0
    native = model.get_protein_noise_decomposition(indices=rows, n_samples=2)["native"]
    np.testing.assert_allclose(array, native[[names[1], names[5]]].to_numpy(), rtol=1e-5)


def test_b1_isotype_columns_stay_finite_in_plain_ambient_mode(sim):
    # isotype_names alone do not blank columns; only ambient_isotype mode does.
    model, _ = _model(sim, protein_noise_model="ambient", isotype_names=ISOTYPES)
    _, protein = model.get_normalized_expression(n_samples=2)
    assert list(protein.columns) == _protein_names(model)
    assert np.isfinite(protein[ISOTYPES].to_numpy()).all()


def test_b1_include_protein_background_keeps_the_fitted_mean(sim):
    model, _ = _model(sim, protein_noise_model="ambient")
    scvi.settings.seed = 0
    _, native = model.get_normalized_expression(n_samples=3)
    scvi.settings.seed = 0
    _, full = model.get_normalized_expression(n_samples=3, include_protein_background=True)
    assert np.isfinite(full.to_numpy()).all()
    assert not np.allclose(full.to_numpy(), native.to_numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transform_batch": "well0"},
        {"scale_protein": True},
        {"sample_protein_mixing": True},
        {"n_samples": 2, "return_mean": False},
    ],
)
def test_b1_unsupported_options_raise_in_ambient_modes(sim, kwargs):
    model, _ = _model(sim, protein_noise_model="ambient")
    with pytest.raises(NotImplementedError, match="include_protein_background"):
        model.get_normalized_expression(**kwargs)


def test_b1_totalvi_mode_is_unchanged(sim):
    model, _ = _model(sim)
    rna, protein = model.get_normalized_expression(transform_batch="well0", scale_protein=True)
    assert np.isfinite(protein.to_numpy()).all()
    np.testing.assert_allclose(protein.to_numpy().sum(axis=1), 1.0, rtol=1e-4)


# ----------------------------------------------------------------------------
# B2 - proteins not measured in a panel are NaN
# ----------------------------------------------------------------------------


def _zero_in(adata, protein, cells):
    adata = adata.copy()
    adata.obsm["protein"].loc[cells, protein] = 0.0
    return adata


def _assert_nan_pattern(values, names, protein, missing_cells):
    column = names.index(protein)
    assert np.isnan(values[missing_cells, column]).all()
    assert np.isfinite(values[~missing_cells, column]).all()
    others = np.delete(values, column, axis=1)
    assert np.isfinite(others).all()


@pytest.mark.parametrize("mode", ["totalvi", "ambient"])
def test_b2_unmeasured_protein_is_nan_in_all_getters(sim, mode):
    base = sim[0]
    missing = (base.obs["batch"] == "well0").to_numpy()
    adata = _zero_in(base, "P00", base.obs_names[missing])
    model, adata = _model(sim, adata=adata, protein_noise_model=mode)
    assert model.module.protein_batch_mask is not None
    names = _protein_names(model)
    extra = {"include_protein_background": True} if mode != "totalvi" else {}
    _, protein = model.get_normalized_expression(**extra)
    _assert_nan_pattern(protein.to_numpy(), names, "P00", missing)
    fg = model.get_protein_foreground_probability()
    _assert_nan_pattern(fg.to_numpy(), names, "P00", missing)
    native = model.get_protein_noise_decomposition(n_samples=2)["native"]
    _assert_nan_pattern(native.to_numpy(), names, "P00", missing)
    # protein_list, indices and request order carry the mask with them
    rows = [30, 1, 22, 5]
    _, sub = model.get_normalized_expression(indices=rows, protein_list=["P02", "P00"], return_numpy=True, **extra)
    assert np.array_equal(np.isnan(sub[:, 0]), missing[rows])
    assert np.isfinite(sub[:, 1]).all()


def test_b2_panel_mask_is_keyed_by_panel_codes(sim):
    # Panel codes differ from batch codes: panel pB is donors 4-7 (both wells), and
    # P00 is zero in every pB cell. A lookup by batch code would mark the wrong cells.
    base = sim[0]
    base = base.copy()
    base.obs["panel"] = np.where(base.obs["condition"] == "stim", "pB", "pA")
    missing = (base.obs["panel"] == "pB").to_numpy()
    adata = _zero_in(base, "P00", base.obs_names[missing])
    model, adata = _model(sim, adata=adata, setup_kwargs={"panel_key": "panel"})
    assert model.module.panel_key == "panel"
    batch_code = np.asarray(model.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel()
    panel_code = np.asarray(model.adata_manager.get_from_registry("panel")).ravel()
    assert (batch_code != panel_code).any()
    assert (batch_code[missing] == 0).any()
    assert (batch_code[missing] == 1).any()
    names = _protein_names(model)
    _assert_nan_pattern(model.get_protein_noise_decomposition(n_samples=2)["native"].to_numpy(), names, "P00", missing)
    _assert_nan_pattern(model.get_normalized_expression()[1].to_numpy(), names, "P00", missing)
    _assert_nan_pattern(model.get_protein_foreground_probability().to_numpy(), names, "P00", missing)


# ----------------------------------------------------------------------------
# B3 - between-seed summaries
# ----------------------------------------------------------------------------


def _seed_results(draws, extra_vars=("log_density", "log_ratio", "group_log_density")):
    """``draws`` is (seeds, draws, cells); every seed gets the same variables."""
    results = {}
    for seed, values in enumerate(draws):
        variables = {"feat": (("draw", "cell"), values)}
        variables.update(dict.fromkeys(extra_vars, (("draw", "cell"), values)))
        results[seed] = xr.Dataset(
            variables,
            coords={"draw": np.arange(values.shape[0]), "cell": np.arange(values.shape[1])},
            attrs={"contrast": "a,b", "group_key": "g"},
        )
    return results


def test_b3_between_seed_sd_matches_closed_form_and_draw_correction():
    rng = np.random.default_rng(0)
    n_seeds, n_draws, n_cells = 5, 4, 50
    draws = rng.normal(0.0, 1.0, (n_seeds, n_draws, n_cells))
    combined = combine_mrtotalvi_seed_results(_seed_results(draws))
    var_between = draws.mean(axis=1).var(axis=0, ddof=1)
    var_within = draws.var(axis=1, ddof=1).mean(axis=0)
    np.testing.assert_allclose(combined["feat_between_seed_sd"], np.sqrt(var_between), rtol=1e-6)
    np.testing.assert_allclose(
        combined["feat_between_seed_sd_draw_corrected"],
        np.sqrt(np.maximum(var_between - var_within / n_draws, 0.0)),
        rtol=1e-5,
        atol=1e-7,
    )
    assert "draw noise" in combined.attrs["between_seed_sd_note"]


def test_b3_identical_truth_gives_half_sd_uncorrected_and_less_corrected():
    rng = np.random.default_rng(1)
    draws = rng.normal(0.0, 1.0, (6, 4, 4000))  # same true value (0), draw SD 1, D=4
    combined = combine_mrtotalvi_seed_results(_seed_results(draws, extra_vars=()))
    uncorrected = combined["feat_between_seed_sd"].to_numpy()
    corrected = combined["feat_between_seed_sd_draw_corrected"].to_numpy()
    assert abs(np.mean(uncorrected**2) - 0.25) < 0.02  # SD of a mean of 4 draws is 1/2
    assert np.mean(corrected**2) < 0.1


def test_b3_raw_density_variables_have_no_between_seed_summary():
    rng = np.random.default_rng(2)
    combined = combine_mrtotalvi_seed_results(_seed_results(rng.normal(size=(3, 4, 5))))
    for name in ("log_density", "group_log_density"):
        assert f"{name}_within_seed_posterior_mean" in combined
        assert f"{name}_within_seed_posterior_sd" in combined
        for suffix in ("between_seed_mean", "between_seed_sd", "between_seed_sd_draw_corrected"):
            assert f"{name}_{suffix}" not in combined
    assert combined.attrs["between_seed_skipped_density_variables"] == "log_density,group_log_density"
    for suffix in ("between_seed_mean", "between_seed_sd", "between_seed_sd_draw_corrected"):
        assert f"log_ratio_{suffix}" in combined
        assert f"feat_{suffix}" in combined


def test_b3_nan_draws_within_a_seed_raise():
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(3, 4, 5))
    draws[1, 2, 3] = np.nan
    with pytest.raises(ValueError, match=r"feat.*\b1\b"):
        combine_mrtotalvi_seed_results(_seed_results(draws, extra_vars=()))


def test_b3_entries_missing_in_every_draw_are_still_allowed():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(3, 4, 5))
    draws[:, :, 3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined = combine_mrtotalvi_seed_results(_seed_results(draws, extra_vars=()))
    assert np.isnan(combined["feat_between_seed_mean"].sel(cell=3))
    assert np.isfinite(combined["feat_between_seed_mean"].sel(cell=2))


def test_b3_numpy_integer_seed_keys():
    rng = np.random.default_rng(5)
    results = {np.int64(seed + 1): ds for seed, ds in _seed_results(rng.normal(size=(2, 4, 5))).items()}
    combined = combine_mrtotalvi_seed_results(results)
    assert combined.attrs["training_seed_order"] == "1,2"
    with pytest.raises(TypeError):
        combine_mrtotalvi_seed_results({True: results[np.int64(1)], 2: results[np.int64(2)]})


# ----------------------------------------------------------------------------
# C1 - estimate_adt_ambient input checks
# ----------------------------------------------------------------------------


def _estimate(empties, **overrides):
    args = {
        "raw_counts": empties["raw_counts"],
        "raw_barcodes": empties["raw_barcodes"],
        "cell_barcodes": empties["raw_barcodes"][:160],
        "feature_names": empties["feature_names"],
        "adt_features": empties["adt_features"],
    }
    args.update(overrides)
    return estimate_adt_ambient(**args)


def test_c1_cell_barcodes_absent_from_raw_raise(sim):
    empties = sim[2]
    cells = list(empties["raw_barcodes"][:160]) + ["ghost1", "ghost2"]
    with pytest.raises(ValueError, match=r"2 .*not in raw_barcodes"):
        _estimate(empties, cell_barcodes=cells)
    assert isinstance(_estimate(empties), AmbientProfile)


def test_c1_empty_cell_barcodes_raise(sim):
    with pytest.raises(ValueError, match="cell_barcodes is empty"):
        _estimate(sim[2], cell_barcodes=[])


def test_c1_duplicate_adt_features_raise(sim):
    features = list(sim[2]["adt_features"])
    with pytest.raises(ValueError, match="adt_features.*unique"):
        _estimate(sim[2], adt_features=features + features[:1])


# ----------------------------------------------------------------------------
# C2 - specified_* without the matching policy
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"specified_batch": "batch_0"}, "specified_batch.*batch_policy"),
        ({"specified_panel": "batch_0"}, "specified_panel.*panel_policy"),
        ({"specified_library_size": 100.0}, "specified_library_size.*library_policy"),
    ],
)
def test_c2_specified_value_needs_specified_policy(kwargs, match):
    model, adata = _make_v2_model()
    with pytest.raises(ValueError, match=match):
        model.get_counterfactual_expression(adata=adata, indices=[0, 1], **kwargs)


# ----------------------------------------------------------------------------
# C3 - reference subsample independent of random_state in latent_mean mode
# ----------------------------------------------------------------------------


def _enrichment_model():
    adata = scvi.data.synthetic_iid(batch_size=30, n_genes=5, n_proteins=3, n_regions=2, n_batches=2, n_labels=2)
    adata.obs["sample"] = np.asarray([f"sample_{index % 4}" for index in range(adata.n_obs)])
    MrTotalVI.setup_anndata(
        adata, protein_expression_obsm_key="protein_expression", sample_key="sample", batch_key="batch"
    )
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


def test_c3_latent_mean_reference_subsample_ignores_random_state():
    model, adata = _enrichment_model()
    runs = [
        model.local_sample_enrichment(
            adata=adata, indices=[0, 1, 2, 3], max_reference_cells_per_sample=3, random_state=state
        )
        for state in (0, 1, 2, 3, 4)
    ]
    assert (runs[0]["n_reference_cells"] <= 3).all()
    for other in runs[1:]:
        np.testing.assert_array_equal(other["log_density"].to_numpy(), runs[0]["log_density"].to_numpy())


# ----------------------------------------------------------------------------
# C4 - bare strings and numpy integers
# ----------------------------------------------------------------------------


def test_c4_isotype_names_string_raises(sim):
    with pytest.raises(TypeError, match="list"):
        _model(sim, protein_noise_model="ambient_isotype", isotype_names="IgG1_iso")


@pytest.mark.parametrize(
    "kwargs",
    [{"target_samples": "sample_0"}, {"gene_list": "gene_1"}, {"protein_list": "protein_1"}],
)
def test_c4_bare_string_feature_and_target_lists_raise(kwargs):
    model, adata = _make_v2_model()
    with pytest.raises(TypeError, match="list"):
        model.get_counterfactual_expression(adata=adata, indices=[0, 1], **kwargs)


def test_c4_bare_target_string_raises_in_enrichment():
    model, adata = _make_v2_model()
    with pytest.raises(TypeError, match="list"):
        model.local_sample_enrichment(adata=adata, indices=[0, 1], target_samples="sample_0")


def test_c4_numpy_integers_are_accepted_for_n_draws_and_batch_size():
    model, adata = _make_v2_model()
    by_int = model.local_sample_enrichment(adata=adata, indices=[0, 1], n_draws=1, batch_size=2)
    by_np = model.local_sample_enrichment(adata=adata, indices=[0, 1], n_draws=np.int64(1), batch_size=np.int32(2))
    np.testing.assert_array_equal(by_np["log_density"].to_numpy(), by_int["log_density"].to_numpy())
    latent = model.get_counterfactual_latent(adata=adata, indices=[0, 1], n_draws=np.int64(1), batch_size=np.int64(2))
    assert latent.sizes["cell_name"] == 2
    with pytest.raises(ValueError, match="n_draws"):
        model.local_sample_enrichment(adata=adata, indices=[0, 1], n_draws=True)


# ----------------------------------------------------------------------------
# C5 - compute_log_enrichment without covariates
# ----------------------------------------------------------------------------


def test_c5_log_enrichment_needs_sample_cov_keys():
    model, adata = _make_v2_model()
    assert "log_probs" in model.differential_abundance(adata=adata)
    with pytest.raises(ValueError, match="sample_cov_keys"):
        model.differential_abundance(adata=adata, compute_log_enrichment=True)


# ----------------------------------------------------------------------------
# C6 - residual collapse metrics need two cells
# ----------------------------------------------------------------------------


def _loss_metrics(model, adata, rows):
    loader = model._make_data_loader(adata=adata, indices=rows, batch_size=len(rows))
    tensors = next(iter(loader))
    model.module.eval()
    with torch.no_grad():
        _, _, loss = model.module(tensors)
    return loss.extra_metrics


def test_c6_single_cell_minibatch_logs_no_residual_metrics():
    model, adata = _make_v2_model()
    assert "residual_share" in _loss_metrics(model, adata, [0, 1, 2, 3])
    one = _loss_metrics(model, adata, [0])
    assert "residual_share" not in one
    assert "residual_active_units" not in one


# ----------------------------------------------------------------------------
# C7 - per-batch ambient_profile dict
# ----------------------------------------------------------------------------


def test_c7_profile_key_matching_no_batch_raises(sim):
    profile = sim[1]
    with pytest.raises(ValueError, match="wellX"):
        _model(
            sim, protein_noise_model="ambient", ambient_profile={"well0": profile, "well1": profile, "wellX": profile}
        )


def test_c7_profile_value_must_be_an_ambient_profile(sim):
    profile = sim[1]
    with pytest.raises(TypeError, match="AmbientProfile"):
        _model(sim, protein_noise_model="ambient", ambient_profile={"well0": profile, "well1": "not a profile"})
