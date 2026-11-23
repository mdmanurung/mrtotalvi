"""ADR-0011: ambient calibration and the opt-in protein noise models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq


@pytest.fixture(scope="module")
def sim():
    adata, truth, empties = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=800)
    profile = estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    return adata, truth, empties, profile


def _model(sim, **kwargs):
    adata, _, _, profile = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile, n_latent=4, **kwargs), adata


def test_profile_recovers_planted_composition(sim):
    _, truth, _, profile = sim
    planted = truth["composition"].values
    # Independent expectation: the simulator's planted Dirichlet composition.
    assert np.abs(profile.profile - planted).sum() < 0.05
    assert profile.profile.sum() == pytest.approx(1.0)
    assert profile.settings["prob"] == 0.995


def test_adt_only_selection_fails_closed(sim):
    _, _, empties, _ = sim
    raw = empties["raw_counts"].tocsc()
    adt_cols = [empties["feature_names"].index(n) for n in empties["adt_features"]]
    # scAR's cutoff scales with feature count, so a 22-feature matrix selects nothing.
    with pytest.raises(ValueError, match="whole transcriptome"):
        estimate_adt_ambient(raw[:, adt_cols], empties["raw_barcodes"], [], empties["adt_features"])


def test_profile_rejects_mismatched_feature_names(sim):
    _, _, _, profile = sim
    with pytest.raises(ValueError, match="do not match"):
        profile.aligned(list(profile.feature_names[:-1]) + ["not_a_protein"])
    reordered = profile.aligned(profile.feature_names[::-1])
    np.testing.assert_allclose(reordered.profile, profile.profile[::-1])


def test_default_model_has_no_new_parameters(sim):
    model, _ = _model(sim)
    names = dict(model.module.named_parameters())
    assert not any(n.startswith(("nuisance_", "nuisance_encoder")) for n in names)
    assert model.module.protein_noise_model == "totalvi"


def test_isotype_mode_fails_closed_without_isotypes(sim):
    with pytest.raises(ValueError, match="at least one isotype"):
        _model(sim, protein_noise_model="ambient_isotype")


def test_unknown_isotype_name_fails(sim):
    with pytest.raises(ValueError, match="not on the protein axis"):
        _model(sim, protein_noise_model="ambient_isotype", isotype_names=["IgG_missing"])


def test_zscore_blocked_with_vamp(sim):
    with pytest.raises(ValueError, match="blocked with u_prior='vamp'"):
        _model(sim, protein_encoder_input="ambient_zscore", u_prior="vamp")


def test_seeded_prior_matches_empty_droplet_stats(sim):
    model, _ = _model(sim, seed_background_prior_from_ambient=True)
    _, _, _, profile = sim
    alpha = model.module.background_pro_alpha.detach().numpy()
    np.testing.assert_allclose(alpha[:, 0], profile.log1p_mean, rtol=1e-5)
    np.testing.assert_allclose(alpha[:, 1], profile.log1p_mean, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protein_noise_model": "ambient"},
        {"protein_noise_model": "ambient_isotype", "isotype_names": ISOTYPES},
        {
            "protein_noise_model": "ambient_isotype",
            "isotype_names": ISOTYPES,
            "protein_encoder_input": "ambient_zscore",
        },
        {"protein_background_source": "z_base"},
    ],
)
def test_opt_in_modes_train_and_decompose(sim, kwargs):
    model, adata = _model(sim, **kwargs)
    model.train(max_epochs=2, accelerator="cpu", batch_size=128)
    comp = model.get_protein_noise_decomposition(n_samples=3)
    for key in ("native", "ambient", "nonspecific", "observed"):
        assert np.isfinite(comp[key].values).all()
        assert (comp[key].values >= 0).all()
    # The free-ambient component exists only in the ambient modes.
    if kwargs.get("protein_noise_model", "totalvi") == "totalvi":
        assert (comp["ambient"].values == 0).all()
    else:
        assert (comp["ambient"].values > 0).all()
    if "isotype_names" in kwargs:
        assert not set(ISOTYPES) & set(comp["native"].columns)


def test_isotype_columns_have_no_native_term(sim):
    model, adata = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=32)))
    out = model.module.protein_components(tensors)
    iso = model.module.isotype_index
    assert torch.all(out["native"][:, iso] == 0)


def test_counterfactual_decoding_fails_closed_in_opt_in_modes(sim):
    model, _ = _model(sim, protein_noise_model="ambient")
    with pytest.raises(NotImplementedError, match="ADR-0011"):
        model.module._deterministic_decoder_parameters(
            torch.zeros(2, 4), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)
        )


def test_zbase_background_ignores_sample_switch(sim):
    model, adata = _model(sim, protein_background_source="z_base")
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=16)))
    module = model.module
    module.eval()
    with torch.no_grad():
        inf = module.inference(**module._get_inference_input(tensors))
        gen_in = module._get_generative_input(tensors, inf)
        a = module.generative(**gen_in)["py_"]["back_alpha"]
        # A different z with the same z_base: the background must not move.
        gen_in["z"] = gen_in["z"] + 1.0
        b = module.generative(**gen_in)["py_"]["back_alpha"]
    torch.testing.assert_close(a, b)


def test_per_batch_profiles_stack_in_registry_order():
    adata, _, empties = simulate_citeseq(seed=1, cells_per_donor=40, n_empty=800, batch_specific_ambient=True)
    by = empties["droplet_batch"]
    profiles = {
        b: estimate_adt_ambient(
            empties["raw_counts"][by == b],
            np.asarray(empties["raw_barcodes"])[by == b],
            adata.obs_names,
            empties["feature_names"],
            adt_features=empties["adt_features"],
        )
        for b in np.unique(by)
    }
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    model = MrTotalVI(adata, sample_key="donor", ambient_profile=profiles, protein_noise_model="ambient", n_latent=4)
    table = model.module.ambient_profile.numpy()
    np.testing.assert_allclose(table[0], profiles["well0"].profile, rtol=1e-5)
    np.testing.assert_allclose(table[1], profiles["well1"].profile, rtol=1e-5)
    with pytest.raises(ValueError, match="No ambient profile"):
        MrTotalVI(
            adata, sample_key="donor", ambient_profile={"well0": profiles["well0"]}, protein_noise_model="ambient"
        )


@pytest.mark.parametrize("batch_size", [2, 256])  # 2 crosses minibatch boundaries
def test_decomposition_follows_request_order_on_backed_adata(sim, tmp_path, batch_size):
    """scvi sorts backed minibatches; the returned rows must follow the requested order."""
    import anndata as ad

    model, adata = _model(sim, protein_noise_model="ambient")
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)
    request = np.array([37, 2, 25, 0, 30, 1, 2])  # unsorted, with a duplicate
    # Both calls load the same unique rows in ascending order, so with the same seed
    # they draw the same posterior samples; only the final reordering differs.
    torch.manual_seed(0)
    ref = model.get_protein_noise_decomposition(backed, indices=np.unique(request), batch_size=batch_size)
    torch.manual_seed(0)
    out = model.get_protein_noise_decomposition(backed, indices=request, batch_size=batch_size)
    for key in ("native", "ambient", "nonspecific", "observed", "foreground_probability"):
        assert list(out[key].index) == list(backed.obs_names[request])
        np.testing.assert_allclose(out[key].values, ref[key].loc[backed.obs_names[request]].values, rtol=1e-6)


def test_per_batch_profiles_accept_numeric_batch_keys():
    adata, _, empties = simulate_citeseq(seed=1, cells_per_donor=40, n_empty=800, batch_specific_ambient=True)
    adata.obs["batch_num"] = adata.obs["batch"].map({"well0": 0, "well1": 1}).astype(int)
    by = empties["droplet_batch"]
    profiles = {
        int(b[-1]): estimate_adt_ambient(
            empties["raw_counts"][by == b],
            np.asarray(empties["raw_barcodes"])[by == b],
            adata.obs_names,
            empties["feature_names"],
            adt_features=empties["adt_features"],
        )
        for b in np.unique(by)
    }
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch_num")
    model = MrTotalVI(adata, sample_key="donor", ambient_profile=profiles, protein_noise_model="ambient", n_latent=4)
    np.testing.assert_allclose(model.module.ambient_profile.numpy()[1], profiles[1].profile, rtol=1e-5)
    with pytest.raises(ValueError, match="collide as strings"):
        MrTotalVI(
            adata,
            sample_key="donor",
            ambient_profile={0: profiles[0], 1: profiles[1], "1": profiles[1]},
            protein_noise_model="ambient",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protein_noise_model": "ambient"},
        {"protein_noise_model": "ambient_isotype", "isotype_names": ISOTYPES},
        {"protein_background_source": "z_base"},
    ],
)
def test_observed_equals_the_likelihood_mixture_mean(sim, kwargs):
    """observed = eff * (pi * mean_back + (1 - pi) * mean_fore), rebuilt from the model."""
    model, adata = _model(sim, **kwargs)
    module = model.module
    module.eval()
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=32)))
    torch.manual_seed(0)
    out = module.protein_components(tensors)
    torch.manual_seed(0)
    with torch.no_grad():
        inf = module.inference(**module._get_inference_input(tensors))
        gen = module.generative(**module._get_generative_input(tensors, inf))
    py_, eff = gen["py_"], gen["per_batch_efficiency"]
    pi = torch.sigmoid(py_["mixing"])
    amb = py_["ambient_mean"]
    if kwargs.get("protein_noise_model") == "ambient_isotype":
        back = amb + py_["nonspecific_mean"]
        fore = back + py_["native_mean"]
        # Free ambient is the burden times this batch's profile row.
        np.testing.assert_allclose(
            amb.numpy(), (torch.exp(inf["log_c"]) * module.ambient_profile[0]).numpy(), rtol=1e-5
        )
    else:
        b_mean = torch.exp(py_["back_alpha"] + 0.5 * py_["back_beta"].square())
        back = amb + b_mean
        fore = amb + b_mean * py_["fore_scale"]
    expected = eff * (pi * back + (1 - pi) * fore)
    np.testing.assert_allclose(out["observed"].numpy(), expected.numpy(), rtol=1e-4)


def test_counterfactual_u_encoding_uses_zscored_protein_input(sim):
    from mrtotalvi._counterfactual import _encode_u_params

    model, adata = _model(sim, protein_encoder_input="ambient_zscore")
    idx = np.arange(20)
    loc, _, _ = _encode_u_params(model, adata, idx, 32)
    module = model.module
    module.eval()
    tensors = next(iter(model._make_data_loader(adata=adata, indices=idx, batch_size=32)))
    with torch.no_grad():
        ref = module.inference(**module._get_inference_input(tensors))["qu"].loc
    np.testing.assert_allclose(np.asarray(loc), ref.numpy(), atol=1e-5)


def test_iterations_must_be_positive(sim):
    _, _, empties, _ = sim
    adata = sim[0]
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        estimate_adt_ambient(
            empties["raw_counts"],
            empties["raw_barcodes"],
            adata.obs_names,
            empties["feature_names"],
            adt_features=empties["adt_features"],
            iterations=0,
        )


def test_protein_absent_from_empties_warns_and_blocks_zscore(sim):
    adata, _, empties, _ = sim
    raw = empties["raw_counts"].tolil()
    col = empties["feature_names"].index(empties["adt_features"][0])
    is_cell = np.isin(empties["raw_barcodes"], adata.obs_names)
    raw[np.flatnonzero(~is_cell), col] = 0
    with pytest.warns(UserWarning, match="not estimable"):
        profile = estimate_adt_ambient(
            raw.tocsr(),
            empties["raw_barcodes"],
            adata.obs_names,
            empties["feature_names"],
            adt_features=empties["adt_features"],
        )
    data = adata.copy()
    MrTotalVI.setup_anndata(data, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    with pytest.raises(ValueError, match="positive empty-droplet log SD"):
        MrTotalVI(data, sample_key="donor", ambient_profile=profile, protein_encoder_input="ambient_zscore")


def test_decomposition_drops_isotypes_with_integer_protein_names(sim):
    # Integer protein column labels: isotype_names are matched as strings at setup,
    # so the decomposition must compare str names too, or isotypes leak through.
    import dataclasses

    adata, truth, empties, profile = sim
    adata = adata.copy()
    old = list(adata.obsm["protein"].columns)
    adata.obsm["protein"].columns = range(len(old))
    int_profile = dataclasses.replace(profile, feature_names=[old.index(str(n)) for n in profile.feature_names])
    iso = [str(old.index(n)) for n in ISOTYPES]
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    model = MrTotalVI(
        adata,
        sample_key="donor",
        ambient_profile=int_profile,
        n_latent=4,
        protein_noise_model="ambient_isotype",
        isotype_names=iso,
    )
    out = model.get_protein_noise_decomposition(indices=np.arange(8), n_samples=2)
    for frame in out.values():
        assert not set(map(str, frame.columns)) & set(iso)
        assert frame.shape[1] == len(old) - len(iso)
