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
    # The three count components sum to the observed mean.
    np.testing.assert_allclose(
        comp["observed"].values,
        comp["native"].values + comp["ambient"].values + comp["nonspecific"].values,
        rtol=1e-5,
    )
    assert np.isfinite(comp["observed"].values).all()
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
