"""Fixes from the 2026-11-24 ambient_isotype review (AMBIENT_ISOTYPE_REVIEW.md, engineering findings 1-5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq


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
    return adata, profile


def _model(sim, adata=None, **kwargs):
    base, profile = sim
    adata = (base if adata is None else adata).copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile, n_latent=4, **kwargs)


def _unmeasured(sim, columns):
    """Copy of the data with ``columns`` zero in well1, which totalVI registers as unmeasured there."""
    adata = sim[0].copy()
    protein = adata.obsm["protein"].copy()
    protein.loc[(adata.obs["batch"].astype(str) == "well1").to_numpy(), columns] = 0
    adata.obsm["protein"] = protein
    return adata


# Finding 1: isotype correlations depend on an unidentified gate -------------------------------


def test_correlation_masks_isotypes_in_ambient_isotype(sim):
    model = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    corr = model.get_feature_correlation_matrix(n_samples=2)
    assert corr.loc[ISOTYPES].isna().all().all()
    assert corr[ISOTYPES].isna().all().all()
    targets = [n for n in corr.index if n not in ISOTYPES]
    assert np.isfinite(corr.loc[targets, targets].to_numpy()).mean() > 0.9


def test_correlation_keeps_isotypes_in_ambient_mode(sim):
    model = _model(sim, protein_noise_model="ambient", isotype_names=ISOTYPES)
    corr = model.get_feature_correlation_matrix(n_samples=2)
    assert corr.loc[ISOTYPES, ISOTYPES].notna().all().all()


# Finding 2: the foreground getter runs in evaluation mode --------------------------------------


def test_foreground_probability_runs_in_eval_mode(sim):
    model = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    seen = []
    # Upstream calls inference/generative directly, so watch the dropout and BatchNorm layers.
    layers = [m for m in model.module.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d))]
    handles = [m.register_forward_pre_hook(lambda module, args: seen.append(module.training)) for m in layers]
    model.module.train()
    try:
        model.get_protein_foreground_probability(n_samples=1)
    finally:
        for handle in handles:
            handle.remove()
    assert seen and not any(seen)
    assert model.module.training  # the caller's mode is restored


# Finding 3: settings with no effect warn -------------------------------------------------------


def test_seeded_background_prior_warns_in_ambient_isotype(sim):
    with pytest.warns(UserWarning, match="seed_background_prior_from_ambient has no effect"):
        _model(
            sim,
            protein_noise_model="ambient_isotype",
            isotype_names=ISOTYPES,
            seed_background_prior_from_ambient=True,
        )


def test_panel_without_controls_warns(sim):
    adata = _unmeasured(sim, ISOTYPES)
    with pytest.warns(UserWarning, match="measure no isotype control"):
        _model(sim, adata=adata, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)


# Finding 4: constructor values that break the objective ----------------------------------------


@pytest.mark.parametrize("reg", [-1.0, float("nan"), float("inf")])
def test_background_mixing_reg_must_be_finite_nonnegative(sim, reg):
    with pytest.raises(ValueError, match="background_mixing_reg"):
        _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES, background_mixing_reg=reg)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_n_nuisance_must_be_positive_whole(sim, n):
    with pytest.raises(ValueError, match="n_nuisance"):
        _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES, n_nuisance=n)


def test_background_mixing_reg_zero_is_accepted(sim):
    _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES, background_mixing_reg=0.0)


# Finding 2 (loss): unmeasured targets carry no mixing penalty ----------------------------------


def _penalty(model, adata, mixing_shift):
    """kl_div_back_pro per cell after shifting the decoded mixing logit of every protein."""
    module = model.module
    module.eval()
    loader = model._make_data_loader(adata=adata, batch_size=adata.n_obs, shuffle=False)
    tensors = next(iter(loader))
    torch.manual_seed(0)
    with torch.no_grad():
        inference = module.inference(**module._get_inference_input(tensors))
        generative = module.generative(**module._get_generative_input(tensors, inference))
        generative["py_"]["mixing"] = generative["py_"]["mixing"] + mixing_shift
        return module.loss(tensors, inference, generative).kl_local["kl_div_back_pro"], tensors


def test_unmeasured_target_has_no_mixing_penalty(sim):
    names = [str(n) for n in sim[0].obsm["protein"].columns]
    target = names[0]
    adata = _unmeasured(sim, [target])
    model = _model(sim, adata=adata, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    adata = model.adata
    shift = torch.zeros(len(names))
    shift[names.index(target)] = 5.0
    before, tensors = _penalty(model, adata, 0.0)
    after, _ = _penalty(model, adata, shift)
    in_batch1 = (tensors[REGISTRY_KEYS.BATCH_KEY].reshape(-1) == 1).numpy()
    # well1 (code 1) never measured the target: its penalty ignores that gate. Batch 0 did measure it.
    assert torch.allclose(before[in_batch1], after[in_batch1], atol=1e-5)
    assert not torch.allclose(before[~in_batch1], after[~in_batch1], atol=1e-5)
