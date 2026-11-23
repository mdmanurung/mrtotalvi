"""Review 2026-11-23, inference-state: eval mode for inference and ambient input validation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import simulate_citeseq


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


def _model(sim, profile=None, **kwargs):
    adata, base = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile or base, n_latent=4, **kwargs)


def _bn_counters(module):
    return {k: v.clone() for k, v in module.state_dict().items() if k.endswith("num_batches_tracked")}


def test_decomposition_refuses_untrained_model(sim):
    model = _model(sim)
    with pytest.raises(RuntimeError):
        model.get_protein_noise_decomposition(n_samples=2)


@pytest.mark.parametrize("n_samples", [0, -3, 2.5, True])
def test_decomposition_rejects_bad_n_samples(sim, n_samples):
    model = _model(sim)
    model.is_trained_ = True
    with pytest.raises(ValueError, match="n_samples"):
        model.get_protein_noise_decomposition(n_samples=n_samples)


@pytest.mark.parametrize("start_training", [True, False])
def test_decomposition_keeps_batchnorm_state_and_flag(sim, start_training):
    model = _model(sim, protein_noise_model="ambient", use_batch_norm="both")
    model.is_trained_ = True
    model.module.train(start_training)
    before = _bn_counters(model.module)
    assert before
    model.get_protein_noise_decomposition(n_samples=2)
    after = _bn_counters(model.module)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert model.module.training is start_training


def test_feature_correlation_and_normalized_expression_keep_batchnorm_state(sim):
    model = _model(sim, use_batch_norm="both")
    model.is_trained_ = True
    model.module.train()
    before = _bn_counters(model.module)
    model.get_feature_correlation_matrix(n_samples=2)
    model.get_normalized_expression(n_samples_overall=10)
    after = _bn_counters(model.module)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert model.module.training is True


def test_zero_ambient_composition_is_refused(sim):
    _, profile = sim
    zero = dataclasses.replace(profile, profile=np.zeros_like(profile.profile))
    with pytest.raises(ValueError, match="positive"):
        _model(sim, zero, protein_noise_model="ambient")


def test_zero_row_in_per_batch_composition_is_refused(sim):
    adata, profile = sim
    zero = dataclasses.replace(profile, profile=np.zeros_like(profile.profile))
    batches = [str(b) for b in adata.obs["batch"].unique()]
    per_batch = {b: (zero if i == 0 else profile) for i, b in enumerate(batches)}
    with pytest.raises(ValueError, match="positive"):
        _model(sim, per_batch, protein_noise_model="ambient")


@pytest.mark.parametrize("field", ["log1p_mean", "log1p_sd", "dsb_log_mean", "dsb_log_sd"])
def test_non_finite_calibration_statistics_are_refused(sim, field):
    _, profile = sim
    bad = getattr(profile, field).copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _model(sim, dataclasses.replace(profile, **{field: bad}), protein_noise_model="ambient")


@pytest.mark.parametrize("prior", [(0.0, 0.0), (0.0, -1.0), (0.0, float("nan")), (float("inf"), 2.0)])
def test_bad_ambient_burden_prior_is_refused(sim, prior):
    with pytest.raises(ValueError, match="ambient_burden_prior"):
        _model(sim, protein_noise_model="ambient", ambient_burden_prior=prior)
