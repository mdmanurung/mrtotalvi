"""Learnable ambient composition fitted jointly on cell-free droplet counts (opt-in)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import simulate_citeseq


@pytest.fixture(scope="module")
def sim():
    adata, _, empties = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=800)
    profile = estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    return adata, profile


def _model(sim, **kwargs):
    adata, profile = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile, n_latent=4, **kwargs), adata


def test_profile_keeps_the_summed_droplet_counts(sim):
    _, profile = sim
    # Independent expectation: the profile is the normalised sum, so sum * profile = counts.
    assert profile.droplet_counts is not None
    np.testing.assert_allclose(profile.droplet_counts / profile.droplet_counts.sum(), profile.profile)
    rev = profile.aligned(profile.feature_names[::-1])
    np.testing.assert_allclose(rev.droplet_counts, profile.droplet_counts[::-1])


def test_fixed_profile_has_no_logits(sim):
    model, _ = _model(sim, protein_noise_model="ambient")
    assert not hasattr(model.module, "ambient_logits")


def test_learnable_needs_an_ambient_mode(sim):
    with pytest.raises(ValueError, match="learn_ambient_profile"):
        _model(sim, learn_ambient_profile=True)


def test_learnable_needs_droplet_counts(sim):
    adata, profile = sim
    bare = profile.aligned(profile.feature_names)
    bare.droplet_counts = None
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    with pytest.raises(ValueError, match="droplet_counts"):
        MrTotalVI(
            adata, sample_key="donor", ambient_profile=bare, protein_noise_model="ambient", learn_ambient_profile=True
        )


def test_learnable_starts_at_the_profile_with_zero_penalty(sim):
    model, _ = _model(sim, protein_noise_model="ambient", learn_ambient_profile=True)
    comp = model.module.ambient_composition().detach().numpy()
    np.testing.assert_allclose(comp[0], sim[1].profile, atol=1e-6)
    # At the droplet MLE the penalty N * KL(p_hat || a) is zero.
    assert float(model.module.ambient_droplet_penalty()) == pytest.approx(0.0, abs=1e-3)


def test_penalty_is_counts_times_kl(sim):
    model, _ = _model(sim, protein_noise_model="ambient", learn_ambient_profile=True)
    counts = sim[1].droplet_counts
    p_hat = counts / counts.sum()
    a = np.full_like(p_hat, 1.0 / p_hat.size)
    with torch.no_grad():
        model.module.ambient_logits.zero_()
    keep = p_hat > 0
    # Independent expectation: closed-form multinomial KL, written out in numpy.
    expected = counts.sum() * np.sum(p_hat[keep] * np.log(p_hat[keep] / a[keep]))
    assert float(model.module.ambient_droplet_penalty()) == pytest.approx(expected, rel=1e-4)


def test_learnable_trains_moves_logits_and_round_trips(sim, tmp_path):
    model, adata = _model(sim, protein_noise_model="ambient", learn_ambient_profile=True)
    before = model.module.ambient_logits.detach().clone()
    model.train(max_epochs=2, accelerator="cpu", batch_size=128)
    assert not torch.equal(before, model.module.ambient_logits.detach())
    comp = model.get_protein_noise_decomposition(n_samples=3)
    assert np.isfinite(comp["ambient"].values).all() and (comp["ambient"].values > 0).all()
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=adata)
    torch.testing.assert_close(loaded.module.ambient_logits.cpu(), model.module.ambient_logits.cpu())
