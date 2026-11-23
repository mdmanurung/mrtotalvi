"""Donor-balanced validation loss is logged next to the unchanged elbo monitor."""

from __future__ import annotations

import numpy as np
import pytest
import scvi
import torch

from mrtotalvi import MrTotalVI

SETUP = {"protein_expression_obsm_key": "protein_expression", "sample_key": "sample", "batch_key": "batch"}


def _adata():
    adata = scvi.data.synthetic_iid(n_genes=20, n_proteins=5, batch_size=50)
    # unequal donors so that donor weights differ from cell weights
    adata.obs["sample"] = ["s0"] * 40 + ["s1"] * 25 + ["s2"] * 15 + ["s3"] * 20
    MrTotalVI.setup_anndata(adata, **SETUP)
    return adata


def _fit(**kw):
    scvi.settings.seed = 0
    model = MrTotalVI(_adata(), sample_key="sample", n_latent=5, n_hidden=16, **kw)
    model.train(
        max_epochs=2,
        train_size=0.7,
        batch_size=64,
        accelerator="cpu",
        check_val_every_n_epoch=1,
        early_stopping=False,
        enable_progress_bar=False,
    )
    return model


@pytest.fixture(scope="module")
def scaled_model():
    return _fit(scale_observations=True)


def test_donor_balanced_metric_logged_when_scaled(scaled_model):
    history = scaled_model.history
    assert "donor_balanced_loss_validation" in history
    assert len(history["donor_balanced_loss_validation"]) == 2
    assert np.isfinite(history["donor_balanced_loss_validation"].to_numpy(dtype=float)).all()


def test_default_history_has_no_new_metric_and_elbo_is_unchanged():
    # Default fit: no new metric, and the same seed gives the same elbo_validation.
    default = _fit().history
    assert not any(k.startswith("donor_balanced") for k in default)
    np.testing.assert_array_equal(default["elbo_validation"], _fit().history["elbo_validation"])


def test_metric_equals_training_loss(scaled_model):
    model = scaled_model
    tensors = next(iter(model._make_data_loader(model.adata, batch_size=64, shuffle=False)))
    model.module.eval()
    with torch.no_grad():
        out = model.module(tensors, loss_kwargs={"kl_weight": 1.0})[2]
    torch.testing.assert_close(out.extra_metrics["donor_balanced_loss"], out.loss.detach())
