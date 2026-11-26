"""Opt-in protein likelihood weight (EXPERIMENTAL): scope, saving, plan_kwargs and validation."""

from __future__ import annotations

import numpy as np
import pytest
import scvi

from mrtotalvi import MrTotalVI
from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

PATHS = ["legacy", "centered_v2", "mc_samples", "scale_observations", "no_hierarchy"]
TRAIN = {"train_size": 0.7, "batch_size": 64, "accelerator": "cpu", "enable_progress_bar": False}


def _model(path="legacy", **kw):
    return MrTotalVI(
        _make_v2_adata(n_samples=3),
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2" if path == "centered_v2" else "legacy",
        scale_observations=path == "scale_observations",
        **kw,
    )


def _outputs(model, path):
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
    kwargs = {"n_samples": 2} if path == "mc_samples" else {}
    inference, generative = module(tensors, inference_kwargs=kwargs, compute_loss=False)
    if path == "no_hierarchy":
        inference = {k: v for k, v in inference.items() if k != "eps"}
    return module, tensors, inference, generative


def _fit(max_epochs=2, **kw):
    scvi.settings.seed = 0
    model = _model(**kw)
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    return model


def test_default_is_one_and_changes_nothing():
    model = _model()
    assert model.init_params_["non_kwargs"]["protein_likelihood_weight"] == 1.0
    module, tensors, inference, generative = _outputs(model, "legacy")
    base = module.loss(tensors, inference, generative, kl_weight=0.7).loss
    module.protein_likelihood_weight = 1.0
    again = module.loss(tensors, inference, generative, kl_weight=0.7).loss
    assert base.item() == again.item()


def test_early_stopping_monitor_stays_elbo_validation():
    model = _fit(protein_likelihood_weight=2.0, max_epochs=2)
    assert "elbo_validation" in model.history
    assert not any(k.startswith("weighted_loss") for k in model.history)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kl_weight", [1.0, 0.25])
def test_weight_scales_protein_reconstruction_and_background_terms(path, kl_weight):
    model = _model(path)
    module, tensors, inference, generative = _outputs(model, path)

    def loss_at(weight):
        module.protein_likelihood_weight = weight
        return module.loss(tensors, inference, generative, kl_weight=kl_weight)

    one, two = loss_at(1.0), loss_at(2.0)
    recon = one.reconstruction_loss["reconst_loss_protein"]
    back = one.kl_local["kl_div_back_pro"]
    per_cell = recon + back
    if path == "scale_observations":
        counts = module.n_obs_per_sample[tensors["sample"].long().flatten()]
        per_cell = per_cell / counts
    # Annealed by kl_weight, as the unweighted protein terms are (G94).
    expected = kl_weight * per_cell.mean().item()
    assert (two.loss - one.loss).item() == pytest.approx(expected, rel=1e-5, abs=1e-6)
    # The reported components stay unweighted.
    for key, value in one.reconstruction_loss.items():
        np.testing.assert_array_equal(value.detach().numpy(), two.reconstruction_loss[key].detach().numpy())
    for key, value in one.kl_local.items():
        np.testing.assert_array_equal(value.detach().numpy(), two.kl_local[key].detach().numpy())


def test_weight_is_saved_and_round_trips(tmp_path):
    model = _model(protein_likelihood_weight=5.0)
    assert model.init_params_["non_kwargs"]["protein_likelihood_weight"] == 5.0
    assert model.module.protein_likelihood_weight == 5.0
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=model.adata)
    assert loaded.init_params_["non_kwargs"]["protein_likelihood_weight"] == 5.0
    assert loaded.module.protein_likelihood_weight == 5.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_weight_raises(bad):
    with pytest.raises(ValueError, match="protein_likelihood_weight"):
        _model(protein_likelihood_weight=bad)


def test_plan_kwargs_weight_conflict_raises():
    model = _model(protein_likelihood_weight=2.0)
    for given in (3.0, 1.0):
        with pytest.raises(ValueError, match="pro_recons_weight"):
            model.train(max_epochs=1, plan_kwargs={"pro_recons_weight": given}, **TRAIN)


def test_plan_kwargs_weight_equal_to_argument_is_applied_once():
    history = []
    for plan_kwargs in (None, {"pro_recons_weight": 2.0}):
        scvi.settings.seed = 1
        model = _model(protein_likelihood_weight=2.0)
        model.train(max_epochs=1, plan_kwargs=plan_kwargs, **TRAIN)
        history.append(model.history["elbo_train"].to_numpy(dtype=float))
    np.testing.assert_allclose(history[0], history[1])


def test_plan_kwargs_alone_keeps_the_scvi_reconstruction_only_behaviour():
    _model().train(max_epochs=1, plan_kwargs={"pro_recons_weight": 3.0}, **TRAIN)
