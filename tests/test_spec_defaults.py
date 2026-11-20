"""Defaults that follow released MrVI, and checkpoint compatibility with the old ones."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import scvi
import mrtotalvi
from mrtotalvi import MrTotalVI


def _adata():
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = np.array([f"d{i % 4}" for i in range(adata.n_obs)])
    return adata


def _setup(adata):
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )


@pytest.fixture()
def adata():
    adata = _adata()
    _setup(adata)
    return adata


def test_package_resolves_inside_worktree():
    assert "src/mrtotalvi" in mrtotalvi.__file__


def test_new_default_architecture(adata):
    model = MrTotalVI(adata, sample_key="sample")
    module = model.module
    assert (module.n_latent, module.n_latent_u) == (30, 10)
    assert module.qz.fc is not None and module.qz.fc.in_features == 10
    assert module.qz_kwargs["stop_gradients_mlp"] is True
    assert module.qz_kwargs["dropout_rate"] == 0.03
    assert module.qz.stop_gradients_mlp is True and module.qz.dropout_rate == 0.03
    assert module.qu_kwargs["n_layers"] == 2  # MrVI two-layer qu (p05)
    assert model.init_params_["non_kwargs"]["qu_kwargs"] == {"n_layers": 2}  # saved resolved
    assert model.init_params_["non_kwargs"]["spec_defaults"] is True


def test_user_kwargs_override_per_key(adata):
    model = MrTotalVI(
        adata,
        sample_key="sample",
        qz_kwargs={"dropout_rate": 0.0},
        qu_kwargs={"n_hidden": 16},
    )
    assert model.module.qz.dropout_rate == 0.0
    assert model.module.qz.stop_gradients_mlp is True  # default key kept
    assert model.module.qu_kwargs == {"n_layers": 2, "n_hidden": 16}  # default key kept
    assert MrTotalVI(adata, sample_key="sample", n_latent_u=None).module.qz.fc is None


def test_train_default_optimiser_and_user_override(adata):
    model = MrTotalVI(adata, sample_key="sample", n_latent=6)
    model.train(max_epochs=1, accelerator="cpu", check_val_every_n_epoch=1)
    defaults = model.trainer.optimizers[0].defaults
    assert (defaults["lr"], defaults["eps"], defaults["weight_decay"]) == (2e-3, 1e-8, 1e-8)
    assert len(model.trainer.lr_scheduler_configs) == 0  # MrVI: no plateau schedule

    model.train(
        max_epochs=1,
        accelerator="cpu",
        plan_kwargs={"eps": 0.01, "weight_decay": 1e-6},
        check_val_every_n_epoch=1,
    )
    defaults = model.trainer.optimizers[0].defaults
    assert (defaults["eps"], defaults["weight_decay"]) == (0.01, 1e-6)


def test_checkpoint_saved_with_old_defaults_loads_old_values(adata, tmp_path):
    # Old build: qz without stop-gradient or dropout, isomorphic u.
    old = MrTotalVI(adata, sample_key="sample", n_latent=6, n_latent_u=None, spec_defaults=False)
    old.save(tmp_path / "old", overwrite=True, save_anndata=False)
    # Files written before this change have no ``spec_defaults`` entry; remove it.
    path = tmp_path / "old" / "model.pt"
    saved = torch.load(path, weights_only=False)
    del saved["attr_dict"]["init_params_"]["non_kwargs"]["spec_defaults"]
    torch.save(saved, path)

    loaded = MrTotalVI.load(tmp_path / "old", adata=adata, accelerator="cpu")
    assert loaded.module.qu_kwargs.get("n_layers", 1) == 1
    assert loaded.module.qz.stop_gradients_mlp is False
    assert loaded.module.qz.dropout_rate == 0.0
    assert loaded.module.qz.fc is None
    assert loaded.init_params_["non_kwargs"]["spec_defaults"] is False
    for name, tensor in old.module.state_dict().items():
        assert torch.equal(tensor.cpu(), loaded.module.state_dict()[name].cpu()), name


def test_checkpoint_with_new_defaults_round_trips(adata, tmp_path):
    model = MrTotalVI(adata, sample_key="sample")
    model.save(tmp_path / "new", overwrite=True, save_anndata=False)
    loaded = MrTotalVI.load(tmp_path / "new", adata=adata, accelerator="cpu")
    assert (loaded.module.n_latent, loaded.module.n_latent_u) == (30, 10)
    assert loaded.module.qz.dropout_rate == 0.03 and loaded.module.qz.stop_gradients_mlp
    assert loaded.module.qu_kwargs["n_layers"] == 2


def _parameter_shift(scale_observations, plan_kwargs):
    scvi.settings.seed = 0
    adata = _adata()
    _setup(adata)
    model = MrTotalVI(adata, sample_key="sample", n_latent=10, scale_observations=scale_observations)
    before = torch.cat([p.detach().flatten().clone() for p in model.module.parameters()])
    model.train(
        max_epochs=5,
        accelerator="cpu",
        plan_kwargs=plan_kwargs,
        early_stopping=False,
        check_val_every_n_epoch=5,
    )
    after = torch.cat([p.detach().flatten() for p in model.module.parameters()])
    return (after - before).norm().item()


@pytest.mark.slow
def test_scale_observations_trains_under_new_eps():
    plain = _parameter_shift(False, None)
    scaled = _parameter_shift(True, None)
    assert scaled / plain > 0.8  # measured 0.996 on this toy set
    # With TOTALVI's eps=0.01 (the old default) the scaled loss under-trains.
    plain_old = _parameter_shift(False, {"eps": 0.01})
    scaled_old = _parameter_shift(True, {"eps": 0.01})
    assert scaled_old / plain_old < 0.7  # measured 0.47


def test_checkpoint_saved_with_one_layer_qu_default_loads_one_layer(adata, tmp_path):
    # Files saved while the qu default was one layer hold spec_defaults=True and qu_kwargs=None.
    old = MrTotalVI(adata, sample_key="sample", n_latent=6, qu_kwargs={"n_layers": 1})
    old.save(tmp_path / "p03", overwrite=True, save_anndata=False)
    path = tmp_path / "p03" / "model.pt"
    saved = torch.load(path, weights_only=False)
    saved["attr_dict"]["init_params_"]["non_kwargs"]["qu_kwargs"] = None
    torch.save(saved, path)

    loaded = MrTotalVI.load(tmp_path / "p03", adata=adata, accelerator="cpu")
    assert loaded.init_params_["non_kwargs"]["spec_defaults"] is True
    assert loaded.module.qu_kwargs.get("n_layers", 1) == 1
    for name, tensor in old.module.state_dict().items():
        assert torch.equal(tensor.cpu(), loaded.module.state_dict()[name].cpu()), name


def test_load_with_positional_prefix(adata, tmp_path):
    # load() used to read model.pt itself before scvi did, taking ``prefix`` from keyword
    # arguments only, so a positional prefix raised FileNotFoundError.
    model = MrTotalVI(adata, sample_key="sample", n_latent=6)
    model.save(tmp_path / "m", prefix="run1_", overwrite=True, save_anndata=False)
    loaded = MrTotalVI.load(tmp_path / "m", adata, "cpu", "auto", "run1_")
    assert loaded.init_params_["non_kwargs"]["spec_defaults"] is True
    assert loaded.module.qz.dropout_rate == 0.03


def test_conflicting_plan_values_warn_without_changing_precedence(adata, monkeypatch):
    """train() arguments still win over plan_kwargs (as TOTALVI), now with a warning."""
    model = MrTotalVI(
        adata,
        sample_key="sample",
        u_prior="standard",
        empirical_protein_background_prior=False,
    )
    seen = {}

    def runner(_model, **kwargs):
        seen["plan"] = kwargs["training_plan"]
        return lambda: None

    monkeypatch.setattr(model, "_train_runner_cls", runner)
    supplied = {"lr": 0.123, "n_steps_kl_warmup": 7, "n_epochs_kl_warmup": 40}
    before = supplied.copy()
    with pytest.warns(UserWarning, match="plan_kwargs values ignored") as record:
        model.train(max_epochs=1000, accelerator="cpu", plan_kwargs=supplied)
    message = next(str(w.message) for w in record if "plan_kwargs" in str(w.message))
    # Each overridden key is named with the value train() used instead.
    assert "lr=0.123 (used 0.002)" in message
    assert f"n_steps_kl_warmup=7 (used {int(0.75 * adata.n_obs)})" in message
    assert "n_epochs_kl_warmup=40 (used None)" in message

    # Expected values: the train() defaults and scvi/model/_totalvi.py:315-329.
    plan = seen["plan"]
    assert plan.lr == 2e-3
    assert plan.n_steps_kl_warmup == int(0.75 * adata.n_obs)
    assert plan.n_epochs_kl_warmup is None
    assert supplied == before


def test_matching_plan_values_do_not_warn(adata, monkeypatch):
    model = MrTotalVI(
        adata, sample_key="sample", u_prior="standard", empirical_protein_background_prior=False
    )
    monkeypatch.setattr(model, "_train_runner_cls", lambda _model, **kwargs: lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train(max_epochs=1000, accelerator="cpu", plan_kwargs={"lr": 2e-3, "eps": 1e-8})
    assert not any("plan_kwargs values ignored" in str(w.message) for w in caught)
