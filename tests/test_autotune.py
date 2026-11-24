"""Autotune support: residual-collapse metrics and the run_autotune wrapper (ADR-0013)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

import mrtotalvi
from mrtotalvi import MrTotalVI
from mrtotalvi._module import residual_collapse_metrics
from tests._simulate_adt import simulate_citeseq

needs_ray = pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="needs mrtotalvi[autotune]")


@pytest.fixture(scope="module")
def adata():
    adata, _, _ = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=10)
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return adata


def test_collapse_metrics_match_a_numpy_closed_form():
    rng = np.random.default_rng(0)
    # Two live residual dimensions (sd 1 and 0.5), one dead (sd 0.01 -> var 1e-4 < 0.01).
    eps = rng.normal(size=(500, 3)) * np.array([1.0, 0.5, 0.01])
    z = eps + rng.normal(size=(500, 3)) * 2.0
    got = residual_collapse_metrics(torch.as_tensor(eps), torch.as_tensor(z))
    assert got["residual_active_units"].item() == 2
    expected_share = (eps**2).sum(1).mean() / (z**2).sum(1).mean()
    np.testing.assert_allclose(got["residual_share"].item(), expected_share, rtol=1e-10)


def test_collapsed_residual_reads_zero():
    z = torch.randn(64, 4)
    got = residual_collapse_metrics(torch.zeros(64, 4), z)
    assert got["residual_active_units"].item() == 0
    assert got["residual_share"].item() == 0


def test_collapse_metrics_use_the_residual_posterior_mean(adata):
    # With use_map=False, eps is a draw from q(eps); the diagnostics must read q(eps)'s mean
    # (Burda et al. define active units on posterior means), so a collapsed q(eps) = N(0, s)
    # reads as inactive instead of as noise. Codex review, PR #8.
    model = MrTotalVI(adata, sample_key="donor", n_latent=4, use_map=False)
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=64)))
    inference, _, loss = model.module(tensors)
    mean = inference["eps_dist"].loc
    expected = residual_collapse_metrics(mean, inference["z_base"] + mean)
    for key, value in expected.items():
        assert loss.extra_metrics[key].item() == pytest.approx(value.item(), rel=1e-6)
    drawn = residual_collapse_metrics(inference["eps"], inference["z"])
    assert drawn["residual_share"].item() != pytest.approx(expected["residual_share"].item(), rel=1e-3)


def test_training_logs_the_collapse_metrics(adata):
    model = MrTotalVI(adata, sample_key="donor", n_latent=4)
    model.train(max_epochs=2, check_val_every_n_epoch=1, batch_size=64)
    for key in ("residual_active_units", "residual_share"):
        for stage in ("train", "validation"):
            series = model.history[f"{key}_{stage}"].iloc[:, 0].astype(float)
            assert len(series) == 2
            assert np.isfinite(series).all()
    assert (model.history["residual_active_units_validation"].iloc[:, 0].astype(float) <= 4).all()


@needs_ray
def test_wrapper_fills_sample_key_and_adds_guard_metrics(adata, monkeypatch):
    seen = {}

    def fake(model_cls, data, metrics, mode, search_space, num_samples, **kwargs):
        seen.update(model_cls=model_cls, metrics=metrics, search_space=search_space, kwargs=kwargs)
        return "experiment"

    monkeypatch.setattr("scvi.autotune.run_autotune", fake)
    out = mrtotalvi.run_autotune(
        adata,
        metrics="elbo_validation",
        mode="min",
        search_space={"model_params": {"n_latent": [4, 8]}},
        num_samples=2,
        seed=1,
    )
    assert out == "experiment"
    assert seen["model_cls"] is MrTotalVI
    assert seen["metrics"] == ["elbo_validation", "residual_active_units_validation", "residual_share_validation"]
    assert seen["search_space"]["model_params"] == {"sample_key": "donor", "n_latent": [4, 8]}
    assert seen["kwargs"] == {"seed": 1}


@needs_ray
def test_wrapper_leaves_scib_metrics_alone(adata, monkeypatch):
    seen = {}
    monkeypatch.setattr("scvi.autotune.run_autotune", lambda *a, **k: seen.update(metrics=a[2]))
    mrtotalvi.run_autotune(
        adata, metrics="iLISI", mode="max", search_space={"train_params": {"lr": [1e-3]}}, num_samples=1
    )
    assert seen["metrics"] == ["iLISI"]


@needs_ray
def test_wrapper_rejects_a_conflicting_sample_key(adata):
    with pytest.raises(ValueError, match="sample_key"):
        mrtotalvi.run_autotune(
            adata,
            metrics="elbo_validation",
            mode="min",
            search_space={"model_params": {"sample_key": "batch"}},
            num_samples=1,
        )


@pytest.mark.slow
@needs_ray
def test_run_autotune_end_to_end(adata, tmp_path):
    import ray
    from ray import tune

    # A small local Ray instance: run_autotune would otherwise claim every CPU on the node.
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    experiment = mrtotalvi.run_autotune(
        adata,
        metrics="elbo_validation",
        mode="min",
        search_space={
            "model_params": {"n_latent": tune.choice([4, 6])},
            "train_params": {"max_epochs": 2, "batch_size": 64},
        },
        num_samples=2,
        scheduler="fifo",
        searcher="random",
        seed=0,
        resources={"cpu": 1},
        logging_dir=str(tmp_path),
        ignore_reinit_error=True,
    )
    ray.shutdown()
    frame = experiment.result_grid.get_dataframe()
    assert len(frame) == 2
    for column in ("elbo_validation", "residual_active_units_validation", "residual_share_validation"):
        assert frame[column].notna().all()
