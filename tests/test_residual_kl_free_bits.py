"""KL weight and free-bits floor on the protein residual latent w (EXPERIMENTAL)."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import scvi
import torch
from scvi import REGISTRY_KEYS
from torch.distributions import Normal

from mrtotalvi import MrTotalVI
from tests.test_protein_residual_latent import PATHS, TRAIN, _outputs
from tests.test_protein_residual_latent import _model as _w_model

# State-dict SHA-256 after 3 seeded epochs, recorded on main before the knobs existed.
MAIN_HASH_W10 = "5787b2be294f76cffe317b0347a423efc087f3763f64546dca5b9b876c32caeb"
MAIN_HASH_W0 = "64a45f246e7862756cf810fa85330ddab3b83310f27b80dab0ecc23c4e22e8f7"
KNOBS = {
    "protein_residual_kl_weight": 1.0,
    "protein_residual_free_bits": 0.0,
    "protein_residual_free_bits_granularity": "per_dim",
    "protein_residual_free_bits_schedule": "constant",
    "protein_residual_free_bits_statistic": "batch_mean",
}


def _state_hash(n_w):
    scvi.settings.seed = 0
    model = _w_model(n_w=n_w)
    model.train(max_epochs=3, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    digest = hashlib.sha256()
    for key, value in model.module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _kl_to_loc(kl):
    """Posterior mean of N(mu, 1) whose KL to N(0, 1) is ``kl`` nats: KL = mu^2 / 2."""
    return torch.sqrt(2 * torch.as_tensor(kl, dtype=torch.float32))


def _loss(module, tensors, inference, generative, loc, kl_weight=1.0, **knobs):
    """Module loss with q(w) forced to N(loc, 1); ``loc`` has shape (batch, n_w)."""
    for name, value in {**KNOBS, **knobs}.items():
        setattr(module, name, value)
    patched = {**inference, "qw": Normal(loc, torch.ones_like(loc))}
    return module.loss(tensors, patched, generative, kl_weight=kl_weight)


def _setup(n_w, path="legacy"):
    return _outputs(_w_model(path, n_w=n_w), path)


def _floor_term(n_w, kl_per_cell, lam, **knobs):
    """Mean over cells of the floored term, read off the legacy-path loss at kl_weight 1."""
    module, tensors, inference, generative = _setup(n_w)
    loc = _kl_to_loc(kl_per_cell)
    off = _loss(module, tensors, inference, generative, loc)
    on = _loss(module, tensors, inference, generative, loc, protein_residual_free_bits=lam, **knobs)
    true_kl = off.kl_local["kl_div_w"].mean().item()
    return (on.loss - off.loss).item() + true_kl, on, true_kl


def _cells(*rows):
    """Eight cells made of the given rows, each repeated equally: shape (8, n_w) of per-dimension KL."""
    return torch.tensor(rows, dtype=torch.float32).repeat_interleave(8 // len(rows), dim=0)


# -- defaults ------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_w", "expected"), [(10, MAIN_HASH_W10), (0, MAIN_HASH_W0)])
def test_defaults_reproduce_main_state_dict_byte_for_byte(n_w, expected):
    assert _state_hash(n_w) == expected


def test_default_values_are_saved():
    model = _w_model()
    assert {k: model.init_params_["non_kwargs"][k] for k in KNOBS} == KNOBS
    assert all(getattr(model.module, k) == v for k, v in KNOBS.items())


# -- beta_w --------------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kl_weight", [1.0, 0.7])
def test_beta_w_scales_the_kl_term_exactly(path, kl_weight):
    module, tensors, inference, generative = _setup(10, path)
    loc = torch.full((8, 10), 0.5)  # KL = 10 * 0.5 * 0.25 = 1.25 nats per cell

    def loss(beta):
        return _loss(module, tensors, inference, generative, loc, kl_weight, protein_residual_kl_weight=beta).loss

    per_cell = 1.25
    if path == "scale_observations":
        counts = module.n_obs_per_sample[tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()]
        per_cell = (1.25 / counts).mean().item()
    assert (loss(2.0) - loss(1.0)).item() == pytest.approx(kl_weight * per_cell, rel=1e-5)
    assert (loss(0.5) - loss(1.0)).item() == pytest.approx(-0.5 * kl_weight * per_cell, rel=1e-5)


# -- free-bits arithmetic ------------------------------------------------------------------


def test_floor_on_per_dim_batch_mean_kl_gives_0_75():
    # KL per dimension 0.1 and 0.5, floor 0.25 nats: max(0.25, 0.1) + max(0.25, 0.5) = 0.75.
    term, on, true_kl = _floor_term(2, _cells([0.1, 0.5]), 0.25)
    assert term == pytest.approx(0.75, abs=1e-5)
    assert true_kl == pytest.approx(0.6, abs=1e-5)
    # The logged KL is the true one, not the clamped one.
    assert on.kl_local["kl_div_w"].mean().item() == pytest.approx(0.6, abs=1e-5)
    assert on.extra_metrics["kl_w"].item() == pytest.approx(0.6, abs=1e-5)


def test_floor_removes_the_gradient_below_it_and_keeps_it_above():
    module, tensors, inference, generative = _setup(2)
    mu = _kl_to_loc(torch.tensor([0.1, 0.5])).requires_grad_(True)
    out = _loss(module, tensors, inference, generative, mu.expand(8, 2), protein_residual_free_bits=0.25)
    # d/dmu of mean-over-cells KL = mu^2 / 2 is mu; q(w) is patched in after inference, so no
    # other loss term depends on mu. Floored dimension: 0. Dimension above the floor: mu = 1.
    grad = torch.autograd.grad(out.loss, mu)[0]
    np.testing.assert_allclose(grad.numpy(), [0.0, 1.0], atol=1e-5)


def test_batch_mean_and_per_cell_statistics_differ_as_documented():
    # One dimension, half the cells at KL 0.1 and half at 0.5, floor 0.25.
    cells = _cells([0.1], [0.5])
    # Batch mean 0.3 is above the floor: unclamped, term 0.3.
    term, _, _ = _floor_term(1, cells, 0.25, protein_residual_free_bits_statistic="batch_mean")
    assert term == pytest.approx(0.3, abs=1e-5)
    # Per-cell clamp: (0.25 + 0.5) / 2 = 0.375.
    term, _, true_kl = _floor_term(1, cells, 0.25, protein_residual_free_bits_statistic="per_cell")
    assert term == pytest.approx(0.375, abs=1e-5)
    assert true_kl == pytest.approx(0.3, abs=1e-5)


def test_all_grouping_clamps_the_summed_kl():
    cells = _cells([0.1, 0.5])  # group KL 0.6 per cell
    assert _floor_term(2, cells, 0.25, protein_residual_free_bits_granularity="group")[0] == pytest.approx(
        0.6, abs=1e-5
    )
    # Group KL 0.6 is under a floor of 1.0: the whole group sits at 1.0.
    assert _floor_term(2, cells, 1.0, protein_residual_free_bits_granularity="group")[0] == pytest.approx(1.0, abs=1e-5)
    # Per-dim at the same floor: max(1, 0.1) + max(1, 0.5) = 2.
    assert _floor_term(2, cells, 1.0, protein_residual_free_bits_granularity="per_dim")[0] == pytest.approx(
        2.0, abs=1e-5
    )
    # Per-cell, one group: cells at group KL 0.2 and 1.0, floor 0.5 -> (0.5 + 1.0) / 2.
    cells = _cells([0.1, 0.1], [0.5, 0.5])
    term = _floor_term(
        2, cells, 0.5, protein_residual_free_bits_granularity="group", protein_residual_free_bits_statistic="per_cell"
    )[0]
    assert term == pytest.approx(0.75, abs=1e-5)


def test_floor_below_the_kl_changes_nothing():
    term, _, true_kl = _floor_term(2, _cells([0.4, 0.5]), 0.25)
    assert term == pytest.approx(true_kl, abs=1e-6)


def test_floor_is_scaled_by_beta_and_kl_weight():
    module, tensors, inference, generative = _setup(2)
    loc = _kl_to_loc(_cells([0.1, 0.5]))
    off = _loss(module, tensors, inference, generative, loc, 0.7, protein_residual_kl_weight=2.0)
    on = _loss(
        module,
        tensors,
        inference,
        generative,
        loc,
        0.7,
        protein_residual_kl_weight=2.0,
        protein_residual_free_bits=0.25,
    )
    assert (on.loss - off.loss).item() == pytest.approx(0.7 * 2.0 * (0.75 - 0.6), abs=1e-5)


def test_floor_under_donor_weights_clamps_the_unweighted_batch_mean():
    module, tensors, inference, generative = _setup(1, "scale_observations")
    counts = module.n_obs_per_sample[tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()]
    loc = _kl_to_loc(torch.full((8, 1), 0.1))
    off = _loss(module, tensors, inference, generative, loc)
    on = _loss(module, tensors, inference, generative, loc, protein_residual_free_bits=0.25)
    # Every cell is clamped to 0.25 nats; each cell keeps its own donor weight 1 / n_sample_cells.
    assert (on.loss - off.loss).item() == pytest.approx(((0.25 - 0.1) / counts).mean().item(), rel=1e-4)


# -- schedule ------------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
def test_after_warmup_is_inactive_during_warmup_and_active_at_full_kl_weight(path):
    module, tensors, inference, generative = _setup(2, path)
    loc = _kl_to_loc(_cells([0.1, 0.5]))
    for kl_weight, active in [(0.0, False), (0.5, False), (0.999, False), (1.0, True)]:
        off = _loss(module, tensors, inference, generative, loc, kl_weight)
        on = _loss(
            module,
            tensors,
            inference,
            generative,
            loc,
            kl_weight,
            protein_residual_free_bits=0.25,
            protein_residual_free_bits_schedule="after_warmup",
        )
        always = _loss(module, tensors, inference, generative, loc, kl_weight, protein_residual_free_bits=0.25)
        if active:
            assert on.loss.item() == pytest.approx(always.loss.item(), rel=1e-6)
            assert on.loss.item() > off.loss.item()
        else:
            assert on.loss.item() == off.loss.item()
            assert kl_weight == 0.0 or always.loss.item() > off.loss.item()


# -- logging -------------------------------------------------------------------------------


def test_per_dimension_kl_is_logged_unclamped():
    module, tensors, inference, generative = _setup(2)
    loc = _kl_to_loc(_cells([0.1, 0.5]))
    out = _loss(module, tensors, inference, generative, loc, protein_residual_free_bits=0.25)
    np.testing.assert_allclose(out.extra_metrics["kl_w_dim_0"].item(), 0.1, atol=1e-6)
    np.testing.assert_allclose(out.extra_metrics["kl_w_dim_1"].item(), 0.5, atol=1e-6)


def test_fit_logs_per_dimension_kl_with_a_floor():
    scvi.settings.seed = 0
    model = _w_model(protein_residual_free_bits=0.5)
    model.train(max_epochs=2, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    assert "kl_w_train" in model.history
    assert math.isfinite(float(model.history["kl_w_dim_0_train"].iloc[-1, 0]))


# -- saving and validation -----------------------------------------------------------------


def test_init_params_round_trip(tmp_path):
    knobs = {
        "protein_residual_kl_weight": 0.5,
        "protein_residual_free_bits": 0.125,
        "protein_residual_free_bits_granularity": "group",
        "protein_residual_free_bits_schedule": "after_warmup",
        "protein_residual_free_bits_statistic": "per_cell",
    }
    model = _w_model(**knobs)
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=model.adata)
    for name, value in knobs.items():
        assert model.init_params_["non_kwargs"][name] == value
        assert loaded.init_params_["non_kwargs"][name] == value
        assert getattr(loaded.module, name) == value


@pytest.mark.parametrize(
    "knob",
    [
        {"protein_residual_kl_weight": 0.5},
        {"protein_residual_free_bits": 0.25},
        {"protein_residual_free_bits_granularity": "group"},
        {"protein_residual_free_bits_schedule": "after_warmup"},
        {"protein_residual_free_bits_statistic": "per_cell"},
    ],
)
def test_non_default_knob_without_residual_latent_raises(knob):
    with pytest.raises(ValueError, match="n_latent_protein_residual"):
        _w_model(n_w=0, **knob)


@pytest.mark.parametrize(
    "knob",
    [
        {"protein_residual_kl_weight": -0.1},
        {"protein_residual_kl_weight": float("nan")},
        {"protein_residual_free_bits": -0.1},
        {"protein_residual_free_bits": float("inf")},
        {"protein_residual_free_bits_granularity": "k3"},
        {"protein_residual_free_bits_schedule": "ramp"},
        {"protein_residual_free_bits_statistic": "median"},
    ],
)
def test_invalid_knob_raises(knob):
    with pytest.raises(ValueError):
        _w_model(**knob)
