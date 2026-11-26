"""Opt-in protein residual latent w (EXPERIMENTAL): encoder, KL, decoder heads, saving and logging."""

from __future__ import annotations

import numpy as np
import pytest
import scvi
import torch
from scvi import REGISTRY_KEYS
from scvi.nn import DecoderTOTALVI
from torch.distributions import Normal

from mrtotalvi import MrTotalVI
from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

N_W = 10
PATHS = ["legacy", "centered_v2", "mc_samples", "scale_observations", "no_hierarchy"]
TRAIN = {"train_size": 0.7, "batch_size": 64, "accelerator": "cpu", "enable_progress_bar": False}


def _model(path="legacy", n_w=N_W, **kw):
    return MrTotalVI(
        _make_v2_adata(n_samples=3),
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2" if path == "centered_v2" else "legacy",
        scale_observations=path == "scale_observations",
        n_latent_protein_residual=n_w,
        **kw,
    )


def _outputs(model, path="legacy"):
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
    kwargs = {"n_samples": 2} if path == "mc_samples" else {}
    inference, generative = module(tensors, inference_kwargs=kwargs, compute_loss=False)
    if path == "no_hierarchy":
        inference = {k: v for k, v in inference.items() if k != "eps"}
    return module, tensors, inference, generative


def _generate(module, tensors, inference, w):
    """Run the decoder on the cell's own z with the given w (None = the decoder default of 0)."""
    inputs = module._get_generative_input(tensors, inference)
    inputs["w"] = w
    return module.generative(**inputs)


def _fit(max_epochs=2, **kw):
    scvi.settings.seed = 0
    model = _model(**kw)
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    return model


# -- default off --------------------------------------------------------------------------


def test_default_is_off_and_builds_no_extra_modules():
    model = _make_default()
    assert model.init_params_["non_kwargs"]["n_latent_protein_residual"] == 0
    assert model.module.n_latent_protein_residual == 0
    decoder = model.module.decoder
    assert type(decoder) is DecoderTOTALVI
    assert not any(key.startswith("qw") for key in model.module.state_dict())
    inference = _outputs(model)[2]
    assert "w" not in inference and "qw" not in inference
    explicit = _make_default(n_latent_protein_residual=0)
    assert model.module.state_dict().keys() == explicit.module.state_dict().keys()


def _make_default(**kw):
    return MrTotalVI(_make_v2_adata(n_samples=3), sample_key="sample", n_latent=3, n_latent_u=2, **kw)


@pytest.mark.parametrize("path", PATHS)
def test_off_loss_has_no_kl_w(path):
    scvi.settings.seed = 0
    model = _model(path, n_w=0)
    module, tensors, inference, generative = _outputs(model, path)
    out = module.loss(tensors, inference, generative, kl_weight=0.7)
    assert "kl_div_w" not in out.kl_local
    assert "kl_w" not in out.extra_metrics


# -- KL of w ------------------------------------------------------------------------------


class _FixedQw(torch.nn.Module):
    def __init__(self, loc, n_w=N_W):
        super().__init__()
        self.loc, self.n_w = loc, n_w

    def forward(self, y, sample_index=None):
        shape = (y.shape[0], self.n_w)
        return Normal(torch.full(shape, self.loc), torch.ones(shape))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kl_weight", [1.0, 0.7])
def test_kl_of_shifted_posterior_is_exactly_half_n_dim_times_shift_squared(path, kl_weight):
    model = _model(path)
    module, tensors, inference, generative = _outputs(model, path)
    y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

    def loss_with(loc):
        patched = {**inference, "qw": _FixedQw(loc)(y)}
        return module.loss(tensors, patched, generative, kl_weight=kl_weight)

    base, shifted = loss_with(0.0), loss_with(0.5)
    expected = 0.5 * N_W * 0.25  # 1.25 nats per cell
    np.testing.assert_allclose(base.kl_local["kl_div_w"].detach().numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(shifted.kl_local["kl_div_w"].detach().numpy(), expected, rtol=1e-6)
    # Weight 1, annealed with kl_weight like kl_z.
    per_cell = expected
    if path == "scale_observations":
        counts = module.n_obs_per_sample[tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()]
        per_cell = (expected / counts).mean().item()
    assert (shifted.loss - base.loss).item() == pytest.approx(kl_weight * per_cell, rel=1e-5)


# -- decoder heads ------------------------------------------------------------------------


def _perturbed(model, path="legacy", gene_likelihood=None):
    module, tensors, inference, _ = _outputs(model, path)
    torch.manual_seed(0)
    w0 = torch.zeros_like(inference["w"])
    w1 = torch.randn_like(inference["w"]) * 3.0
    return _generate(module, tensors, inference, w0), _generate(module, tensors, inference, w1)


@pytest.mark.parametrize("heads", ["fore_mixing", "all"])
@pytest.mark.parametrize("path", ["legacy", "mc_samples"])
def test_perturbing_w_leaves_rna_scale_exactly_unchanged(heads, path):
    model = _model(path, protein_residual_heads=heads)
    a, b = _perturbed(model, path)
    assert torch.equal(a["px_"]["scale"], b["px_"]["scale"])
    assert torch.equal(a["px_"]["rate"], b["px_"]["rate"])
    assert (a["px_"]["scale"] - b["px_"]["scale"]).abs().max().item() == 0.0
    # The protein side does move.
    assert not torch.equal(a["py_"]["mixing"], b["py_"]["mixing"])
    assert not torch.equal(a["py_"]["fore_scale"], b["py_"]["fore_scale"])


def test_fore_mixing_leaves_background_head_unchanged_and_all_changes_it():
    a, b = _perturbed(_model(protein_residual_heads="fore_mixing"))
    assert torch.equal(a["py_"]["back_alpha"], b["py_"]["back_alpha"])
    assert torch.equal(a["py_"]["back_beta"], b["py_"]["back_beta"])
    a, b = _perturbed(_model(protein_residual_heads="all"))
    assert not torch.equal(a["py_"]["back_alpha"], b["py_"]["back_alpha"])
    assert not torch.equal(a["py_"]["back_beta"], b["py_"]["back_beta"])


def test_zinb_leaks_w_into_rna_dropout_but_not_into_rna_scale():
    # G93: the mixing trunk (sigmoid_decoder) is shared with the RNA dropout head, so under the zinb gene
    # likelihood w reaches px dropout. The RNA scale stays exact. Documented, not blocked.
    a, b = _perturbed(_model(gene_likelihood="zinb"))
    assert torch.equal(a["px_"]["scale"], b["px_"]["scale"])
    assert not torch.equal(a["px_"]["dropout"], b["px_"]["dropout"])


# -- counterfactual and DE decodes: w = 0 --------------------------------------------------


@torch.no_grad()
def test_counterfactual_decode_equals_generative_with_w_zero():
    model = _model()
    module, tensors, inference, _ = _outputs(model)
    gen = _generate(module, tensors, inference, torch.zeros_like(inference["w"]))
    py_ = gen["py_"]
    det = module._deterministic_decoder_parameters(
        inference["z"],
        inference["library_gene"],
        tensors[REGISTRY_KEYS.BATCH_KEY],
    )
    np.testing.assert_allclose(
        det["protein_foreground_probability"].numpy(), (1 - torch.sigmoid(py_["mixing"])).numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(det["rna_scale"].numpy(), gen["px_"]["scale"].numpy(), rtol=1e-6)
    # Default of the decoder (no w passed) is the same decode: nothing in a w-free path draws a w.
    default = _generate(module, tensors, inference, None)
    assert torch.equal(default["py_"]["mixing"], py_["mixing"])
    assert torch.equal(default["py_"]["fore_scale"], py_["fore_scale"])


def test_counterfactual_outputs_ignore_the_cells_own_w():
    model = _model("centered_v2", protein_residual_heads="all")
    model.is_trained_ = True
    first = model.get_counterfactual_expression(model.adata)
    # A far-away q(w) changes the cell's own w draws, which a w-free decode must not notice.
    model.module.qw = _FixedQw(8.0)
    second = model.get_counterfactual_expression(model.adata)
    assert set(first.data_vars) == set(second.data_vars)
    for key in first.data_vars:
        np.testing.assert_array_equal(first[key].to_numpy(), second[key].to_numpy())


def test_de_hook_ignores_w():
    model = _model()
    module, tensors, inference, _ = _outputs(model)
    kwargs = {
        "x": tensors[REGISTRY_KEYS.X_KEY],
        "y": tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY],
        "sample_index": tensors[REGISTRY_KEYS.SAMPLE_KEY],
        "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
        "extra_eps": torch.zeros(tensors[REGISTRY_KEYS.X_KEY].shape[0], 3),
        "u_anchor": inference["u"],
        "_lfc_aux": {"library_gene": inference["library_gene"]},
    }
    first = module.compute_h_from_x_eps(**kwargs)
    second = module.compute_h_from_x_eps(**kwargs)
    assert torch.equal(first, second)
    # Decoding with explicit w = 0 gives the same protein scale: the hook does not read any w.
    z = inference["z_base"] + 0.0
    zero = module.generative(
        z=z,
        library_gene=inference["library_gene"],
        batch_index=tensors[REGISTRY_KEYS.BATCH_KEY],
        label=tensors[REGISTRY_KEYS.LABELS_KEY],
        w=torch.zeros(z.shape[0], N_W),
    )
    mixing = torch.sigmoid(zero["py_"]["mixing"])
    expected = torch.nn.functional.normalize(
        (1.0 - mixing) * torch.exp(zero["py_"]["back_alpha"]) * zero["py_"]["fore_scale"], p=1, dim=-1
    )
    torch.testing.assert_close(first[:, -expected.shape[1] :], expected)


# -- factual readouts use the cell's own q(w) ----------------------------------------------


def test_factual_readout_uses_the_cells_own_w_draws():
    model = _model(protein_residual_heads="all")
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
    torch.manual_seed(0)
    own = module.protein_components(tensors, n_samples=1)
    torch.manual_seed(0)
    again = module.protein_components(tensors, n_samples=1)
    torch.testing.assert_close(own["native"], again["native"])
    # Replace the encoder by a far-away posterior: the readout moves, so it does read q(w).
    module.qw = _FixedQw(8.0)
    torch.manual_seed(0)
    moved = module.protein_components(tensors, n_samples=1)
    assert not torch.allclose(own["native"], moved["native"])


# -- saving and validation ----------------------------------------------------------------


def test_init_params_round_trip(tmp_path):
    model = _model(protein_residual_heads="all")
    assert model.init_params_["non_kwargs"]["n_latent_protein_residual"] == N_W
    assert model.init_params_["non_kwargs"]["protein_residual_heads"] == "all"
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=model.adata)
    assert loaded.init_params_["non_kwargs"]["n_latent_protein_residual"] == N_W
    assert loaded.init_params_["non_kwargs"]["protein_residual_heads"] == "all"
    assert loaded.module.n_latent_protein_residual == N_W
    assert loaded.module.protein_residual_heads == "all"
    for (k1, v1), (k2, v2) in zip(model.module.state_dict().items(), loaded.module.state_dict().items(), strict=True):
        assert k1 == k2
        assert torch.equal(v1.cpu(), v2.cpu())


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_invalid_size_raises(bad):
    with pytest.raises((ValueError, TypeError), match="n_latent_protein_residual"):
        _model(n_w=bad)


def test_invalid_heads_raises():
    with pytest.raises(ValueError, match="protein_residual_heads"):
        _model(protein_residual_heads="background")


@pytest.mark.parametrize("mode", ["ambient", "ambient_isotype"])
def test_ambient_modes_raise_not_implemented(mode):
    with pytest.raises(NotImplementedError, match="protein_noise_model='totalvi'"):
        _model(protein_noise_model=mode)


def test_z_base_background_source_raises_not_implemented():
    model = _model()
    with pytest.raises(NotImplementedError, match="n_latent_protein_residual"):
        model.module.configure_protein_noise(protein_background_source="z_base")


# -- logging ------------------------------------------------------------------------------


def test_kl_w_and_active_units_are_logged_each_epoch():
    model = _fit(max_epochs=3)
    for split in ("train", "validation"):
        for name in ("kl_w", "protein_residual_active_units"):
            column = f"{name}_{split}"
            assert column in model.history, column
            assert len(model.history[column]) == 3
    assert (model.history["kl_w_train"].to_numpy(dtype=float) >= 0).all()
    units = model.history["protein_residual_active_units_train"].to_numpy(dtype=float)
    assert ((units >= 0) & (units <= N_W)).all()


def test_off_model_logs_no_residual_metrics():
    model = _fit(max_epochs=1, n_w=0)
    assert not any(key.startswith(("kl_w", "protein_residual")) for key in model.history)


# -- other decoder layouts and unsupported settings ----------------------------------------


@pytest.mark.parametrize("heads", ["fore_mixing", "all"])
@torch.no_grad()
def test_embedded_batch_layout_matches_between_factual_and_counterfactual_decodes(heads):
    model = _model(
        protein_residual_heads=heads,
        batch_representation="embedding",
        batch_embedding_kwargs={"embedding_dim": 2},
    )
    module, tensors, inference, _ = _outputs(model)
    gen = _generate(module, tensors, inference, torch.zeros_like(inference["w"]))
    det = module._deterministic_decoder_parameters(
        inference["z"], inference["library_gene"], tensors[REGISTRY_KEYS.BATCH_KEY]
    )
    expected = 1 - torch.sigmoid(gen["py_"]["mixing"])
    torch.testing.assert_close(det["protein_foreground_probability"], expected, rtol=1e-5, atol=1e-6)
    a, b = _perturbed(model)
    assert torch.equal(a["px_"]["scale"], b["px_"]["scale"])
    assert not torch.equal(a["py_"]["mixing"], b["py_"]["mixing"])


def test_z_base_background_source_raises_at_model_construction():
    with pytest.raises(NotImplementedError, match="n_latent_protein_residual"):
        _model(protein_background_source="z_base")


# -- encoder input: protein only, or protein plus donor (G121) -------------------------------


def _qw_mean(module, tensors, sample):
    inputs = module._get_inference_input(tensors)
    inputs["sample_index"] = torch.full_like(inputs["sample_index"], sample)
    with torch.no_grad():
        return module.inference(**inputs)["qw"].loc


def test_default_encoder_input_is_protein_and_ignores_the_sample():
    model = _model()
    assert model.init_params_["non_kwargs"]["protein_residual_encoder_input"] == "protein"
    module, tensors, _, _ = _outputs(model)
    assert torch.equal(_qw_mean(module, tensors, 0), _qw_mean(module, tensors, 1))


@pytest.mark.parametrize("path", ["legacy", "centered_v2"])
def test_protein_sample_encoder_input_depends_on_the_sample(path):
    model = _model(path, protein_residual_encoder_input="protein_sample")
    module, tensors, _, _ = _outputs(model)
    assert not torch.equal(_qw_mean(module, tensors, 0), _qw_mean(module, tensors, 1))


def test_protein_sample_round_trips_and_counterfactual_ignores_w(tmp_path):
    model = _model("centered_v2", protein_residual_encoder_input="protein_sample")
    assert model.init_params_["non_kwargs"]["protein_residual_encoder_input"] == "protein_sample"
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=model.adata)
    assert loaded.module.protein_residual_encoder_input == "protein_sample"
    for (k1, v1), (k2, v2) in zip(model.module.state_dict().items(), loaded.module.state_dict().items(), strict=True):
        assert k1 == k2
        assert torch.equal(v1.cpu(), v2.cpu())
    first = model.get_counterfactual_expression(model.adata)
    model.module.qw = _FixedQw(8.0)
    second = model.get_counterfactual_expression(model.adata)
    for key in first.data_vars:
        np.testing.assert_array_equal(first[key].to_numpy(), second[key].to_numpy())


def test_protein_sample_trains_and_logs():
    model = _fit(max_epochs=2, protein_residual_encoder_input="protein_sample")
    assert "kl_w_train" in model.history


def test_invalid_encoder_input_raises():
    with pytest.raises(ValueError, match="protein_residual_encoder_input"):
        _model(protein_residual_encoder_input="protein_rna")
