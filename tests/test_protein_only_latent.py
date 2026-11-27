"""Opt-in protein-only latent z_p (EXPERIMENTAL): encoder, prior, KL, heads, CF/DE decodes, saving."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import scvi
import torch
from scvi import REGISTRY_KEYS
from scvi.nn import DecoderTOTALVI
from torch.distributions import Normal

from mrtotalvi import MrTotalVI
from mrtotalvi._kl_free_bits import kl_with_free_bits
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import simulate_citeseq
from tests.test_mrtotalvi_v2 import _make_adata as _make_v2_adata

N_ZP = 4  # differs from n_latent=3 so a replace-mode shape mix-up fails loudly
N_Z = 3
PATHS = ["legacy", "centered_v2", "mc_samples", "scale_observations", "no_hierarchy"]
TRAIN = {"train_size": 0.7, "batch_size": 64, "accelerator": "cpu", "enable_progress_bar": False}

# State-dict sha256 after 3 seeded epochs of the default model, measured on origin/main (77f3643) with
# torch 2.15.0+cu132. Hashes are comparable only on the same torch build, so other builds skip.
MAIN_HASH_TORCH = "2.15.0+cu132"
MAIN_HASH = {
    "default": "64a45f246e7862756cf810fa85330ddab3b83310f27b80dab0ecc23c4e22e8f7",
    "legacy_no_map": "5f8d2dfe57d08cac117ab51ddcb9ca04bbddf1c6066d280cfd2972526c7aa88a",
}


def _model(path="legacy", n_zp=N_ZP, **kw):
    return MrTotalVI(
        _make_v2_adata(n_samples=3),
        sample_key="sample",
        n_latent=N_Z,
        n_latent_u=2,
        hierarchy_mode="centered_v2" if path == "centered_v2" else "legacy",
        scale_observations=path == "scale_observations",
        n_latent_protein_only=n_zp,
        **kw,
    )


def _outputs(model, path="legacy", n_cells=8):
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=n_cells)))
    kwargs = {"n_samples": 2} if path == "mc_samples" else {}
    inference, generative = module(tensors, inference_kwargs=kwargs, compute_loss=False)
    if path == "no_hierarchy":
        inference = {k: v for k, v in inference.items() if k not in ("eps", "eps_dist")}
    return module, tensors, inference, generative


def _generate(module, tensors, inference, zp):
    inputs = module._get_generative_input(tensors, inference)
    inputs["zp"] = zp
    return module.generative(**inputs)


def _fit(max_epochs=2, **kw):
    scvi.settings.seed = 0
    model = _model(**kw)
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    return model


def _digest(**kw):
    scvi.settings.seed = 0
    model = MrTotalVI(_make_v2_adata(n_samples=3), sample_key="sample", n_latent=N_Z, n_latent_u=2, **kw)
    model.train(max_epochs=3, check_val_every_n_epoch=1, early_stopping=False, **TRAIN)
    h = hashlib.sha256()
    for key, value in model.module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class _FixedQzp(torch.nn.Module):
    """Stands in for q(z_p | y): a unit-scale Normal at ``loc`` (a float or a (dim,) tensor)."""

    def __init__(self, loc, dim=N_ZP):
        super().__init__()
        self.loc, self.dim = loc, dim

    def forward(self, y, sample_index=None):
        mean = torch.as_tensor(self.loc, dtype=torch.float32).expand(y.shape[0], self.dim)
        return Normal(mean, torch.ones(y.shape[0], self.dim))


# -- default off ---------------------------------------------------------------------------


@pytest.mark.skipif(torch.__version__ != MAIN_HASH_TORCH, reason="hash recorded on another torch build")
def test_default_state_dict_hash_matches_main_after_three_epochs():
    assert _digest() == MAIN_HASH["default"]
    assert _digest(use_map=False, hierarchy_mode="legacy") == MAIN_HASH["legacy_no_map"]


def test_default_is_off_and_builds_no_extra_modules():
    model = MrTotalVI(_make_v2_adata(n_samples=3), sample_key="sample", n_latent=N_Z, n_latent_u=2)
    assert model.init_params_["non_kwargs"]["n_latent_protein_only"] == 0
    assert model.module.n_latent_protein_only == 0
    assert type(model.module.decoder) is DecoderTOTALVI
    assert not any(key.startswith(("qzp", "pzp")) for key in model.module.state_dict())
    inference = _outputs(model)[2]
    assert "zp" not in inference and "qzp" not in inference


@pytest.mark.parametrize("path", PATHS)
def test_off_loss_has_no_kl_zp(path):
    model = _model(path, n_zp=0)
    module, tensors, inference, generative = _outputs(model, path)
    out = module.loss(tensors, inference, generative, kl_weight=0.7)
    assert "kl_div_zp" not in out.kl_local
    assert "kl_zp" not in out.extra_metrics


# -- KL of z_p, KL weight and free bits ----------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kl_weight", [1.0, 0.7])
def test_kl_of_shifted_posterior_is_exactly_half_n_dim_times_shift_squared(path, kl_weight):
    model = _model(path)
    module, tensors, inference, generative = _outputs(model, path)
    y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

    def loss_with(loc):
        patched = {**inference, "qzp": _FixedQzp(loc)(y)}
        return module.loss(tensors, patched, generative, kl_weight=kl_weight)

    base, shifted = loss_with(0.0), loss_with(0.5)
    expected = 0.5 * N_ZP * 0.25  # 0.5 nats per cell
    np.testing.assert_allclose(base.kl_local["kl_div_zp"].detach().numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(shifted.kl_local["kl_div_zp"].detach().numpy(), expected, rtol=1e-6)
    per_cell = expected
    if path == "scale_observations":
        counts = module.n_obs_per_sample[tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()]
        per_cell = (expected / counts).mean().item()
    assert (shifted.loss - base.loss).item() == pytest.approx(kl_weight * per_cell, rel=1e-4)


def test_kl_weight_scales_the_loss_term_but_not_the_logged_kl():
    model = _model()
    module, tensors, inference, generative = _outputs(model)
    patched = {**inference, "qzp": _FixedQzp(0.5)(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])}
    one = module.loss(tensors, patched, generative, kl_weight=1.0)
    module.protein_latent_kl_weight = 2.0
    two = module.loss(tensors, patched, generative, kl_weight=1.0)
    assert (two.loss - one.loss).item() == pytest.approx(0.5, rel=1e-5)  # one extra copy of 0.5 nats
    torch.testing.assert_close(two.kl_local["kl_div_zp"], one.kl_local["kl_div_zp"])


def test_free_bits_hand_arithmetic_and_gradient():
    # Two dims with KL_d = 0.1 and 0.5 (q = N(mu, 1) so KL_d = mu^2 / 2) and a floor of 0.25 nats.
    model = _model(n_zp=2)
    module, tensors, inference, generative = _outputs(model)
    y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
    mu = torch.tensor([0.2**0.5, 1.0], requires_grad=True)

    def loss_with(free_bits):
        module.protein_latent_free_bits = free_bits
        patched = {**inference, "qzp": _FixedQzp(mu, dim=2)(y)}
        return module.loss(tensors, patched, generative, kl_weight=1.0)

    plain = loss_with(0.0)
    floored = loss_with(0.25)
    np.testing.assert_allclose(floored.kl_local["kl_div_zp"].detach().numpy(), 0.6, rtol=1e-5)  # true KL is logged
    assert (floored.loss - plain.loss).item() == pytest.approx(0.75 - 0.6, rel=1e-4)  # 0.25 + 0.5 versus 0.1 + 0.5
    (grad,) = torch.autograd.grad(floored.loss, mu)
    np.testing.assert_allclose(grad.numpy(), [0.0, 1.0], atol=1e-6)  # the floored dim has no gradient


def test_free_bits_helper_statistics_and_granularity():
    # Design check 4: half the cells at KL 0.1 and half at 0.5, floor 0.25, one dimension.
    kl = torch.tensor([[0.1], [0.1], [0.5], [0.5]])
    assert kl_with_free_bits(kl, 0.25, "per_dim", "batch_mean").mean().item() == pytest.approx(0.3)
    assert kl_with_free_bits(kl, 0.25, "per_dim", "per_cell").mean().item() == pytest.approx(0.375)
    # Floor of 0 leaves the plain sum, bit for bit.
    two = torch.rand(5, 3)
    assert torch.equal(kl_with_free_bits(two, 0.0, "per_dim", "batch_mean"), two.sum(-1))
    # Two dims of 0.1: per dimension both are lifted to 0.25; as one group the sum 0.2 is lifted to 0.25.
    pair = torch.full((3, 2), 0.1)
    assert kl_with_free_bits(pair, 0.25, "per_dim", "batch_mean").mean().item() == pytest.approx(0.5)
    assert kl_with_free_bits(pair, 0.25, "group", "batch_mean").mean().item() == pytest.approx(0.25)
    # A mean above the floor keeps its own per-cell values, so the batch mean is untouched.
    assert kl_with_free_bits(kl, 0.25, "group", "batch_mean").tolist() == pytest.approx([0.1, 0.1, 0.5, 0.5])


def test_free_bits_schedule_after_warmup_applies_only_at_full_kl_weight():
    model = _model(n_zp=2, protein_latent_free_bits=0.25, protein_latent_free_bits_schedule="after_warmup")
    module, tensors, inference, generative = _outputs(model)
    patched = {**inference, "qzp": _FixedQzp(0.0, dim=2)(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])}  # KL = 0
    early = module.loss(tensors, patched, generative, kl_weight=0.5)
    module.protein_latent_free_bits = 0.0
    assert early.loss.item() == pytest.approx(module.loss(tensors, patched, generative, kl_weight=0.5).loss.item())
    module.protein_latent_free_bits = 0.25
    late = module.loss(tensors, patched, generative, kl_weight=1.0)
    module.protein_latent_free_bits = 0.0
    assert (late.loss - module.loss(tensors, patched, generative, kl_weight=1.0).loss).item() == pytest.approx(0.5)


# -- decoder: replace versus concat, heads, RNA isolation -----------------------------------


def _inference_pair(model):
    """Inference on x and on a strongly perturbed x with the same y, in eval mode."""
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
    perturbed = dict(tensors)
    perturbed[REGISTRY_KEYS.X_KEY] = tensors[REGISTRY_KEYS.X_KEY] * 3 + 5
    return (
        module,
        tensors,
        module.inference(**module._get_inference_input(tensors)),
        perturbed,
        module.inference(**module._get_inference_input(perturbed)),
    )


@torch.no_grad()
def test_replace_mode_protein_heads_ignore_rna_input_given_fixed_zp():
    model = _model(protein_latent_combination="replace", protein_latent_heads="all")
    module, tensors, inf_a, perturbed, inf_b = _inference_pair(model)
    # q(z_p | y) reads y only, so perturbing x leaves it identical while z moves.
    assert torch.equal(inf_a["qzp"].loc, inf_b["qzp"].loc)
    assert not torch.equal(inf_a["z"], inf_b["z"])
    zp = torch.randn(8, N_ZP)
    a = _generate(module, tensors, inf_a, zp)["py_"]
    b = _generate(module, perturbed, inf_b, zp)["py_"]
    assert (a["mixing"] - b["mixing"]).abs().max().item() == 0.0
    assert (a["fore_scale"] - b["fore_scale"]).abs().max().item() == 0.0
    # G181: in replace mode the background head reads z_p too.
    assert (a["back_alpha"] - b["back_alpha"]).abs().max().item() == 0.0
    assert (a["back_beta"] - b["back_beta"]).abs().max().item() == 0.0


def test_replace_mode_refuses_heads_that_leave_the_background_on_z():
    with pytest.raises(ValueError, match="replace"):
        _model(protein_latent_combination="replace", protein_latent_heads="fore_mixing")
    _model(protein_latent_combination="concat", protein_latent_heads="fore_mixing")
    assert _model().module.protein_latent_heads == "all"  # the signature default is valid with replace


@pytest.mark.parametrize("heads", ["fore_mixing", "all"])
@torch.no_grad()
def test_concat_mode_protein_heads_change_with_rna_input(heads):
    model = _model(protein_latent_combination="concat", protein_latent_heads=heads)
    module, tensors, inf_a, perturbed, inf_b = _inference_pair(model)
    zp = torch.randn(8, N_ZP)
    a = _generate(module, tensors, inf_a, zp)["py_"]
    b = _generate(module, perturbed, inf_b, zp)["py_"]
    assert not torch.equal(a["mixing"], b["mixing"])
    assert not torch.equal(a["fore_scale"], b["fore_scale"])


@pytest.mark.parametrize("combination", ["replace", "concat"])
@torch.no_grad()
def test_perturbing_zp_leaves_rna_scale_exactly_unchanged(combination):
    model = _model(protein_latent_combination=combination, protein_latent_heads="all")
    module, tensors, inference, _ = _outputs(model)
    torch.manual_seed(0)
    a = _generate(module, tensors, inference, torch.zeros(8, N_ZP))
    b = _generate(module, tensors, inference, torch.randn(8, N_ZP) * 3.0)
    assert (a["px_"]["scale"] - b["px_"]["scale"]).abs().max().item() == 0.0
    assert torch.equal(a["px_"]["rate"], b["px_"]["rate"])
    assert not torch.equal(a["py_"]["mixing"], b["py_"]["mixing"])  # the protein side does move


@torch.no_grad()
def test_heads_knob_is_honoured_in_concat_mode():
    combination = "concat"

    def pair(heads):
        module, tensors, inference, _ = _outputs(
            _model(protein_latent_combination=combination, protein_latent_heads=heads)
        )
        torch.manual_seed(0)
        return (
            _generate(module, tensors, inference, torch.zeros(8, N_ZP))["py_"],
            _generate(module, tensors, inference, torch.randn(8, N_ZP) * 3.0)["py_"],
        )

    a, b = pair("fore_mixing")
    assert torch.equal(a["back_alpha"], b["back_alpha"]) and torch.equal(a["back_beta"], b["back_beta"])
    assert not torch.equal(a["fore_scale"], b["fore_scale"]) and not torch.equal(a["mixing"], b["mixing"])
    a, b = pair("all")
    assert not torch.equal(a["back_alpha"], b["back_alpha"])
    assert not torch.equal(a["back_beta"], b["back_beta"])


def test_zinb_leaks_zp_into_rna_dropout_but_not_into_rna_scale():
    # G93: the mixing trunk (sigmoid_decoder) is shared with the RNA dropout head. Documented, not blocked.
    module, tensors, inference, _ = _outputs(_model(gene_likelihood="zinb"))
    a = _generate(module, tensors, inference, torch.zeros(8, N_ZP))
    b = _generate(module, tensors, inference, torch.ones(8, N_ZP) * 3.0)
    assert torch.equal(a["px_"]["scale"], b["px_"]["scale"])
    assert not torch.equal(a["px_"]["dropout"], b["px_"]["dropout"])


def test_replace_protein_reconstruction_has_no_gradient_into_the_u_encoder():
    # All three heads read z_p, so the protein term must not reach q(u); only the RNA term does.
    model = _model(protein_latent_heads="all")
    module, tensors, inference, generative = _outputs(model)
    module.train()
    inference = module.inference(**module._get_inference_input(tensors))
    generative = module.generative(**module._get_generative_input(tensors, inference))
    loss = module.loss(tensors, inference, generative)
    protein = loss.reconstruction_loss["reconst_loss_protein"].mean()
    rna = loss.reconstruction_loss["reconst_loss_gene"].mean()
    qu_params = list(module.qu.parameters())
    grads = torch.autograd.grad(protein, qu_params, retain_graph=True, allow_unused=True)
    assert all(g is None or float(g.abs().max()) == 0.0 for g in grads)
    rna_grads = torch.autograd.grad(rna, qu_params, retain_graph=True, allow_unused=True)
    assert any(g is not None and float(g.abs().max()) > 0.0 for g in rna_grads)


# -- counterfactual and DE decodes ----------------------------------------------------------


def test_counterfactual_protein_decode_is_invariant_to_donor_in_replace_mode():
    model = _model("centered_v2", protein_latent_combination="replace", protein_latent_heads="all")
    model.is_trained_ = True
    out = model.get_counterfactual_expression(model.adata)
    assert out.sizes["target_sample"] == 3
    for key in out.data_vars:
        if key.startswith("protein_") and key != "protein_available":
            values = out[key].to_numpy()
            # float32 rounding only: CI BLAS differs by 1 ulp (rel 1e-7) across row batches; a donor effect is >1e-2.
            np.testing.assert_allclose(values, np.broadcast_to(values[:, :, :1], values.shape), rtol=1e-5)
    rna = out["rna_scale"].to_numpy()
    assert not np.array_equal(rna, np.broadcast_to(rna[:, :, :1], rna.shape))  # donors still move RNA


def test_counterfactual_protein_decode_reads_the_cells_own_zp():
    model = _model("centered_v2", protein_latent_combination="replace", protein_latent_heads="all")
    model.is_trained_ = True
    first = model.get_counterfactual_expression(model.adata)
    model.module.qzp = _FixedQzp(8.0)
    second = model.get_counterfactual_expression(model.adata)
    assert not np.array_equal(first["protein_total_mean"].to_numpy(), second["protein_total_mean"].to_numpy())
    np.testing.assert_array_equal(first["rna_scale"].to_numpy(), second["rna_scale"].to_numpy())


def test_counterfactual_in_concat_mode_still_moves_protein_with_donor():
    model = _model("centered_v2", protein_latent_combination="concat", protein_latent_heads="all")
    model.is_trained_ = True
    values = model.get_counterfactual_expression(model.adata)["protein_total_mean"].to_numpy()
    assert not np.array_equal(values, np.broadcast_to(values[:, :, :1], values.shape))


def _hook_kwargs(module, tensors, inference, eps):
    return {
        "x": tensors[REGISTRY_KEYS.X_KEY],
        "y": tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY],
        "sample_index": tensors[REGISTRY_KEYS.SAMPLE_KEY],
        "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
        "extra_eps": eps,
        "u_anchor": inference["u"],
        "_lfc_aux": {"library_gene": inference["library_gene"]},
    }


@torch.no_grad()
def test_de_hook_protein_columns_ignore_the_donor_shift_in_replace_mode():
    model = _model(protein_latent_combination="replace", protein_latent_heads="all")
    module, tensors, inference, _ = _outputs(model)
    zero = module.compute_h_from_x_eps(**_hook_kwargs(module, tensors, inference, torch.zeros(8, N_Z)))
    shifted = module.compute_h_from_x_eps(**_hook_kwargs(module, tensors, inference, torch.randn(8, N_Z) * 3))
    n_genes = tensors[REGISTRY_KEYS.X_KEY].shape[1]
    assert torch.equal(zero[:, n_genes:], shifted[:, n_genes:])
    assert not torch.equal(zero[:, :n_genes], shifted[:, :n_genes])


@torch.no_grad()
def test_deterministic_decode_equals_generative_at_the_posterior_mean_zp():
    model = _model(protein_latent_heads="all")
    module, tensors, inference, _ = _outputs(model)
    zp = inference["qzp"].loc
    gen = _generate(module, tensors, inference, zp)["py_"]
    det = module._deterministic_decoder_parameters(
        inference["z"], inference["library_gene"], tensors[REGISTRY_KEYS.BATCH_KEY], zp=zp
    )
    np.testing.assert_allclose(
        det["protein_foreground_probability"].numpy(), (1 - torch.sigmoid(gen["mixing"])).numpy(), rtol=1e-5
    )
    with pytest.raises(RuntimeError, match="z_p"):
        module._deterministic_decoder_parameters(
            inference["z"], inference["library_gene"], tensors[REGISTRY_KEYS.BATCH_KEY]
        )


# -- factual readouts and the decode-draw knob ---------------------------------------------


def _foreground_probability(module, tensors, seed):
    torch.manual_seed(seed)
    return module.protein_components(tensors, n_samples=1)["foreground_probability"]


def test_decode_draw_mean_makes_the_readout_independent_of_the_zp_draw():
    kw = {"protein_latent_combination": "replace", "protein_latent_heads": "all"}
    draw_model = _model(**kw, protein_latent_decode_draw="draw")
    tensors = next(iter(draw_model._make_data_loader(adata=draw_model.adata, batch_size=8)))
    draw = draw_model.module.eval()
    mean = _model(**kw, protein_latent_decode_draw="mean").module.eval()
    assert not torch.equal(_foreground_probability(draw, tensors, 0), _foreground_probability(draw, tensors, 1))
    assert torch.equal(_foreground_probability(mean, tensors, 0), _foreground_probability(mean, tensors, 1))


def test_decode_draw_mean_reaches_the_scvi_readouts_and_leaves_the_loss_alone():
    kw = {"protein_latent_combination": "replace", "protein_latent_heads": "all"}
    scvi.settings.seed = 0
    draw = _model(**kw, protein_latent_decode_draw="draw")
    scvi.settings.seed = 0
    mean = _model(**kw, protein_latent_decode_draw="mean")
    for model in (draw, mean):
        model.is_trained_ = True
    probabilities = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        probabilities.append(mean.get_protein_foreground_probability(n_samples=1))
    np.testing.assert_array_equal(np.asarray(probabilities[0]), np.asarray(probabilities[1]))
    # The validation loss is computed in eval mode too; it must keep drawing z_p.
    losses = []
    for model in (draw, mean):
        module = model.module.eval()
        tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
        torch.manual_seed(3)
        losses.append(module(tensors)[2].loss.item())  # forward with compute_loss=True, as validation does
    assert losses[0] == pytest.approx(losses[1])


# -- encoder: input and conditioning --------------------------------------------------------


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


def _sim_model(sim, **kw):
    adata, profile = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", n_latent=4, n_latent_protein_only=5, ambient_profile=profile, **kw)


@torch.no_grad()
def test_ambient_zscore_input_feeds_the_dsb_zscore_to_q_zp_only(sim):
    model = _sim_model(sim, protein_latent_encoder_input="ambient_zscore")
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=16)))
    y, batch = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY], tensors[REGISTRY_KEYS.BATCH_KEY]
    zscored = (torch.log(y + 10.0) - module.dsb_log_mean[0]) / module.dsb_log_sd[0]
    inference = module.inference(**module._get_inference_input(tensors))
    torch.testing.assert_close(inference["qzp"].loc, module.qzp(zscored, None).loc)
    assert module.protein_encoder_input == "raw"  # q(u) keeps the raw counts
    assert not module.qu.protein_input_pretransformed
    log1p = _sim_model(sim).module.eval()
    assert not torch.allclose(log1p.qzp(torch.log1p(y), None).loc, module.qzp(zscored, None).loc)
    assert batch.shape[0] == 16


def test_ambient_zscore_input_needs_an_ambient_profile(sim):
    adata, _ = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    with pytest.raises(ValueError, match="ambient_profile"):
        MrTotalVI(
            adata,
            sample_key="donor",
            n_latent=4,
            n_latent_protein_only=5,
            protein_latent_encoder_input="ambient_zscore",
        )


@torch.no_grad()
def test_donor_conditioning_changes_q_zp_with_the_donor_and_blind_does_not():
    blind = _model(protein_latent_conditioning="blind").module.eval()
    donor = _model(protein_latent_conditioning="donor").module.eval()
    y = torch.rand(6, 3) * 20
    s0, s1 = torch.zeros(6, 1, dtype=torch.long), torch.ones(6, 1, dtype=torch.long)
    assert torch.equal(blind.qzp(y, s0).loc, blind.qzp(y, s1).loc)
    assert not torch.equal(donor.qzp(y, s0).loc, donor.qzp(y, s1).loc)


@torch.no_grad()
def test_counterfactual_donor_never_reaches_q_zp():
    model = _model(protein_latent_conditioning="donor")
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=model.adata, batch_size=8)))
    inputs = module._get_inference_input(tensors)
    own = module.inference(**inputs)["qzp"].loc
    other = module._regular_inference(**inputs, cf_sample=(inputs["sample_index"] + 1) % 3, n_samples=1)["qzp"].loc
    assert torch.equal(own, other)


# -- prior: N(0, I) and MoG -----------------------------------------------------------------


def test_mog_prior_has_twenty_components_and_trains():
    model = _fit(protein_latent_prior="mog")
    assert model.module.pzp.u_prior_means.shape == (20, N_ZP)
    for name in ("kl_zp_mc_estimate_train", "kl_zp_mc_estimate_validation"):
        assert np.isfinite(model.history[name].to_numpy(dtype=float)).all()
    assert "kl_zp_train" not in model.history  # the MoG KL is a Monte Carlo estimate and is labelled as one
    module, tensors, inference, generative = _outputs(model)
    module.train()
    inference = module.inference(**module._get_inference_input(tensors))
    generative = module.generative(**module._get_generative_input(tensors, inference))
    module.loss(tensors, inference, generative).loss.backward()
    assert float(module.pzp.u_prior_means.grad.abs().sum()) > 0.0


@torch.no_grad()
def test_mog_with_one_live_component_matches_the_closed_form_kl():
    model = _model(protein_latent_prior="mog")
    module, tensors, inference, _ = _outputs(model)
    prior = module.pzp
    prior.u_prior_means.zero_()
    prior.u_prior_scales.zero_()  # scale = exp(0) = 1
    prior.u_prior_logits.fill_(-1e4)
    prior.u_prior_logits[0] = 0.0  # one live component: N(0, I)
    qzp = Normal(torch.full((8, N_ZP), 0.5), torch.ones(8, N_ZP))
    draws = qzp.rsample((20000,))
    kl = module._kl_zp({**inference, "qzp": qzp, "zp": draws})
    np.testing.assert_allclose(kl.numpy(), 0.5, rtol=0.05)  # 0.5 * 4 * 0.25


# -- consistency KL(q(z | x, y) || q(z_p | y)) ----------------------------------------------


def _consistency_model(**kw):
    return _model(n_zp=N_Z, use_map=False, protein_latent_consistency_weight=kw.pop("weight", 0.5), **kw)


def test_consistency_kl_matches_the_closed_form_on_a_hand_example():
    # q(z | u) = N(1, 2^2) and q(z_p | y) = N(0, 1) in 3 dims: 3 * 0.5 * (1 + 4 - 1 - ln 4) = 3.9206 nats.
    model = _consistency_model()
    module, tensors, inference, generative = _outputs(model)
    y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
    eps_dist = Normal(1.0 - inference["z_base"], torch.full_like(inference["z_base"], 2.0))
    patched = {**inference, "eps_dist": eps_dist, "qzp": _FixedQzp(0.0, dim=N_Z)(y)}
    expected = 3 * 0.5 * (1 + 4 - 1 - np.log(4))
    np.testing.assert_allclose(module._consistency_kl(patched).detach().numpy(), expected, rtol=1e-5)
    with_weight = module.loss(tensors, patched, generative, kl_weight=0.3)
    module.protein_latent_consistency_weight = 0.0
    without = module.loss(tensors, patched, generative, kl_weight=0.3)
    # G180: the weight is annealed with kl_weight like the other KL terms.
    assert (with_weight.loss - without.loss).item() == pytest.approx(0.3 * 0.5 * expected, rel=1e-4)
    # A2: the consistency KL is a training term and a logged metric only; it never enters kl_local (ELBO).
    assert "kl_div_zp_consistency" not in with_weight.kl_local
    assert with_weight.extra_metrics["kl_zp_consistency"].item() == pytest.approx(expected, rel=1e-5)


def test_consistency_weight_leaves_the_reported_elbo_terms_unchanged():
    # Codex A2: enabling the weight must not change validation ELBO (sum of kl_local) or early stopping.
    model = _consistency_model()
    module, tensors, inference, generative = _outputs(model)
    on = module.loss(tensors, inference, generative, kl_weight=1.0)
    module.protein_latent_consistency_weight = 0.0
    off = module.loss(tensors, inference, generative, kl_weight=1.0)
    assert on.kl_local.keys() == off.kl_local.keys()
    for key in off.kl_local:
        assert torch.equal(on.kl_local[key], off.kl_local[key]), key
    assert on.extra_metrics["kl_zp_consistency"].item() > 0
    assert "kl_zp_consistency" not in off.extra_metrics


def test_consistency_off_by_default_logs_nothing():
    module, tensors, inference, generative = _outputs(_model())
    out = module.loss(tensors, inference, generative)
    assert "kl_div_zp_consistency" not in out.kl_local
    assert "kl_zp_consistency" not in out.extra_metrics


def test_consistency_with_mismatched_dimensions_stops():
    with pytest.raises(ValueError, match="n_latent"):
        _model(n_zp=N_Z + 1, use_map=False, protein_latent_consistency_weight=0.1)


@pytest.mark.parametrize("kw", [{"use_map": True}, {"use_map": True, "hierarchy_mode": "centered_v2"}])
def test_consistency_needs_an_analytic_q_z(kw):
    with pytest.raises((ValueError, NotImplementedError), match="consistency"):
        MrTotalVI(
            _make_v2_adata(3),
            sample_key="sample",
            n_latent=N_Z,
            n_latent_u=2,
            n_latent_protein_only=N_Z,
            protein_latent_consistency_weight=0.1,
            **kw,
        )


# -- logging --------------------------------------------------------------------------------


def test_kl_zp_active_units_and_per_dimension_kl_are_logged_each_epoch():
    model = _fit(max_epochs=3)
    for split in ("train", "validation"):
        for name in ("kl_zp", "protein_latent_active_units", *(f"kl_zp_dim_{d:02d}" for d in range(N_ZP))):
            column = f"{name}_{split}"
            assert column in model.history, column
            assert len(model.history[column]) == 3
    assert (model.history["kl_zp_train"].to_numpy(dtype=float) >= 0).all()
    dims = np.stack([model.history[f"kl_zp_dim_{d:02d}_train"].to_numpy(dtype=float) for d in range(N_ZP)])
    np.testing.assert_allclose(dims.sum(0), model.history["kl_zp_train"].to_numpy(dtype=float), rtol=1e-4)
    units = model.history["protein_latent_active_units_train"].to_numpy(dtype=float)
    assert ((units >= 0) & (units <= N_ZP)).all()


def test_off_model_logs_no_protein_latent_metrics():
    model = _fit(max_epochs=1, n_zp=0)
    assert not any(key.startswith(("kl_zp", "protein_latent")) for key in model.history)


# -- saving and validation ------------------------------------------------------------------

KNOBS = {
    "n_latent_protein_only": N_ZP,
    "protein_latent_combination": "concat",
    "protein_latent_heads": "all",
    "protein_latent_prior": "mog",
    "protein_latent_encoder_input": "log1p",
    "protein_latent_conditioning": "donor",
    "protein_latent_kl_weight": 0.5,
    "protein_latent_free_bits": 0.125,
    "protein_latent_free_bits_granularity": "group",
    "protein_latent_free_bits_statistic": "per_cell",
    "protein_latent_free_bits_schedule": "after_warmup",
    "protein_latent_mog_kl_draws": 4,
    "protein_latent_decode_draw": "mean",
}


def test_init_params_round_trip(tmp_path):
    model = _model(**{k: v for k, v in KNOBS.items() if k != "n_latent_protein_only"})
    for key, value in KNOBS.items():
        assert model.init_params_["non_kwargs"][key] == value
    model.is_trained_ = True
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=model.adata)
    for key, value in KNOBS.items():
        assert loaded.init_params_["non_kwargs"][key] == value
        assert getattr(loaded.module, key) == value
    assert loaded.init_params_["non_kwargs"]["protein_latent_consistency_weight"] == 0.0
    for (k1, v1), (k2, v2) in zip(model.module.state_dict().items(), loaded.module.state_dict().items(), strict=True):
        assert k1 == k2
        assert torch.equal(v1.cpu(), v2.cpu())


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_invalid_size_raises(bad):
    with pytest.raises((ValueError, TypeError), match="n_latent_protein_only"):
        _model(n_zp=bad)


@pytest.mark.parametrize(
    "kw",
    [
        {"protein_latent_combination": "mix"},
        {"protein_latent_heads": "background"},
        {"protein_latent_prior": "vamp"},
        {"protein_latent_encoder_input": "raw"},
        {"protein_latent_conditioning": "batch"},
        {"protein_latent_free_bits_granularity": "row"},
        {"protein_latent_free_bits_statistic": "median"},
        {"protein_latent_free_bits_schedule": "ramp"},
        {"protein_latent_decode_draw": "sample"},
        {"protein_latent_kl_weight": -1.0},
        {"protein_latent_free_bits": -0.1},
        {"protein_latent_consistency_weight": -0.1},
    ],
)
def test_invalid_options_raise(kw):
    with pytest.raises(ValueError, match="protein_latent"):
        _model(**kw)


def test_mog_prior_with_per_dimension_free_bits_is_not_implemented():
    with pytest.raises(NotImplementedError, match="free_bits"):
        _model(protein_latent_prior="mog", protein_latent_free_bits=0.25)
    _model(
        protein_latent_prior="mog",
        protein_latent_free_bits=0.25,
        protein_latent_free_bits_granularity="group",
        protein_latent_mog_kl_draws=4,
    )


def test_mog_free_bits_needs_an_explicit_number_of_kl_draws():
    # No package default: the user picks how many draws are averaged before the floor.
    kw = {
        "protein_latent_prior": "mog",
        "protein_latent_free_bits": 0.25,
        "protein_latent_free_bits_granularity": "group",
    }
    with pytest.raises(ValueError, match="protein_latent_mog_kl_draws"):
        _model(**kw)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="protein_latent_mog_kl_draws"):
            _model(**kw, protein_latent_mog_kl_draws=bad)
    _model(protein_latent_prior="mog")  # without free bits the option is not needed


@torch.no_grad()
def test_mog_free_bits_average_draws_before_clamping():
    # Codex A3: identical N(0, 1) components, q = N(0.2, 1), true KL 0.02, floor 0.125. One draw clamps a
    # noisy estimate (about 0.163 per cell); many draws average the noise out before the clamp (about 0.125).
    def mean_cost(draws):
        model = _model(
            n_zp=1,
            protein_latent_prior="mog",
            protein_latent_free_bits=0.125,
            protein_latent_free_bits_granularity="group",
            protein_latent_free_bits_statistic="per_cell",
            protein_latent_mog_kl_draws=draws,
        )
        module = model.module.eval()
        module.pzp.u_prior_means.zero_()
        module.pzp.u_prior_scales.zero_()
        qzp = Normal(torch.full((4000, 1), 0.2), torch.ones(4000, 1))
        torch.manual_seed(0)
        term, _, _ = module._zp_loss({"qzp": qzp, "zp": qzp.sample()}, kl_weight=1.0)
        return term.mean().item()

    assert mean_cost(1) > 0.15
    assert mean_cost(256) < 0.135


@pytest.mark.parametrize("mode", ["ambient", "ambient_isotype"])
def test_ambient_modes_raise_not_implemented(mode):
    with pytest.raises(NotImplementedError, match="protein_noise_model='totalvi'"):
        _model(protein_noise_model=mode)


def test_residual_latent_w_is_not_combined_with_z_p():
    with pytest.raises(NotImplementedError, match="n_latent_protein_residual"):
        _model(n_latent_protein_residual=4)


def test_u_invariance_is_not_combined_with_z_p():
    with pytest.raises(NotImplementedError, match="u_invariance"):
        _model(u_invariance_weight=0.1)


def test_background_source_z_base_needs_a_background_head_that_ignores_z_p():
    model = _model(protein_latent_combination="concat", protein_latent_heads="fore_mixing")
    model.module.configure_protein_noise(protein_background_source="z_base")
    assert model.module.protein_background_source == "z_base"
    with pytest.raises(ValueError, match="protein_background_source"):
        _model(protein_latent_heads="all").module.configure_protein_noise(protein_background_source="z_base")


@torch.no_grad()
def test_z_base_background_ignores_z_p_and_the_donor():
    model = _model(protein_latent_combination="concat", protein_latent_heads="fore_mixing")
    model.module.configure_protein_noise(protein_background_source="z_base")
    module, tensors, inference, _ = _outputs(model)
    torch.manual_seed(0)
    a = _generate(module, tensors, inference, torch.zeros(8, N_ZP))["py_"]
    b = _generate(module, tensors, inference, torch.randn(8, N_ZP) * 3.0)["py_"]
    assert torch.equal(a["back_alpha"], b["back_alpha"])
    assert not torch.equal(a["mixing"], b["mixing"])


# -- precomputed encoder input (for example full dsb) ----------------------------------------


def _precomputed_adata(mutate=None):
    import pandas as pd

    adata = scvi.data.synthetic_iid(batch_size=4, n_genes=5, n_proteins=3, n_regions=2, n_batches=2, n_labels=2)
    adata.obs["sample"] = np.asarray([f"sample_{index % 3}" for index in range(adata.n_obs)])
    names = list(adata.uns["protein_names"])
    rng = np.random.default_rng(0)
    adata.obsm["dsb"] = pd.DataFrame(rng.normal(size=(adata.n_obs, len(names))), index=adata.obs_names, columns=names)
    if mutate is not None:
        mutate(adata)
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        protein_names_uns_key="protein_names",
        protein_latent_input_obsm_key="dsb",
    )
    return adata


class _Recorder(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen = inner, []

    def forward(self, y, sample_index=None):
        self.seen.append(y.detach().clone())
        return self.inner(y, sample_index)


@torch.no_grad()
def test_precomputed_matrix_reaches_the_encoder_exactly_per_minibatch():
    adata = _precomputed_adata()
    model = MrTotalVI(
        adata, sample_key="sample", n_latent=N_Z, n_latent_u=2, n_latent_protein_only=N_ZP,
        protein_latent_encoder_input="precomputed",
    )  # fmt: skip
    module = model.module.eval()
    module.qzp = recorder = _Recorder(module.qzp)
    assert module.qzp.inner.log1p_input is False
    for tensors in model._make_data_loader(adata=adata, batch_size=16):
        rows = tensors[REGISTRY_KEYS.INDICES_KEY].long().flatten().numpy()
        module.inference(**module._get_inference_input(tensors))
        expected = torch.as_tensor(adata.obsm["dsb"].to_numpy()[rows], dtype=torch.float32)
        assert torch.equal(recorder.seen[-1], expected)
    assert not recorder.seen[-1].requires_grad


def test_precomputed_input_needs_the_registered_matrix():
    adata = _make_v2_adata(3)
    with pytest.raises(ValueError, match="protein_latent_input_obsm_key"):
        MrTotalVI(
            adata, sample_key="sample", n_latent=N_Z, n_latent_protein_only=N_ZP,
            protein_latent_encoder_input="precomputed",
        )  # fmt: skip


def test_precomputed_matrix_with_wrong_protein_order_raises():
    def reorder(adata):
        adata.obsm["dsb"] = adata.obsm["dsb"].iloc[:, ::-1]

    with pytest.raises(ValueError, match="names and order"):
        _precomputed_adata(reorder)


def test_precomputed_matrix_with_nan_or_wrong_shape_raises():
    def add_nan(adata):
        adata.obsm["dsb"].iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        _precomputed_adata(add_nan)

    def drop_column(adata):
        adata.obsm["dsb"] = adata.obsm["dsb"].iloc[:, :2]

    with pytest.raises(ValueError, match="shape"):
        _precomputed_adata(drop_column)


def test_precomputed_matrix_reaches_the_private_lfc_path():
    # Codex A4: the private LFC path dropped protein_latent_input in both inference calls and raised.
    def add_condition(adata):
        adata.obs["condition"] = np.where(adata.obs["sample"] == "sample_0", "a", "b")

    adata = _precomputed_adata(add_condition)
    model = MrTotalVI(
        adata, sample_key="sample", n_latent=N_Z, n_latent_u=2, n_latent_protein_only=N_ZP,
        protein_latent_encoder_input="precomputed",
    )  # fmt: skip
    model.train(max_epochs=1, **TRAIN)
    de = model._legacy_differential_expression_for_reproducibility(
        sample_cov_keys=["condition"], mc_samples=2, batch_size=32, store_lfc=True, delta=None
    )
    assert np.all(np.isfinite(de["lfc"].values))
    # The full-inference fallback (no cached aux) takes the matrix too.
    module = model.module.eval()
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=8)))
    inputs = module._get_inference_input(tensors)
    inputs = {k: v for k, v in inputs.items() if k != "n_samples"}
    zeros = torch.zeros(8, N_Z)
    with pytest.warns(UserWarning):
        out = module.compute_h_from_x_eps(extra_eps=zeros, **inputs)
    assert torch.isfinite(out).all()
    assert torch.isfinite(module._infer_lfc_aux(**inputs)["library_gene"]).all()
