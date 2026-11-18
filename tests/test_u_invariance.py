"""Opt-in counterfactual u-invariance penalty (u_invariance_weight)."""

from __future__ import annotations

import numpy as np
import pytest
import scvi
import torch

from mrtotalvi import MrTotalVI

SETUP = dict(protein_expression_obsm_key="protein_expression", sample_key="sample", batch_key="batch")
BLIND = dict(u_encoder_mode="sample_blind")


def _model(**kw):
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = [f"s{i % 4}" for i in range(adata.n_obs)]
    MrTotalVI.setup_anndata(adata, **SETUP)
    model = MrTotalVI(adata, sample_key="sample", n_latent=10, **kw)
    tensors = next(iter(model._make_data_loader(adata, batch_size=64, shuffle=False)))
    return model, tensors


def _loss(model, tensors):
    torch.manual_seed(0)
    model.module.train()
    return model.module(tensors, loss_kwargs={"kl_weight": 1.0})[2]


def test_zero_weight_is_identical():
    scvi.settings.seed = 0
    off, tensors = _model(**BLIND)
    scvi.settings.seed = 0
    zero, _ = _model(**BLIND, u_invariance_weight=0.0)
    assert torch.equal(_loss(off, tensors).loss, _loss(zero, tensors).loss)


def test_penalty_adds_and_trains_only_the_u_encoder():
    scvi.settings.seed = 0
    model, tensors = _model(**BLIND, u_invariance_weight=5.0)
    out = _loss(model, tensors)
    pen = out.extra_metrics["u_invariance"]
    assert torch.isfinite(pen) and pen > 0

    module = model.module
    module.zero_grad()
    torch.manual_seed(0)
    module.train()
    inf = module.inference(**module._get_inference_input(tensors))
    module.u_invariance_penalty(tensors, inf).backward()
    with_grad = {
        name.split(".")[0]
        for name, p in module.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert with_grad == {"qu"}


def _penalty(model, tensors):
    module = model.module
    module.train()
    torch.manual_seed(0)
    inf = module.inference(**module._get_inference_input(tensors))
    return module.u_invariance_penalty(tensors, inf)


def test_counterfactual_is_exactly_zero_when_samples_are_identical():
    # Identical embedding rows make eps sample-independent, so a deterministic counterfactual
    # decode equals the own-sample decode: no dropout noise may leak into the penalty.
    scvi.settings.seed = 0
    model, tensors = _model(**BLIND, u_invariance_weight=5.0)
    with torch.no_grad():
        w = model.module.qz.embedding.weight
        w[:] = w[0]
    assert float(_penalty(model, tensors)) == 0.0


def test_counterfactual_keeps_train_flags_and_bn_stats():
    scvi.settings.seed = 0
    model, tensors = _model(**BLIND, u_invariance_weight=5.0)
    module = model.module
    module.train()
    buffers = {k: v.clone() for k, v in module.decoder.state_dict().items() if "running" in k}
    torch.manual_seed(0)
    inf = module.inference(**module._get_inference_input(tensors))
    module.u_invariance_penalty(tensors, inf)
    assert module.decoder.training and module.qz.training
    after = {k: v for k, v in module.decoder.state_dict().items() if "running" in k}
    assert all(torch.equal(buffers[k], after[k]) for k in buffers)


def test_dropout_consistency_ablation_measures_noise_only():
    scvi.settings.seed = 0
    model, tensors = _model(**BLIND, u_invariance_weight=5.0, u_invariance_mode="dropout_consistency")
    with torch.no_grad():
        w = model.module.qz.embedding.weight
        w[:] = w[0]
    # Same sample twice: only dropout differs, so the penalty is positive noise.
    assert float(_penalty(model, tensors)) > 0.0


def test_eval_mode_skips_penalty():
    scvi.settings.seed = 0
    model, tensors = _model(**BLIND, u_invariance_weight=5.0)
    model.module.eval()
    torch.manual_seed(0)
    out = model.module(tensors, loss_kwargs={"kl_weight": 1.0})[2]
    assert "u_invariance" not in out.extra_metrics


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"u_invariance_weight": -1.0, **BLIND}, ">= 0"),
        ({"u_invariance_weight": 1.0, "u_invariance_mode": "x", **BLIND}, "u_invariance_mode"),
        ({"u_invariance_weight": 1.0}, "sample_blind"),
        ({"u_invariance_weight": 1.0, "hierarchy_mode": "centered_v2", **BLIND}, "legacy"),
    ],
)
def test_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        _model(**kw)
