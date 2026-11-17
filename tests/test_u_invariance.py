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
        ({"u_invariance_weight": 1.0}, "sample_blind"),
        ({"u_invariance_weight": 1.0, "hierarchy_mode": "centered_v2", **BLIND}, "legacy"),
    ],
)
def test_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        _model(**kw)
