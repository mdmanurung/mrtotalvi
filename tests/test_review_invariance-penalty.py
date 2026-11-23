"""u-invariance penalty decodes deterministically (also with use_map=False)."""

from __future__ import annotations

import scvi
import torch

from mrtotalvi import MrTotalVI

SETUP = {"protein_expression_obsm_key": "protein_expression", "sample_key": "sample", "batch_key": "batch"}
MAP_PENALTY = 6.695907359244302e-05  # value from commit 0030c3d, use_map=True default


def _model(**kw):
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = [f"s{i % 4}" for i in range(adata.n_obs)]
    MrTotalVI.setup_anndata(adata, **SETUP)
    model = MrTotalVI(
        adata, sample_key="sample", n_latent=10, u_encoder_mode="sample_blind", u_invariance_weight=5.0, **kw
    )
    tensors = next(iter(model._make_data_loader(adata, batch_size=64, shuffle=False)))
    return model, tensors


def _penalty(model, tensors):
    module = model.module
    module.train()
    torch.manual_seed(0)
    inf = module.inference(**module._get_inference_input(tensors))
    return module.u_invariance_penalty(tensors, inf)


def test_identical_embeddings_give_zero_penalty_without_map():
    scvi.settings.seed = 0
    model, tensors = _model(use_map=False)
    with torch.no_grad():
        w = model.module.qz.embedding.weight
        w[:] = w[0]
    assert float(_penalty(model, tensors)) == 0.0


def test_default_map_penalty_is_unchanged():
    # use_map=True ignores `deterministic`, so the default penalty must not change.
    scvi.settings.seed = 0
    model, tensors = _model()
    assert float(_penalty(model, tensors)) == MAP_PENALTY
