"""Regression tests for the ZINB counterfactual and sampler review fixes."""

from __future__ import annotations

import numpy as np
import torch

from mrtotalvi import MrTotalVI
from tests.test_mrtotalvi_v2 import _make_adata


def _make_model(gene_likelihood: str):
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        gene_likelihood=gene_likelihood,
    )
    model.is_trained_ = True
    return model, adata


def _fix_dropout(model, logit: float) -> None:
    head = model.module.decoder.px_dropout_decoder_gene
    for layer in head.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.constant_(layer.bias, logit)


def _expression(model, adata):
    return model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 1],
        inference_mode="latent_mean",
        n_draws=1,
        library_policy="specified",
        specified_library_size=52.0,
    )


def test_zinb_counterfactual_reports_zero_probability_and_expected_count():
    """Under ZINB, rna_rate stays the NB mean and the expected count is (1 - p_zero) * rate."""
    model, adata = _make_model("zinb")
    _fix_dropout(model, 0.0)
    result = _expression(model, adata)
    np.testing.assert_allclose(result["rna_rate"].sum("gene").to_numpy(), 52.0, rtol=1e-4)
    np.testing.assert_allclose(result["rna_zero_probability"].to_numpy(), 0.5, atol=1e-6)
    np.testing.assert_allclose(
        result["rna_expected_count"].to_numpy(),
        0.5 * result["rna_rate"].to_numpy(),
        rtol=1e-5,
    )


def test_nb_counterfactual_has_no_zinb_outputs():
    """The NB default keeps its variable set."""
    model, adata = _make_model("nb")
    result = _expression(model, adata)
    assert "rna_rate" in result.data_vars
    assert "rna_zero_probability" not in result.data_vars
    assert "rna_expected_count" not in result.data_vars


def test_zinb_sampler_draws_match_expected_count():
    """MrTotalVAE.sample draws ZINB, so the draw mean follows (1 - p_zero) * rate."""
    model, adata = _make_model("zinb")
    _fix_dropout(model, 0.0)
    module = model.module
    module.eval()
    tensors = next(iter(model._make_data_loader(adata=adata, indices=[0, 1], batch_size=2)))
    n_samples = 4000
    torch.manual_seed(0)
    _, generative = module.forward(tensors, inference_kwargs={"n_samples": n_samples}, compute_loss=False)
    expected = (0.5 * generative["px_"]["rate"]).sum().item() / n_samples
    torch.manual_seed(0)
    rna, _ = module.sample(tensors, n_samples=n_samples)
    observed = rna.sum().item() / n_samples
    assert abs(observed - expected) < 0.05 * expected
