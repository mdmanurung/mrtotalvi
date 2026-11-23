"""Cell selection by position: masks, request order and backed AnnData (review 2026-11-23)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scvi import REGISTRY_KEYS

from tests.test_mrtotalvi_v2 import _make_v2_model

MASK = np.zeros(8, dtype=bool)
MASK[[6, 1, 3]] = True
REQUESTS = {"unsorted": [7, 1, 4], "repeated": [7, 1, 4, 1], "mask": MASK}


@pytest.fixture
def pair(tmp_path):
    model, adata = _make_v2_model()
    adata.write_h5ad(tmp_path / "a.h5ad")
    backed = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    model._validate_anndata(backed)
    model._validate_anndata(adata)
    model.module.eval()
    # Echo the observed counts so posterior_predictive_sample is deterministic per cell.
    model.module.sample = lambda tensors, n_samples=1: (
        tensors[REGISTRY_KEYS.X_KEY].cpu().numpy(),
        tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY].cpu().numpy(),
    )
    return model, adata, backed


def _fingerprint(values):
    if isinstance(values, dict):
        return {key: _fingerprint(value) for key, value in values.items()}
    if isinstance(values, tuple):
        return tuple(_fingerprint(value) for value in values)
    if isinstance(values, pd.DataFrame):
        return values.index.to_list(), values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _assert_same(got, expected, atol):
    if isinstance(expected, dict):
        assert got.keys() == expected.keys()
        for key in expected:
            _assert_same(got[key], expected[key], atol)
    elif isinstance(expected, tuple) and isinstance(expected[0], list):
        assert got[0] == expected[0]
        np.testing.assert_allclose(got[1], expected[1], atol=atol)
    elif isinstance(expected, tuple):
        for g, e in zip(got, expected, strict=True):
            _assert_same(g, e, atol)
    else:
        np.testing.assert_allclose(got, expected, atol=atol)


def _methods(model):
    # (call, atol): stochastic methods average many draws and get a wider tolerance.
    return {
        "latent": (lambda a, i: model.get_latent_representation(adata=a, indices=i, give_mean=True), 1e-5),
        "local_sample_representation": (lambda a, i: model.get_local_sample_representation(adata=a, indices=i), 1e-5),
        "library_size": (lambda a, i: model.get_latent_library_size(adata=a, indices=i), 1e-5),
        "ppc": (lambda a, i: model.posterior_predictive_sample(adata=a, indices=i), 0),
        "foreground": (
            lambda a, i: model.get_protein_foreground_probability(adata=a, indices=i, n_samples=300),
            0.03,
        ),
        "normalized": (
            lambda a, i: model.get_normalized_expression(adata=a, indices=i, n_samples=300),
            0.15,
        ),
        "aggregated": (
            lambda a, i: model.get_aggregated_posterior(adata=a, indices=i).component_distribution.base_dist.loc,
            1e-5,
        ),
    }


@pytest.mark.parametrize("request_name", REQUESTS)
def test_backed_matches_in_memory_for_every_indices_method(pair, request_name):
    model, adata, backed = pair
    indices = REQUESTS[request_name]
    for name, (call, atol) in _methods(model).items():
        torch.manual_seed(0)
        expected = _fingerprint(call(adata, indices))
        torch.manual_seed(0)
        got = _fingerprint(call(backed, indices))
        try:
            _assert_same(got, expected, atol)
        except (AssertionError, TypeError) as err:
            raise AssertionError(f"{name} with {request_name} indices differs on backed data: {err}") from err


@pytest.mark.parametrize("request_name", REQUESTS)
def test_backed_normalized_expression_n_samples_overall(pair, request_name):
    model, adata, backed = pair
    outputs = []
    for data in (adata, backed):
        np.random.seed(3)
        torch.manual_seed(0)
        outputs.append(
            model.get_normalized_expression(
                adata=data, indices=REQUESTS[request_name], n_samples_overall=5, n_samples=300
            )
        )
    (rna_mem, pro_mem), (rna_bk, pro_bk) = outputs
    assert rna_bk.shape[0] == 5
    assert rna_bk.index.to_list() == rna_mem.index.to_list()
    np.testing.assert_allclose(rna_bk.to_numpy(), rna_mem.to_numpy(), atol=0.15)
    np.testing.assert_allclose(pro_bk.to_numpy(), pro_mem.to_numpy(), atol=0.15)


def test_aggregated_posterior_mask_with_sample_selects_masked_rows(pair):
    model, adata, _ = pair
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[[3, 6]] = True
    got = model.get_aggregated_posterior(adata=adata, indices=mask, sample="sample_0")
    expected = model.get_aggregated_posterior(adata=adata, indices=[3, 6], sample="sample_0")
    loc = got.component_distribution.base_dist.loc
    np.testing.assert_allclose(loc.numpy(), expected.component_distribution.base_dist.loc.numpy(), atol=1e-6)
    # Rows 3 and 6 belong to sample_0; row 0 (also sample_0) must not leak in.
    assert loc.shape[0] == 2


def test_boolean_mask_of_wrong_length_raises(pair):
    from mrtotalvi._stats import _as_indices

    _, adata, _ = pair
    with pytest.raises(ValueError, match="length"):
        _as_indices(adata, np.ones(adata.n_obs + 1, dtype=bool))
