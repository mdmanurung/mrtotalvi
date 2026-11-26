"""Opt-in size-factor normalisation of get_protein_noise_decomposition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq

COUNT_KEYS = ("native", "ambient", "nonspecific", "observed")


@pytest.fixture(scope="module")
def fitted():
    adata, _, empties = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=800)
    profile = estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    model = MrTotalVI(
        adata,
        sample_key="donor",
        ambient_profile=profile,
        n_latent=4,
        protein_noise_model="ambient_isotype",
        isotype_names=ISOTYPES,
    )
    model.is_trained_ = True
    return model, adata


def _decompose(model, adata, **kwargs):
    torch.manual_seed(0)
    return model.get_protein_noise_decomposition(adata, n_samples=3, **kwargs)


def _totals(adata):
    # Independent of the package: row sums of the input matrix, isotypes included.
    return np.asarray(adata.obsm["protein"], dtype=float).sum(axis=1)


def test_default_unchanged(fitted):
    model, adata = fitted
    plain = _decompose(model, adata)
    explicit = _decompose(model, adata, size_factor=None)
    assert plain.keys() == explicit.keys()
    for key in plain:
        np.testing.assert_array_equal(plain[key].values, explicit[key].values)


def test_total_adt_divides_counts_by_size_factor(fitted):
    model, adata = fitted
    totals = _totals(adata)
    factor = totals / np.median(totals)
    plain = _decompose(model, adata)
    scaled = _decompose(model, adata, size_factor="total_adt")
    for key in COUNT_KEYS:
        np.testing.assert_allclose(scaled[key].values, plain[key].values / factor[:, None], rtol=1e-5)
    np.testing.assert_array_equal(scaled["foreground_probability"].values, plain["foreground_probability"].values)


def test_median_uses_whole_adata_for_a_subset(fitted):
    model, adata = fitted
    totals = _totals(adata)
    subset = np.array([5, 3, 90, 17])
    factor = totals[subset] / np.median(totals)
    plain = _decompose(model, adata, indices=subset)
    scaled = _decompose(model, adata, indices=subset, size_factor="total_adt")
    np.testing.assert_allclose(scaled["native"].values, plain["native"].values / factor[:, None], rtol=1e-5)
    # A subset median would differ, so this pins the whole-adata scale.
    assert np.median(totals[subset]) != np.median(totals)


def test_zero_total_cell_raises_only_when_requested(fitted):
    model, adata = fitted
    adata = adata.copy()
    protein = adata.obsm["protein"].copy()
    protein.iloc[[2, 7]] = 0
    adata.obsm["protein"] = protein
    with pytest.raises(ValueError, match="2 requested cells"):
        _decompose(model, adata, size_factor="total_adt")
    with pytest.raises(ValueError, match="1 requested cells"):
        _decompose(model, adata, indices=np.array([2, 10]), size_factor="total_adt")
    # Zero cells outside indices: no error, and they stay in the median.
    totals = np.asarray(protein, dtype=float).sum(axis=1)
    subset = np.array([10, 11])
    plain = _decompose(model, adata, indices=subset)
    scaled = _decompose(model, adata, indices=subset, size_factor="total_adt")
    factor = totals[subset] / np.median(totals)
    np.testing.assert_allclose(scaled["native"].values, plain["native"].values / factor[:, None], rtol=1e-5)


def test_invalid_size_factor_raises(fitted):
    model, adata = fitted
    with pytest.raises(ValueError, match="size_factor"):
        _decompose(model, adata, size_factor="median")
