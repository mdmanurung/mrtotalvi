"""Isotype controls have no identified foreground probability in ambient_isotype mode."""

from __future__ import annotations

import numpy as np
import pytest
from scvi import REGISTRY_KEYS

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq


@pytest.fixture(scope="module")
def sim():
    adata, _, empties = simulate_citeseq(seed=0, cells_per_donor=20, n_empty=400)
    profile = estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    return adata, profile


def _model(sim, **kwargs):
    adata, profile = sim
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile, n_latent=4, **kwargs), adata


def test_isotype_foreground_probability_is_nan(sim):
    model, _ = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    fg = model.get_protein_foreground_probability()
    assert fg[ISOTYPES].isna().all().all()
    assert fg.drop(columns=ISOTYPES).notna().all().all()


def test_isotype_normalized_protein_is_nan(sim):
    model, _ = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    _, protein = model.get_normalized_expression()
    assert protein[ISOTYPES].isna().all().all()
    assert protein.drop(columns=ISOTYPES).notna().all().all()


def test_isotype_nan_follows_protein_list_and_numpy(sim):
    model, _ = _model(sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES)
    names = list(model.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
    subset = [names[0], ISOTYPES[0], names[1]]
    fg = model.get_protein_foreground_probability(protein_list=subset, return_numpy=True)
    expected = np.isin([n for n in names if n in subset], ISOTYPES)
    assert np.array_equal(np.isnan(fg).all(axis=0), expected)


@pytest.mark.parametrize("mode", ["totalvi", "ambient"])
def test_other_modes_keep_all_columns_finite(sim, mode):
    model, _ = _model(sim, protein_noise_model=mode)
    assert model.get_protein_foreground_probability().notna().all().all()
    assert model.get_normalized_expression()[1].notna().all().all()
