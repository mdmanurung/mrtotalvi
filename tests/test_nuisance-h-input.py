"""nuisance_h_input='isotypes_batch' keeps target counts out of the encoder for h."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi._components import AmbientNuisanceEncoder
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import ISOTYPES, simulate_citeseq


def _h_mean(encoder, y, iso, batch):
    with torch.no_grad():
        return encoder(y, iso, batch)[1].loc


@pytest.mark.parametrize(("h_input", "moves"), [("isotypes_total", True), ("isotypes_batch", False)])
def test_target_counts_reach_h_only_through_the_total(h_input, moves):
    torch.manual_seed(0)
    encoder = AmbientNuisanceEncoder(6, n_isotypes=2, h_input=h_input, n_batch=3)
    iso = torch.tensor([4, 5])
    y = torch.poisson(torch.full((8, 6), 5.0))
    batch = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]).reshape(-1, 1)
    more_target = y.clone()
    more_target[:, :4] *= 20
    changed = not torch.allclose(_h_mean(encoder, y, iso, batch), _h_mean(encoder, more_target, iso, batch))
    assert changed is moves


def test_isotypes_batch_needs_batch_index():
    encoder = AmbientNuisanceEncoder(6, n_isotypes=2, h_input="isotypes_batch", n_batch=2)
    with pytest.raises(ValueError, match="needs batch_index"):
        encoder(torch.ones(3, 6), torch.tensor([4, 5]))


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
    return MrTotalVI(adata, sample_key="donor", ambient_profile=profile, n_latent=4, **kwargs)


def test_model_trains_and_decomposes_with_isotypes_batch(sim):
    model = _model(
        sim, protein_noise_model="ambient_isotype", isotype_names=ISOTYPES, nuisance_h_input="isotypes_batch"
    )
    model.train(max_epochs=2, accelerator="cpu")
    native = model.get_protein_noise_decomposition(n_samples=2)["native"]
    targets = native.drop(columns=ISOTYPES, errors="ignore")
    assert np.isfinite(targets.to_numpy()).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"protein_noise_model": "ambient_isotype", "isotype_names": ISOTYPES, "nuisance_h_input": "x"}, "must be"),
        ({"protein_noise_model": "ambient", "nuisance_h_input": "isotypes_batch"}, "applies to"),
    ],
)
def test_nuisance_h_input_is_checked(sim, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(sim, **kwargs)
