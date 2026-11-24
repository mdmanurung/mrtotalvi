"""protein_components moves every batch tensor to the module device (z_base background head on GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient
from tests._simulate_adt import simulate_citeseq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zbase_decomposition_runs_on_gpu():
    # Regression: the z_base background head got CPU batch tensors next to CUDA latents.
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
        protein_noise_model="ambient",
        protein_background_source="z_base",
    )
    model.train(max_epochs=1, accelerator="gpu", batch_size=128)
    comp = model.get_protein_noise_decomposition(n_samples=2)
    assert np.isfinite(comp["native"].values).all()
