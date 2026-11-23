"""Feature identity: refuse RNA or protein axes that differ from the registry."""

import numpy as np
import pandas as pd
import pytest
import scvi

from mrtotalvi import MrTotalVI

KW = {"protein_expression_obsm_key": "protein_expression", "sample_key": "sample", "batch_key": "batch"}


def _adata(n_proteins=4, dataframe=False):
    adata = scvi.data.synthetic_iid(batch_size=30, n_genes=8, n_proteins=n_proteins, n_regions=2, n_batches=2)
    adata.obs["sample"] = [f"s{i % 3}" for i in range(adata.n_obs)]
    names = [f"p{i}" for i in range(n_proteins)]
    if dataframe:
        frame = pd.DataFrame(np.asarray(adata.obsm["protein_expression"]), columns=names, index=adata.obs_names)
        adata.obsm["protein_expression"] = frame
    return adata, names


def _model(adata):
    MrTotalVI.setup_anndata(adata, **KW)
    model = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2)
    model.is_trained_ = True
    return model


def test_permuted_rna_query_is_refused():
    adata, _ = _adata()
    model = _model(adata)
    query = adata[:, [1, 0, 2, 3, 4, 5, 6, 7]].copy()
    with pytest.raises(ValueError, match="var_names"):
        model.get_latent_representation(query)


def test_permuted_rna_on_load_is_refused(tmp_path):
    adata, _ = _adata()
    model = _model(adata)
    model.save(tmp_path / "m", save_anndata=False)
    query = adata[:, [1, 0, 2, 3, 4, 5, 6, 7]].copy()
    with pytest.raises(ValueError, match="var_names"):
        MrTotalVI.load(tmp_path / "m", adata=query)


def test_same_rna_query_is_accepted():
    adata, _ = _adata()
    model = _model(adata)
    assert model.get_latent_representation(adata.copy()).shape[0] == adata.n_obs


def test_setup_refuses_dataframe_uns_conflict():
    adata, names = _adata(dataframe=True)
    adata.uns["pn"] = names[::-1]
    with pytest.raises(ValueError, match="agree exactly"):
        MrTotalVI.setup_anndata(adata, protein_names_uns_key="pn", **KW)


def test_setup_refuses_duplicate_protein_names():
    adata, names = _adata()
    adata.uns["pn"] = ["dup"] * len(names)
    with pytest.raises(ValueError, match="unique"):
        MrTotalVI.setup_anndata(adata, protein_names_uns_key="pn", **KW)


def test_permuted_protein_dataframe_query_is_refused():
    adata, _ = _adata(dataframe=True)
    model = _model(adata)
    query = adata.copy()
    query.obsm["protein_expression"] = query.obsm["protein_expression"].iloc[:, [1, 0, 2, 3]]
    with pytest.raises(ValueError, match="protein"):
        model.get_latent_representation(query)
