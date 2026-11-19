"""Unsupported inherited paths must fail loudly, not silently or obscurely."""

from __future__ import annotations

import pytest
import scvi

from mrtotalvi import MrTotalVI

SETUP = dict(
    protein_expression_obsm_key="protein_expression",
    sample_key="sample",
    batch_key="batch",
)


def _adata(n_samples: int = 4):
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = [f"s{i % n_samples}" for i in range(adata.n_obs)]
    return adata


def _trained(tmp_path=None):
    adata = _adata()
    MrTotalVI.setup_anndata(adata, **SETUP)
    model = MrTotalVI(adata, sample_key="sample", n_latent=10)
    model.train(max_epochs=1, accelerator="cpu")
    return model, adata


def test_load_query_data_with_new_samples_is_rejected(tmp_path):
    # Before the fix this loaded (n_sample 4 -> 7), froze the sample tables and
    # left the new samples with random, untrained embedding rows.
    model, adata = _trained()
    ref_dir = str(tmp_path / "ref")
    model.save(ref_dir, overwrite=True, save_anndata=False)
    query = adata.copy()
    # Mixed query: all four reference samples plus three new ones.
    query.obs["sample"] = [
        f"q{i % 3}" if i % 5 == 4 else f"s{i % 4}" for i in range(query.n_obs)
    ]
    with pytest.raises(NotImplementedError, match=r"docs/user_guide\.md"):
        MrTotalVI.load_query_data(query, ref_dir)


def test_load_query_data_with_pure_query_is_rejected_with_doc_pointer(tmp_path):
    # Before the fix: a misleading 'Donors ... have no cells' error.
    model, adata = _trained()
    query = adata.copy()
    query.obs["sample"] = [f"q{i % 3}" for i in range(query.n_obs)]
    with pytest.raises(NotImplementedError, match=r"docs/user_guide\.md"):
        MrTotalVI.load_query_data(query, model)


def test_load_query_data_with_instance_is_rejected():
    model, adata = _trained()
    with pytest.raises(NotImplementedError, match="reference-query surgery"):
        MrTotalVI.load_query_data(adata, model)


def test_minify_adata_is_rejected():
    # Before the fix: KeyError 'X_latent_qzm' from scvi internals.
    model, _ = _trained()
    with pytest.raises(NotImplementedError, match=r"docs/user_guide\.md"):
        model.minify_adata()


def test_gene_label_dispersion_is_rejected_at_construction():
    # Before the fix this built, then training raised
    # AttributeError "'NoneType' object has no attribute 'squeeze'".
    adata = _adata()
    MrTotalVI.setup_anndata(adata, **SETUP)
    with pytest.raises(ValueError, match="gene-label"):
        MrTotalVI(adata, sample_key="sample", gene_dispersion="gene-label")


def test_protein_label_dispersion_is_rejected_at_construction():
    # Before the fix this built a (n_proteins, 0) dispersion, then the first forward
    # pass raised AttributeError "'NoneType' object has no attribute 'squeeze'".
    adata = _adata()
    MrTotalVI.setup_anndata(adata, **SETUP)
    with pytest.raises(ValueError, match="protein-label"):
        MrTotalVI(adata, sample_key="sample", protein_dispersion="protein-label")
