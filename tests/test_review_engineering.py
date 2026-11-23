"""Engineering review fixes: seed hash check, protein subset frames, dispersion options, load flag."""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pandas as pd
import pytest
import scvi
import xarray as xr
from scvi.model import TOTALVI

from mrtotalvi import MrTotalVI, combine_mrtotalvi_seed_results

from .test_mrtotalvi_v2 import _make_adata, _make_v2_model


def _seed_result(hash_value, estimand="observed", offset=0.0):
    draws = np.arange(6, dtype=float).reshape(2, 3) + offset
    attrs = {"observed_library_estimand": estimand, "schema_version": "1"}
    if hash_value is not None:
        attrs["context_table_sha256"] = hash_value
    return xr.Dataset({"rate": (("draw", "gene"), draws)}, attrs=attrs)


def test_seed_combiner_rejects_different_context_hashes():
    results = {0: _seed_result("aaa"), 1: _seed_result("bbb", offset=1.0)}
    with pytest.raises(ValueError, match="context_table_sha256"):
        combine_mrtotalvi_seed_results(results)


def test_seed_combiner_allows_different_hashes_for_posterior_lognormal_mean():
    kind = "posterior_lognormal_mean"
    results = {0: _seed_result("aaa", kind), 1: _seed_result("bbb", kind, offset=1.0)}
    combined = combine_mrtotalvi_seed_results(results)
    assert "aaa" in combined.attrs["context_table_sha256"]
    assert "bbb" in combined.attrs["context_table_sha256"]


def test_seed_combiner_warns_when_hash_is_missing_on_some_results():
    results = {0: _seed_result("aaa"), 1: _seed_result(None, offset=1.0)}
    with pytest.warns(UserWarning, match="context_table_sha256"):
        combine_mrtotalvi_seed_results(results)


def test_seed_combiner_rejects_mismatch_when_template_lacks_hash():
    results = {0: _seed_result(None), 1: _seed_result("aaa", offset=1.0), 2: _seed_result("bbb", offset=2.0)}
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="context_table_sha256 differs"):
        combine_mrtotalvi_seed_results(results)


def test_seed_combiner_lognormal_hash_list_has_no_repeats():
    kind = "posterior_lognormal_mean"
    results = {0: _seed_result("aaa", kind), 1: _seed_result("aaa", kind, offset=1.0)}
    assert combine_mrtotalvi_seed_results(results).attrs["context_table_sha256"] == "aaa"


def test_seed_combiner_accepts_equal_hashes():
    results = {0: _seed_result("aaa"), 1: _seed_result("aaa", offset=1.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        combined = combine_mrtotalvi_seed_results(results)
    assert combined.attrs["context_table_sha256"] == "aaa"


def _frame_protein_model():
    adata = scvi.data.synthetic_iid(batch_size=4, n_genes=5, n_proteins=3, n_regions=2, n_batches=2, n_labels=2)
    adata.obs["sample"] = [f"sample_{i % 3}" for i in range(adata.n_obs)]
    adata.obsm["protein_expression"] = pd.DataFrame(
        np.asarray(adata.obsm["protein_expression"]), index=adata.obs_names, columns=["CD3", "CD4", "CD8"]
    )
    MrTotalVI.setup_anndata(
        adata, protein_expression_obsm_key="protein_expression", sample_key="sample", batch_key="batch"
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2, hierarchy_mode="centered_v2")
    model.is_trained_ = True
    return model, adata


def test_protein_subset_on_dataframe_protein_axis():
    model, adata = _frame_protein_model()
    rna, protein = model.get_normalized_expression(adata, n_samples=1, protein_list=["CD3", "CD8"])
    assert list(protein.columns) == ["CD3", "CD8"]
    assert list(protein.index) == list(adata.obs_names)
    assert rna.shape[1] == adata.n_vars
    genes = list(adata.var_names[:2])
    rna, subset = model.get_normalized_expression(
        adata, indices=[3, 1], n_samples=1, gene_list=genes, protein_list=["CD4"]
    )
    assert list(subset.index) == list(adata.obs_names[[3, 1]])
    assert list(subset.columns) == ["CD4"]
    assert list(rna.columns) == genes


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gene_dispersion": "gene-cell"}, "gene-cell"),
        ({"protein_dispersion": "protein-cell"}, "protein-cell"),
        ({"gene_dispersion": "foo"}, "gene_dispersion must be one of"),
        ({"protein_dispersion": "foo"}, "protein_dispersion must be one of"),
    ],
)
def test_cell_dispersion_is_rejected_at_construction(kwargs, match):
    adata = _make_adata(n_samples=3)
    MrTotalVI.setup_anndata(
        adata, protein_expression_obsm_key="protein_expression", sample_key="sample", batch_key="batch"
    )
    with pytest.raises(ValueError, match=match):
        MrTotalVI(adata, sample_key="sample", **kwargs)


def test_load_flag_does_not_leak_to_other_threads_or_after_load(tmp_path):
    model, adata = _make_v2_model()
    path = str(tmp_path / "m")
    model.save(path, overwrite=True, save_anndata=False)
    inside, proceed = threading.Event(), threading.Event()
    original = TOTALVI.load.__func__

    def slow_load(cls, *args, **kwargs):
        inside.set()
        proceed.wait(30)
        return original(cls, *args, **kwargs)

    TOTALVI.load = classmethod(slow_load)
    try:
        thread = threading.Thread(target=lambda: MrTotalVI.load(path, adata=adata))
        thread.start()
        assert inside.wait(30)
        fresh = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2, hierarchy_mode="centered_v2")
        proceed.set()
        thread.join()
    finally:
        TOTALVI.load = classmethod(original)
    assert fresh.init_params_["non_kwargs"]["spec_defaults"] is True
    after = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2, hierarchy_mode="centered_v2")
    assert after.init_params_["non_kwargs"]["spec_defaults"] is True
