"""Regression tests for defects fixed while extracting MrTotalVI onto upstream scvi-tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import mrtotalvi
from mrtotalvi import MrTotalVI, combine_mrtotalvi_seed_results
from mrtotalvi._stats import _construct_design_matrix, collect_u_posterior
from tests.test_mrtotalvi import _make_adata, _setup_and_train
from tests.test_mrtotalvi_v2 import _make_v2_model


@pytest.fixture(scope="module")
def trained_legacy():
    adata = _make_adata()
    return _setup_and_train(adata, max_epochs=2), adata


def test_package_exposes_version():
    assert isinstance(mrtotalvi.__version__, str) and mrtotalvi.__version__


def test_design_matrix_treats_pandas_string_dtype_as_categorical():
    """pandas >= 3 stores strings as "str", not object; they must become dummies."""
    info = pd.DataFrame(
        {"timepoint": pd.array(["W00", "W22", "W00", "W22"], dtype="string")},
        index=pd.Index(["s0", "s1", "s2", "s3"], name="sample"),
    )
    xmat, names, n_fixed = _construct_design_matrix(info, ["timepoint"])
    assert list(names) == ["timepoint_W22"]
    assert n_fixed == 1
    np.testing.assert_array_equal(xmat[:, 0].numpy(), [0.0, 1.0, 0.0, 1.0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_centered_v2_mean_latent_works_on_gpu():
    """The sample index used to gather z must follow the model onto the GPU."""
    model, adata = _make_v2_model()
    model.to_device("cuda")
    z = model.get_latent_representation(adata, give_z=True, give_mean=True)
    assert z.shape == (adata.n_obs, model.module.n_latent)
    assert np.isfinite(z).all()


def test_learned_eps_prior_scale_starts_at_requested_value():
    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata, protein_expression_obsm_key="protein_expression", sample_key="sample"
    )
    model = MrTotalVI(
        adata, sample_key="sample", z_u_prior_scale=-2.0, learn_z_u_prior_scale=True
    )
    torch.testing.assert_close(
        model.module.pz_scale.detach(), torch.full_like(model.module.pz_scale, -2.0)
    )


def test_local_sample_distances_match_direct_computation(trained_legacy):
    """Chunked distances equal a direct per-cell computation, for both norms."""
    model, adata = trained_legacy
    reps = model.get_local_sample_representation(adata).values
    for norm in ("l2", "l1"):
        dists = model.get_local_sample_distances(adata, norm=norm, batch_size=7).values
        delta = reps[:, :, None, :] - reps[:, None, :, :]
        expected = (
            np.sqrt((delta**2).sum(-1)) if norm == "l2" else np.abs(delta).sum(-1)
        )
        np.testing.assert_allclose(dists, expected, rtol=1e-5, atol=1e-5)


def test_outlier_pairs_exclude_each_cells_own_component(trained_legacy):
    """A cell's own-sample score is the best match among the other cells' components."""
    model, adata = trained_legacy
    out = model.get_outlier_cell_sample_pairs(adata, subsample_size=None)
    us = model.get_latent_representation(adata, give_z=False, give_mean=True)
    loc, scale = collect_u_posterior(model, adata=adata)
    samples = adata.obs["sample"].to_numpy()
    sample = out.coords["sample"].values[0]
    members = np.flatnonzero(samples == sample)
    comp = torch.distributions.Normal(loc[members], scale[members])
    log_p = comp.log_prob(torch.as_tensor(us[members])[:, None, :]).sum(-1)
    log_p.fill_diagonal_(-float("inf"))
    expected = log_p.max(dim=1).values.numpy()
    np.testing.assert_allclose(
        out["log_probs"].sel(sample=sample).values[members], expected, rtol=1e-4, atol=1e-4
    )


def test_counterfactual_expression_unchanged_by_row_chunked_decode(monkeypatch):
    """Forcing many tiny decode chunks gives the same result as one chunk."""
    model, _ = _make_v2_model()
    kwargs = {"batch_policy": "observed", "library_policy": "observed"}
    reference = model.get_counterfactual_expression(**kwargs)
    original = model.module._deterministic_decoder_parameters
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model.module, "_deterministic_decoder_parameters", counting)
    monkeypatch.setattr(model.module, "n_input_genes", 10**8)  # forces 1-row chunks
    chunked = model.get_counterfactual_expression(**kwargs)
    assert max(calls) == 1 and len(calls) > 1
    xr.testing.assert_allclose(reference, chunked)


def test_seed_combiner_skips_between_seed_summary_for_latent_coordinates():
    rng = np.random.default_rng(0)

    def result():
        return xr.Dataset(
            {
                "z": (("draw", "cell_name", "latent_dim"), rng.normal(size=(2, 3, 4))),
                "rna_rate": (("draw", "cell_name", "gene"), rng.normal(size=(2, 3, 5))),
            },
            coords={"cell_name": ["a", "b", "c"]},
        )

    combined = combine_mrtotalvi_seed_results({0: result(), 1: result()})
    assert "rna_rate_between_seed_mean" in combined
    assert "z_between_seed_mean" not in combined
    assert "z_within_seed_posterior_mean" in combined
    assert combined.attrs["between_seed_skipped_latent_variables"] == "z"
