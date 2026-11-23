"""Review fixes: docs and metadata. Only the DA posterior-scale warning changes behaviour."""

from __future__ import annotations

import warnings

import pytest

from mrtotalvi._stats import differential_abundance
from tests.test_da_density_estimator import StubModel, _stub_data


def _run_posterior(scale):
    model, adata, _ = _stub_data(depleted=False, n_per_sample=200)
    model.scale = scale
    return differential_abundance(
        model, adata=adata, sample_key="sample", sample_cov_keys=["cond"], density_estimator="posterior", batch_size=400
    )


def test_posterior_estimator_warns_when_scale_is_below_cell_spacing():
    with pytest.warns(UserWarning, match="scale_floor"):
        _run_posterior(StubModel.__init__.__defaults__[0])


def test_posterior_estimator_is_quiet_when_scale_exceeds_cell_spacing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _run_posterior(0.5)
