"""Memory fixes: the legacy outlier cutoff and the ambient statistics run in row chunks."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

import mrtotalvi._ambient as ambient
from mrtotalvi._stats import _rowwise_max_excluding_diagonal, get_outlier_cell_sample_pairs
from tests._simulate_adt import simulate_citeseq
from tests.test_minor_batch import _OutlierStub
from tests.test_review_fixes import _outlier_data


def test_leave_one_out_max_matches_full_matrix_in_chunks():
    from mrtotalvi._stats import _leave_one_out_max

    gen = torch.Generator().manual_seed(0)
    comps = Independent(Normal(torch.randn(23, 3, generator=gen), torch.ones(23, 3)), 1)
    mixture = MixtureSameFamily(Categorical(torch.ones(23)), comps)
    points = torch.randn(23, 3, generator=gen)
    # Old implementation: the whole n x n matrix, then the diagonal masked.
    old = _rowwise_max_excluding_diagonal(comps.log_prob(points.unsqueeze(-2)))
    new = _leave_one_out_max(mixture.component_distribution, points, 5)
    torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_outlier_cutoff_never_scores_the_full_sample_square(monkeypatch):
    u, labels, adata = _outlier_data()
    request = {"adata": adata, "sample_key": "sample", "subsample_size": None, "random_state": 0}
    whole = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), batch_size=1000, **request)
    seen = []
    log_prob = Independent.log_prob

    def spy(self, x):
        seen.append(x.shape[0])
        return log_prob(self, x)

    monkeypatch.setattr(Independent, "log_prob", spy)
    chunked = get_outlier_cell_sample_pairs(_OutlierStub(u, labels), batch_size=8, **request)
    assert max(seen) <= 8
    np.testing.assert_array_equal(chunked["log_probs"].values, whole["log_probs"].values)
    np.testing.assert_array_equal(chunked["log_ratios"].values, whole["log_ratios"].values)


def test_ambient_log_stats_match_dense_and_densify_in_chunks(monkeypatch):
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.7, size=(53, 6)).astype(np.float64)
    counts = sp.csr_matrix(dense)
    monkeypatch.setattr(ambient, "_STAT_CHUNK_ROWS", 10, raising=False)
    shapes = []
    to_array = sp.csr_matrix.toarray

    def spy(self, *args, **kwargs):
        shapes.append(self.shape[0])
        return to_array(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", spy)
    for pseudocount in (1.0, ambient.DSB_PSEUDOCOUNT):
        mean, sd = ambient._log_stats(counts, pseudocount)
        logged = np.log(dense + pseudocount)
        np.testing.assert_allclose(mean, logged.mean(axis=0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(sd, logged.std(axis=0, ddof=1), rtol=1e-12, atol=1e-14)
    assert max(shapes) <= 10


def test_estimate_adt_ambient_hands_sparse_cell_free_to_log_stats(monkeypatch):
    adata, _, empties = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=800)
    seen = []
    log_stats = ambient._log_stats

    def spy(counts, pseudocount):
        seen.append(sp.issparse(counts))
        return log_stats(counts, pseudocount)

    monkeypatch.setattr(ambient, "_log_stats", spy)
    ambient.estimate_adt_ambient(
        empties["raw_counts"],
        empties["raw_barcodes"],
        adata.obs_names,
        empties["feature_names"],
        adt_features=empties["adt_features"],
    )
    assert seen == [True, True]
