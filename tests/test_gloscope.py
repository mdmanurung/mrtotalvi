"""GloScope-style sample divergence and sample embedding (mrtotalvi._gloscope)."""

from __future__ import annotations

import numpy as np
import pytest

from mrtotalvi import MrTotalVI
from mrtotalvi._gloscope import _gmm, classical_mds, sample_divergence
from tests._simulate_adt import simulate_citeseq


def _two_gaussians(shift, n=3000, seed=0, m=None, scale=1.0):
    """Cells of two samples: n from N(0, I) and m from N(shift, scale^2 I), in 2-D."""
    rng = np.random.default_rng(seed)
    m = n if m is None else m
    x = np.vstack([rng.normal(size=(n, 2)), rng.normal(size=(m, 2)) * scale + np.asarray(shift)])
    return x, np.array(["a"] * n + ["b"] * m)


def test_gmm_kl_matches_closed_form():
    # KL(N(0,I)||N(mu,I)) = |mu|^2 / 2 per direction, so the symmetric sum is |mu|^2 = 1.
    x, ids = _two_gaussians([1.0, 0.0])
    d, samples = sample_divergence(x, ids, num_components=(1,), r=20000, random_state=0)
    assert samples == ["a", "b"]
    assert d[0, 1] == pytest.approx(1.0, abs=0.1)
    assert d[0, 1] == d[1, 0]
    assert np.all(np.diag(d) == 0)


def test_knn_kl_matches_closed_form():
    x, ids = _two_gaussians([1.0, 0.0])
    d, _ = sample_divergence(x, ids, dens="KNN", k=10)
    assert d[0, 1] == pytest.approx(1.0, abs=0.2)


def test_gmm_kl_sums_both_directions():
    # N(0, I) vs N(0, 4I) in 2-D: KL = 0.5 (0.5 - 2 + ln 16) = 0.636 one way and
    # 0.5 (8 - 2 - ln 16) = 1.614 the other; the symmetric sum is 2.250.
    x, ids = _two_gaussians([0.0, 0.0], scale=2.0)
    d, _ = sample_divergence(x, ids, num_components=(1,), r=20000, random_state=0)
    assert d[0, 1] == pytest.approx(2.25, abs=0.15)


def test_knn_kl_unequal_sizes():
    # The log(m / n) terms cancel in the symmetric sum, so unequal sizes leave the target
    # unchanged: about 0 for one distribution and |mu|^2 = 1 for a unit shift.
    same, ids = _two_gaussians([0.0, 0.0], n=500, m=3000)
    shifted, _ = _two_gaussians([1.0, 0.0], n=500, m=3000)
    assert sample_divergence(same, ids, dens="KNN", k=10)[0][0, 1] == pytest.approx(0.0, abs=0.15)
    assert sample_divergence(shifted, ids, dens="KNN", k=10)[0][0, 1] == pytest.approx(1.0, abs=0.25)


def test_gmm_bic_picks_true_component_count_and_caps_at_n_minus_1():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.vstack([rng.normal(size=(300, 2)) + c for c in centres])
    assert _gmm(x, (1, 3), random_state=0).n_components == 3
    assert _gmm(x[:4], (5, 10), random_state=0).n_components == 3


@pytest.mark.parametrize("dens", ["GMM", "KNN"])
def test_js_bounds(dens):
    # JS is near 0 for one distribution and near log 2 for disjoint ones.
    kwargs = {"num_components": (1,), "random_state": 0} if dens == "GMM" else {"k": 10}
    same, ids = _two_gaussians([0.0, 0.0])
    apart, _ = _two_gaussians([30.0, 0.0])
    assert sample_divergence(same, ids, dens=dens, dist_metric="JS", **kwargs)[0][0, 1] == pytest.approx(0, abs=0.05)
    assert sample_divergence(apart, ids, dens=dens, dist_metric="JS", **kwargs)[0][0, 1] == pytest.approx(
        np.log(2), abs=0.05
    )


def test_gmm_seed_reproducible():
    x, ids = _two_gaussians([1.0, 0.5], n=600)
    first = sample_divergence(x, ids, num_components=(1, 2), r=2000, random_state=3)[0]
    second = sample_divergence(x, ids, num_components=(1, 2), r=2000, random_state=3)[0]
    np.testing.assert_array_equal(first, second)


def test_knn_small_sample_is_nan():
    x, ids = _two_gaussians([1.0, 0.0], n=600)
    x, ids = np.vstack([x, np.zeros((5, 2))]), np.concatenate([ids, ["c"] * 5])
    with pytest.warns(UserWarning, match="NaN divergences"):
        d, samples = sample_divergence(x, ids, dens="KNN", k=10)
    c = samples.index("c")
    assert np.isnan(d[c]).all() and np.isnan(d[:, c]).all()
    assert np.isfinite(d[0, 1])


def test_classical_mds_recovers_euclidean_configuration():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    coords, eigval = classical_mds(dist, 2)
    np.testing.assert_allclose(np.linalg.norm(coords[:, None] - coords[None], axis=-1), dist, atol=1e-8)
    assert np.all(eigval > 0)


def test_classical_mds_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        classical_mds(np.array([[0.0, np.nan], [np.nan, 0.0]]))


def test_model_sample_divergence_and_embedding():
    adata, _, _ = simulate_citeseq(seed=0, cells_per_donor=40, n_empty=50)
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
    model = MrTotalVI(adata, sample_key="donor", n_latent=4)
    model.train(max_epochs=1, accelerator="cpu")
    with pytest.warns(UserWarning, match="fewer than 500 cells"):
        div = model.get_sample_divergence(dens="KNN", k=5)
    samples = list(model.sample_order)
    assert list(div["sample_x"].values) == samples
    np.testing.assert_allclose(div.values, div.values.T)
    assert np.all(np.diag(div.values) == 0)
    assert np.all(div.values[~np.eye(len(samples), dtype=bool)] > -1)

    emb = model.get_sample_embedding(div, n_components=2)
    assert list(emb.index) == samples
    assert emb.shape[0] == len(samples)
    with pytest.raises(ValueError, match="not both"):
        model.get_sample_embedding(div, dens="KNN")

    # A selection that leaves out one donor drops that donor from the output.
    first = samples[0]
    mask = adata.obs["donor"].to_numpy() != first
    keep = np.flatnonzero(mask)
    with pytest.warns(UserWarning):
        sub = model.get_sample_divergence(indices=keep, num_components=(1, 2), r=500, random_state=0)
    assert first not in list(sub["sample_x"].values)
    assert np.isfinite(sub.values).all()

    # Request order and repeats do not change the result: each sample is a set of cells.
    shuffled = np.concatenate([np.random.default_rng(1).permutation(keep), keep[:10]])
    with pytest.warns(UserWarning):
        again = model.get_sample_divergence(indices=shuffled, num_components=(1, 2), r=500, random_state=0)
    np.testing.assert_allclose(again.values, sub.values)
    with pytest.warns(UserWarning):
        masked = model.get_sample_divergence(indices=mask, num_components=(1, 2), r=500, random_state=0)
    np.testing.assert_allclose(masked.values, sub.values)


def _three_samples(seed=1, n=300):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(n, 2)), rng.normal(size=(n, 2)) + 0.5, rng.normal(size=(n, 2)) * 1.5])
    return x, np.repeat(["a", "b", "c"], n)


@pytest.mark.parametrize("metric", ["KL", "JS"])
def test_gmm_draws_fresh_points_per_pair_and_direction(monkeypatch, metric):
    # Two draws (one per direction) for each of the 3 pairs; no draw is reused between pairs.
    from mrtotalvi import _gloscope

    real = _gloscope._sample_gmm
    draws = []

    def counting(model, r, rng):
        out = real(model, r, rng)
        draws.append(out)
        return out

    monkeypatch.setattr(_gloscope, "_sample_gmm", counting)
    x, ids = _three_samples()
    sample_divergence(x, ids, dist_metric=metric, num_components=(1,), r=50, random_state=0)
    assert len(draws) == 6
    assert not any(np.array_equal(draws[i], draws[j]) for i in range(6) for j in range(i))


def test_gmm_reproducible_and_invariant_to_sample_order():
    x, ids = _three_samples()
    kw = {"num_components": (1, 2), "r": 500, "random_state": 7}
    d1, s1 = sample_divergence(x, ids, samples=["a", "b", "c"], **kw)
    d2, s2 = sample_divergence(x, ids, samples=["c", "a", "b"], **kw)
    np.testing.assert_array_equal(d1, sample_divergence(x, ids, samples=["a", "b", "c"], **kw)[0])
    perm = [s1.index(s) for s in s2]
    np.testing.assert_array_equal(d1[np.ix_(perm, perm)], d2)


@pytest.mark.parametrize("metric", ["KL", "JS"])
def test_gmm_identical_distributions_give_zero(metric):
    x, ids = _two_gaussians([0.0, 0.0], n=2000)
    d, _ = sample_divergence(x, ids, dist_metric=metric, num_components=(1,), r=10000, random_state=0)
    assert d[0, 1] == pytest.approx(0.0, abs=0.02)


def test_gmm_kl_matches_closed_form_1d():
    # N(0, 1) vs N(1, 2^2): KL(p||q) = ln 2 + (1 + 1) / 8 - 1/2; KL(q||p) = -ln 2 + (4 + 1) / 2 - 1/2.
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=4000), rng.normal(size=4000) * 2 + 1]).reshape(-1, 1)
    ids = np.repeat(["a", "b"], 4000)
    d, _ = sample_divergence(x, ids, num_components=(1,), r=10000, random_state=0)
    expected = np.log(2) + 2 / 8 - 0.5 + (-np.log(2) + 5 / 2 - 0.5)
    assert d[0, 1] == pytest.approx(expected, rel=0.06)


def test_knn_output_unchanged():
    # Values from the code before per-pair GMM draws; the k-NN path must stay byte-identical.
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(size=(80, 3)), rng.normal(size=(70, 3)) + 0.5, rng.normal(size=(60, 3)) * 1.5])
    ids = np.array(["a"] * 80 + ["b"] * 70 + ["c"] * 60)
    kl = sample_divergence(x, ids, dens="KNN", dist_metric="KL", k=5)[0][np.triu_indices(3, 1)]
    js = sample_divergence(x, ids, dens="KNN", dist_metric="JS", k=5)[0][np.triu_indices(3, 1)]
    assert kl.tolist() == [0.8568844875415633, 0.7633190900994535, 0.5810372267817885]
    assert js.tolist() == [0.15727561603897433, 0.1490265491408627, 0.12496325529913326]


def test_gmm_draw_counts_follow_hard_classification():
    from mrtotalvi._gloscope import _sample_gmm

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(300, 2)), rng.normal(size=(100, 2)) + 20])
    model = _gmm(x, (2,), random_state=0)
    np.testing.assert_allclose(model.hard_weights_, np.bincount(model.predict(x), minlength=2) / 400)
    assert sorted(model.hard_weights_) == [0.25, 0.75]
    # Counts follow hard_weights_, not weights_: all mass on one component puts every draw there.
    model.hard_weights_ = np.array([1.0, 0.0])
    draws = _sample_gmm(model, 200, np.random.default_rng(1))
    assert draws.shape == (200, 2)
    assert np.linalg.norm(draws - model.means_[0], axis=1).max() < 8
