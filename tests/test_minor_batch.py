"""Regression tests for the minor architecture-review batch (ranks 7-31)."""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scvi
import torch
import xarray as xr
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from mrtotalvi import MrTotalVI
from mrtotalvi._datamodule import _InMemoryProteinMappedDataset
from mrtotalvi._seed import combine_mrtotalvi_seed_results
from mrtotalvi._stats import (
    _knn_log_density,
    _knn_sample_counts,
    differential_abundance,
    get_outlier_cell_sample_pairs,
)
from tests.test_da_density_estimator import StubModel
from tests.test_mrtotalvi import N_LATENT, _make_adata, _setup_and_train

ROOT = Path(__file__).resolve().parents[1]
SETUP = {"protein_expression_obsm_key": "protein_expression", "sample_key": "sample", "batch_key": "batch"}


@pytest.fixture(scope="module")
def trained():
    scvi.settings.seed = 0
    return _setup_and_train(_make_adata(), max_epochs=2)


def _loss_out(module, tensors, n_samples=1):
    inf = module._regular_inference(**module._get_inference_input(tensors), n_samples=n_samples)
    gen = module.generative(**module._get_generative_input(tensors, inf))
    return module.loss(tensors, inf, gen)


def _tensors(model, n=16):
    return next(iter(model._make_data_loader(adata=model.adata, batch_size=n)))


# --- rank 7: size check, groupby and keep_cell -----------------------------------------


def test_local_distances_size_check_groupby_and_keep_cell(trained):
    plain = trained.get_local_sample_distances(batch_size=64)
    with pytest.raises(ValueError, match="max_gb"):
        trained.get_local_sample_distances(max_gb=1e-9)
    with pytest.raises(ValueError, match="max_gb"):
        trained.get_local_sample_representation(max_gb=1e-9)
    with pytest.raises(ValueError, match="keep_cell=False needs groupby"):
        trained.get_local_sample_distances(keep_cell=False)

    grouped = trained.get_local_sample_distances(batch_size=64, groupby="batch", keep_cell=False)
    assert "cell" not in grouped
    batch = trained.adata.obs["batch"].to_numpy()
    for label in np.unique(batch):
        expected = plain.values[batch == label].mean(axis=0)
        np.testing.assert_allclose(grouped["batch_distances"].sel(batch=label).values, expected, rtol=1e-4, atol=1e-5)
    both = trained.get_local_sample_distances(batch_size=64, groupby=["batch"], keep_cell=True)
    np.testing.assert_allclose(both["cell"].values, plain.values, rtol=1e-6)
    # keep_cell=False stays under a limit that the per-cell output would break
    trained.get_local_sample_distances(groupby="batch", keep_cell=False, max_gb=1e-9)


# --- rank 8: DA directional bias with unequal groups (documented behaviour) --------------


def test_posterior_da_null_tracks_group_size_but_knn_and_floor_do_not():
    rng = np.random.default_rng(0)
    names = [f"a{i}" for i in range(2)] + [f"b{i}" for i in range(8)]
    u = rng.normal(size=(300 * len(names), 2))
    labels = np.repeat(names, 300)
    obs = pd.DataFrame(
        {"sample": labels, "cond": [n[0] for n in labels]},
        index=[f"c{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(np.zeros((len(labels), 1), dtype=np.float32), obs=obs)
    out = {}
    for name, kw in {
        "posterior": {},
        "knn": {"density_estimator": "knn"},
        "floor": {"density_estimator": "scale_floor", "scale_floor": 0.4},
    }.items():
        da = differential_abundance(
            StubModel(u, labels),
            adata=adata,
            sample_key="sample",
            sample_cov_keys=["cond"],
            compute_log_enrichment=True,
            batch_size=600,
            **kw,
        )
        out[name] = float(np.nanmean(da["cond_log_enrichs"].sel(cond_enrichment="a").values))
    # the 2-sample group scores below the 8-sample group under the default estimator
    assert out["posterior"] < -1.0
    assert abs(out["knn"]) < 0.3
    assert abs(out["floor"]) < 3.0


# --- rank 9: knn pseudocount bias for a small sample (documented behaviour) ---------------


def test_knn_per_sample_score_favours_small_sample():
    rng = np.random.default_rng(1)
    sizes = [600, 600, 600, 10]
    codes = np.repeat(np.arange(4), sizes)
    u = rng.normal(size=(len(codes), 2)).astype(np.float32)
    counts, n_ref = _knn_sample_counts(u, codes, 4, 50, False)
    scores = _knn_log_density(counts, n_ref)
    assert (scores.argmax(axis=1) == 3).mean() > 0.9


# --- rank 11 / 15 / 16: documented behaviour -------------------------------------------------


def test_pz_scale_is_clamped_at_minus_four(trained):
    """Documented: a log-scale below -4 acts as -4 (so it cannot be learned below it)."""
    module = trained.module.eval()
    tensors = _tensors(trained)

    def loss_at(value):
        with torch.no_grad():
            module.pz_scale.fill_(value)
            torch.manual_seed(0)
            return float(_loss_out(module, tensors).loss)

    assert loss_at(-6.0) == pytest.approx(loss_at(-4.0), abs=1e-6)
    assert loss_at(-3.0) != pytest.approx(loss_at(-4.0), abs=1e-3)
    module.pz_scale.fill_(0.0)


# --- rank 12: protein samples use the fitted efficiency ------------------------------------


def test_posterior_predictive_protein_sample_uses_efficiency(trained):
    module = trained.module.eval()
    tensors = _tensors(trained, 32)
    original = module.generative

    def with_efficiency(factor):
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            out["per_batch_efficiency"] = out["per_batch_efficiency"] * factor
            return out

        return wrapped

    try:
        means = {}
        for factor in (1.0, 0.01):
            module.generative = with_efficiency(factor)
            torch.manual_seed(0)
            means[factor] = module.sample(tensors)[1].float().mean().item()
    finally:
        module.generative = original
    assert means[0.01] < 0.2 * means[1.0]


# --- rank 13 / 31: wording -----------------------------------------------------------------


def test_docs_and_warning_wording():
    guide = (ROOT / "docs" / "user_guide.md").read_text()
    assert "sample-free" not in guide
    assert "MrTotalVIBatchDataModule" not in guide
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scvi.settings.seed = 0
        adata = _make_adata()
        MrTotalVI.setup_anndata(adata, **SETUP)
        model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
        model._warn_inherited_training_schedule(
            max_epochs=1,
            train_size=None,
            batch_size=None,
            n_steps_kl_warmup=None,
            n_epochs_kl_warmup=None,
            adversarial_classifier=True,
        )
    text = " ".join(str(w.message) for w in caught)
    assert "sample-free" not in text and "not on u alone" in text


# --- rank 14: own-sample admissibility is below 1 by design ---------------------------------


class _OutlierStub(StubModel):
    def get_aggregated_posterior(self, adata, indices, batch_size):
        loc = torch.as_tensor(self.u[indices])
        scale = torch.full_like(loc, 0.5)
        probs = torch.ones(loc.shape[0]) / loc.shape[0]
        return MixtureSameFamily(Categorical(probs=probs), Independent(Normal(loc, scale), 1))


def test_own_sample_pairs_fail_about_quantile_by_design():
    rng = np.random.default_rng(2)
    labels = np.repeat(["s0", "s1", "s2"], 400)
    u = rng.normal(size=(len(labels), 2))
    obs = pd.DataFrame({"sample": labels}, index=[f"c{i}" for i in range(len(labels))])
    adata = ad.AnnData(np.zeros((len(labels), 1), dtype=np.float32), obs=obs)
    out = get_outlier_cell_sample_pairs(
        _OutlierStub(u, labels),
        adata=adata,
        sample_key="sample",
        subsample_size=None,
        quantile_threshold=0.05,
        batch_size=600,
    )
    own = np.array([out["is_admissible"].sel(sample=s).values[labels == s] for s in set(labels)])
    assert 0.9 <= own.mean() <= 0.99


# --- rank 21: singular X^T M X -------------------------------------------------------------


def test_sqrtm_batch_is_finite_for_singular_matrices():
    from mrtotalvi._stats import _sqrtm_batch

    batch = torch.zeros(2, 3, 3)
    batch[1] = torch.eye(3) * 4
    root = _sqrtm_batch(batch)
    assert torch.isfinite(root).all()
    assert torch.allclose(root[1], torch.eye(3) * 2)


# --- rank 22: MC loss shapes ---------------------------------------------------------------


def test_mc_loss_terms_are_per_cell(trained):
    module = trained.module.eval()
    tensors = _tensors(trained, 16)
    with torch.no_grad():
        single = _loss_out(module, tensors, n_samples=1)
        mc = _loss_out(module, tensors, n_samples=5)
    assert mc.n_obs_minibatch == 16
    assert mc.reconstruction_loss["reconst_loss_gene"].shape == (16,)
    assert mc.kl_local["kl_div_back_pro"].shape == (16,)
    assert mc.reconstruction_loss_sum.shape == ()
    # scalar loss stays on the scale of the single-sample loss (same data, resampled z)
    assert float(mc.loss) == pytest.approx(float(single.loss), rel=0.2)


# --- rank 23: docstring ----------------------------------------------------------------------


def test_mc_docstring_does_not_claim_a_jensen_debias():
    doc = differential_abundance.__doc__
    assert "corrects the Jensen-gap" not in doc
    assert "different scales" in doc


# --- ranks 24-27: load, checkpoint, sample_info ------------------------------------------------


def _fresh(adata=None, **kwargs):
    scvi.settings.seed = 0
    adata = _make_adata() if adata is None else adata
    MrTotalVI.setup_anndata(adata, **SETUP)
    return adata, MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, **kwargs)


def test_sample_blind_override_drops_u_encoder_penalty_and_resaves(tmp_path):
    adata, model = _fresh(u_encoder_sample_embedding_prior_scale=1.0)
    model.save(tmp_path / "m", save_anndata=False)
    with pytest.warns(UserWarning) as caught:
        loaded = MrTotalVI.load(
            tmp_path / "m",
            adata=adata,
            u_encoder_mode_override="sample_blind",
            allow_semantic_override=True,
        )
    assert any("switched off" in str(w.message) for w in caught)
    assert loaded.module.u_encoder_sample_embedding_prior_scale is None
    loaded.save(tmp_path / "m2", save_anndata=False)
    reloaded = MrTotalVI.load(tmp_path / "m2", adata=adata)
    assert reloaded.u_encoder_mode == "sample_blind"


def test_load_on_subset_without_a_donor_works_and_train_refuses(trained, tmp_path):
    adata = trained.adata
    trained.save(tmp_path / "m", save_anndata=False)
    subset = adata[adata.obs["sample"] != "donor_3"].copy()
    loaded = MrTotalVI.load(tmp_path / "m", adata=subset)
    assert loaded.get_local_sample_representation().shape[1] == 4
    with pytest.raises(ValueError, match="no cells"):
        loaded.train(max_epochs=1, accelerator="cpu")


def _registered_with_empty_donor():
    """Registered AnnData where no cell carries the code of donor_3."""
    adata = _make_adata()
    MrTotalVI.setup_anndata(adata, **SETUP)
    adata.obs.loc[adata.obs["_scvi_sample"] == 3, "_scvi_sample"] = 0
    return adata


def test_fresh_model_with_an_empty_donor(trained):
    with pytest.raises(ValueError, match="scale_observations"):
        _fresh_registered(_registered_with_empty_donor(), scale_observations=True)
    model = _fresh_registered(_registered_with_empty_donor())
    with pytest.raises(ValueError, match="no cells"):
        model.train(max_epochs=1, accelerator="cpu")


def test_load_on_other_cell_count_warns_when_penalty_active(tmp_path):
    adata, model = _fresh(sample_embedding_prior_scale=0.5, scale_observations=True)
    model.save(tmp_path / "m", save_anndata=False)
    with pytest.warns(UserWarning, match="penalty strength"):
        MrTotalVI.load(tmp_path / "m", adata=adata[:200].copy())


def test_sample_info_follows_sample_order():
    adata = _make_adata()[::-1].copy()
    _, model = _fresh(adata)
    assert list(model.sample_info["sample"]) == list(model.sample_order)
    assert model.sample_order[0] != adata.obs["sample"].iloc[0]


# --- rank 28: seed combiner ----------------------------------------------------------------


def _seed_ds(offset):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 3)) + offset
    return xr.Dataset({"lfc": (["draw", "feature"], values)}, attrs={"schema_version": "1"})


def test_seed_combiner_warns_on_degenerate_inputs():
    with pytest.warns(UserWarning, match="identical"):
        combine_mrtotalvi_seed_results({0: _seed_ds(0.0), 1: _seed_ds(0.0)})
    with pytest.warns(UserWarning, match="Only one training seed"):
        combine_mrtotalvi_seed_results({0: _seed_ds(0.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        combine_mrtotalvi_seed_results({0: _seed_ds(0.0), 1: _seed_ds(1.0)})


# --- rank 30: validation encoders reuse the training codes ---------------------------------------


def test_validation_dataset_reuses_training_codes():
    from tests.test_datamodule import _make_adata as make_dm_adata

    train = make_dm_adata(seed=1)
    valid = make_dm_adata(seed=2)
    valid = valid[valid.obs["sample"] != "donor_0"].copy()  # donor_0 has training code 0
    kw = {"protein_expression_obsm_key": "protein_expression", "require_protein_names": False}
    train_ds = _InMemoryProteinMappedDataset([train], ["sample"], **kw)
    plain = _InMemoryProteinMappedDataset([valid], ["sample"], **kw)
    shared = _InMemoryProteinMappedDataset([valid], ["sample"], encoders=train_ds.encoders, **kw)
    first = valid.obs["sample"].iloc[0]
    assert plain.encoders["sample"][first] != train_ds.encoders["sample"][first]
    assert shared.encoders["sample"][first] == train_ds.encoders["sample"][first]
    valid.obs.loc[valid.obs_names[0], "sample"] = "donor_new"
    with pytest.raises(ValueError, match="training"):
        _InMemoryProteinMappedDataset([valid], ["sample"], encoders=train_ds.encoders, **kw)


def _fresh_registered(adata, **kwargs):
    return MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, **kwargs)
