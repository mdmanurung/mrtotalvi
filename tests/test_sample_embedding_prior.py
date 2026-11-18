"""Opt-in shrinkage prior on per-sample embedding tables (sample_embedding_prior_scale)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import scvi
import torch

import mrtotalvi
from mrtotalvi import MrTotalVI

ROOT = Path(__file__).resolve().parents[1]
MAIN_COMMIT = "b900c13"  # last commit before the prior existed
SETUP = dict(
    protein_expression_obsm_key="protein_expression",
    sample_key="sample",
    batch_key="batch",
)


def _adata(n_samples: int = 4, random_halves: bool = False):
    adata = scvi.data.synthetic_iid()
    if random_halves:
        rng = np.random.default_rng(0)
        ids = rng.permutation(adata.n_obs) % n_samples
    else:
        ids = np.arange(adata.n_obs) % n_samples
    adata.obs["sample"] = [f"s{i}" for i in ids]
    return adata


def _model(adata, **kw):
    MrTotalVI.setup_anndata(adata, **SETUP)
    return MrTotalVI(adata, sample_key="sample", n_latent=10, **kw)


def _first_batch(model, adata):
    loader = model._make_data_loader(adata, batch_size=64, shuffle=False)
    return next(iter(loader))


def _loss(model, tensors, seed=0):
    torch.manual_seed(seed)
    model.module.train()
    return model.module(tensors, loss_kwargs={"kl_weight": 1.0})[2]


# The script below runs under both the pinned main commit and this worktree.
ONE_STEP_SCRIPT = textwrap.dedent(
    """
    import hashlib, sys, warnings
    warnings.filterwarnings("ignore")
    import numpy as np, scvi, torch
    from mrtotalvi import MrTotalVI
    import mrtotalvi
    scvi.settings.seed = 0
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = [f"s{i % 4}" for i in range(adata.n_obs)]
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein_expression",
                            sample_key="sample", batch_key="batch")
    # Old defaults, passed explicitly: this test compares against the pre-change commit,
    # whose defaults differ from the current MrVI-matching ones.
    model = MrTotalVI(
        adata, sample_key="sample", n_latent=10, n_latent_u=None,
        qu_kwargs={"n_layers": 1},
        qz_kwargs={"stop_gradients_mlp": False, "dropout_rate": 0.0},
    )
    tensors = next(iter(model._make_data_loader(adata, batch_size=64, shuffle=False)))
    model.module.train()
    opt = torch.optim.Adam(model.module.parameters(), lr=1e-3)
    torch.manual_seed(0)
    loss = model.module(tensors, loss_kwargs={"kl_weight": 1.0})[2]
    loss.loss.backward()
    opt.step()
    h = hashlib.sha256(loss.loss.detach().numpy().tobytes())
    for k, v in sorted(model.module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    print(mrtotalvi.__file__); print(h.hexdigest())
    """
)


def _run_one_step(src: Path) -> tuple[str, str]:
    import os

    env = {**os.environ, "PYTHONPATH": str(src), "MPLCONFIGDIR": "/tmp/mplconfig"}
    out = subprocess.run(
        [sys.executable, "-c", ONE_STEP_SCRIPT],
        env=env, capture_output=True, text=True, check=True, timeout=600,
    ).stdout.split()
    return out[-2], out[-1]


def test_default_off_is_byte_identical_to_main(tmp_path):
    """(a) Loss and every parameter after one step match the pre-change commit exactly."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    archive = tmp_path / "main.tar"
    try:
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "-o", str(archive), MAIN_COMMIT, "src"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {MAIN_COMMIT} not reachable")
    old = tmp_path / "old"
    old.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(old)], check=True)
    old_file, old_hash = _run_one_step(old / "src")
    new_file, new_hash = _run_one_step(ROOT / "src")
    assert str(old) in old_file and str(ROOT) in new_file  # really two different sources
    assert old_hash == new_hash


def test_import_points_into_this_tree():
    assert str(ROOT / "src") in mrtotalvi.__file__


def test_penalty_term_equals_formula():
    """(b) The loss gains penalty / n_cells; the penalty equals the written formula."""
    adata = _adata()
    sigma = 0.3
    model = _model(adata, sample_embedding_prior_scale=sigma)
    # give the table real spread so the term is not ~0
    with torch.no_grad():
        model.module.qz.embedding.weight.normal_(0.0, 1.0)
    tensors = _first_batch(model, adata)
    on = _loss(model, tensors)
    model.module.sample_embedding_prior_scale = None
    off = _loss(model, tensors)

    w = model.module.qz.embedding.weight.detach().numpy().astype(np.float64)
    expected = ((w - w.mean(axis=0)) ** 2).sum() / (2 * sigma**2)
    n_cells = adata.n_obs
    assert expected / n_cells > 0.1  # term is far above float32 noise
    assert float(on.kl_global_sum.detach()) == pytest.approx(expected, rel=1e-5)
    # the loss is float32 and large, so allow a few units in its last place
    ulp = float(np.spacing(np.float32(abs(float(off.loss.detach())))))
    assert float((on.loss - off.loss).detach()) == pytest.approx(
        expected / n_cells, abs=4 * ulp
    )
    assert float(off.kl_global_sum) == 0.0


def test_shared_offset_is_not_penalised():
    adata = _adata()
    model = _model(adata, sample_embedding_prior_scale=0.5)
    pen = model.module.sample_embedding_penalty()
    with torch.no_grad():
        model.module.qz.embedding.weight += 3.0
    assert float(model.module.sample_embedding_penalty()) == pytest.approx(float(pen), rel=1e-5)


def test_u_encoder_tables_have_their_own_flag():
    adata = _adata()
    model = _model(adata, u_encoder_sample_embedding_prior_scale=0.5)
    assert model.module.sample_embedding_prior_scale is None
    qu = model.module.qu
    with torch.no_grad():
        qu.sample_embed.weight.normal_(0.0, 1.0)
    w = qu.sample_embed.weight.detach().numpy().astype(np.float64)
    others = sum(
        float(((t - t.mean(0, keepdim=True)) ** 2).sum())
        for t in (
            qu.cond_norm1.gamma_embedding.weight,
            qu.cond_norm1.beta_embedding.weight,
            qu.cond_norm2.gamma_embedding.weight,
            qu.cond_norm2.beta_embedding.weight,
        )
    )
    expected = (((w - w.mean(0)) ** 2).sum() + others) / (2 * 0.5**2)
    assert float(model.module.sample_embedding_penalty()) == pytest.approx(expected, rel=1e-4)
    # the eps table is untouched by this flag
    with torch.no_grad():
        model.module.qz.embedding.weight.normal_(0.0, 5.0)
    assert float(model.module.sample_embedding_penalty()) == pytest.approx(expected, rel=1e-4)


def test_invalid_settings_raise():
    adata = _adata()
    MrTotalVI.setup_anndata(adata, **SETUP)
    with pytest.raises(ValueError, match="positive"):
        MrTotalVI(adata, sample_key="sample", n_latent=10, sample_embedding_prior_scale=0.0)
    with pytest.raises(ValueError, match="sample_conditioned"):
        MrTotalVI(
            adata, sample_key="sample", n_latent=10,
            u_encoder_mode="sample_blind", u_encoder_sample_embedding_prior_scale=1.0,
        )


def _train_spread(prior):
    scvi.settings.seed = 0
    adata = _adata(n_samples=2, random_halves=True)
    kw = {} if prior is None else {"sample_embedding_prior_scale": prior}
    model = _model(adata, **kw)
    model.train(
        max_epochs=3, lr=1e-2, batch_size=32, accelerator="cpu",
        early_stopping=False, reduce_lr_on_plateau=False,
    )
    w = model.module.qz.embedding.weight.detach()
    return float((w[0] - w[1]).norm())


def test_prior_shrinks_embeddings_of_two_random_halves():
    """(c) Two samples that are random halves of the same cells end up closer with the prior."""
    without = _train_spread(None)
    with_prior = _train_spread(0.01)
    assert with_prior < 0.8 * without, (without, with_prior)


def test_save_load_keeps_setting(tmp_path):
    """(d) Both scales and the cell count survive save / load."""
    adata = _adata()
    model = _model(
        adata,
        sample_embedding_prior_scale=0.7,
        u_encoder_sample_embedding_prior_scale=1.3,
    )
    model.save(tmp_path / "m", overwrite=True)
    loaded = MrTotalVI.load(tmp_path / "m", adata=adata)
    assert loaded.module.sample_embedding_prior_scale == 0.7
    assert loaded.module.u_encoder_sample_embedding_prior_scale == 1.3
    assert loaded.module._n_obs_total == adata.n_obs
    assert loaded.init_params_["non_kwargs"]["sample_embedding_prior_scale"] == 0.7
    a = float(model.module.sample_embedding_penalty())
    assert float(loaded.module.sample_embedding_penalty()) == pytest.approx(a, rel=1e-6)

    default = _model(_adata())
    default.save(tmp_path / "d", overwrite=True)
    d = MrTotalVI.load(tmp_path / "d", adata=_adata())
    assert d.module.sample_embedding_prior_scale is None
    assert d.module.sample_embedding_penalty() is None
