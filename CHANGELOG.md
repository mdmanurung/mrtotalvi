# Changelog

## Unreleased

### Added

- `differential_abundance(density_estimator=..., knn_k=50, scale_floor=None)`: opt-in calibrated
  density estimators. `"knn"` scores cells by neighbour counts in `u` (pseudocount 0.5,
  sample-size corrected, a cell never counts itself or, with `omit_original_sample=True`, its own
  sample). `"scale_floor"` floors the aggregated-posterior kernel scales. The default
  `"posterior"` and its outputs are unchanged. Non-default runs set
  `attrs["density_estimator"]`.
- Opt-in `sample_embedding_prior_scale` and `u_encoder_sample_embedding_prior_scale` on
  `MrTotalVI`: a Gaussian shrinkage log-prior `sum_s ||e_s - mean_s e_s||^2 / (2 sigma^2)` on
  the per-sample embedding tables that feed `eps` (and, separately, the sample-conditioned
  u-encoder). Both default to `None` (off); loss and parameters are unchanged then. The term
  enters the loss as `penalty / n_cells`. See `docs/user_guide.md`.

### Fixed

- `get_local_sample_distances` and `get_local_sample_representation` check the output size before
  encoding and raise above `max_gb=4.0` GiB (`max_gb=None` turns the check off). Distances are
  computed per minibatch. New `groupby` and `keep_cell` arguments return group-mean matrices
  (`<column>_distances`) without the per-cell array.
- `MrTotalVAE.sample` (used by `posterior_predictive_sample`) applies `per_batch_efficiency`, as the
  loss does. Protein samples were about `1/efficiency` too large.
- The Wald-statistic matrix square root maps non-finite entries to 0, so a cell with a singular
  `X^T M X` no longer aborts the private DE run with a `ValueError` from the Chi2 CDF.
- `module.loss` with Monte-Carlo samples (`n_samples > 1`) returns per-cell terms of shape
  `(batch,)` and `n_obs_minibatch=batch`. The scalar loss is unchanged; `compute_elbo` no longer fails
  on stacked terms of different shapes.
- `MrTotalVI.load(..., u_encoder_mode_override="sample_blind")` switches the u-encoder shrinkage
  penalty off with a warning, in the module and in `init_params_`. A re-saved checkpoint failed to
  load.
- `MrTotalVI.load` accepts an AnnData that lacks some registered donors (inference works). A fresh
  model with an empty donor raises at construction only with `scale_observations=True`, and
  otherwise at `train()`. The message no longer tells users to edit `sample_key`.
- `MrTotalVI.load` warns when the sample-embedding penalty is active and the loaded AnnData has a
  different cell count from the training run, because `penalty / n_cells` changes.
- `model.sample_info` follows `sample_order`, as in MrVI, instead of the order of first appearance.
- `combine_mrtotalvi_seed_results` warns on a single seed, fewer than two draws, or identical results
  under two seed keys (a between-seed SD of 0).
- The private registry adapter gives validation datasets the training categories' codes and raises
  on unseen categories. Validation codes used to be re-derived from the validation set alone.

### Changed

- The adversarial-classifier warning and the user guide say the classifier acts on `z = u + eps`,
  "not on `u` alone". The default u-encoder is sample-conditioned, so "sample-free u" was wrong. The
  adversary target itself is unchanged.
- The user guide names the real class (`_MrTotalVIRegistryAdapter`, private) instead of the
  non-existent `MrTotalVIBatchDataModule`.

### Documented

- Directional bias of the default `"posterior"` DA estimator with unequal group sizes, the `"knn"`
  per-sample pseudocount bias for small samples, and the scale of `n_mc_samples > 1` scores
  (docstring corrected: it is not a Jensen debias).
- The `eps` prior log-scale clamp at `-4.0`; `get_elbo` comparability within one configuration; the
  penalty under `scale_observations=True`; the validation-metric scaling of the penalty.
- The two admissibility rules (`is_admissible` against `admissible`) and the leave-one-out
  own-sample failure rate.

## 0.1.0.dev0 (2026-11-17): extraction from the `mdmanurung/scvi-tools` fork

MrTotalVI moved from `scvi.external.mrtotalvi` in the fork to the standalone package `mrtotalvi`,
built on upstream `scvi-tools` 1.5.2. MrMultiVI and CytoANVI stay in the fork.

The fork's frozen regression oracle (`tests/legacy_oracle/`, commit `d8c8e997`) replays under
upstream scvi-tools 1.5.2, torch 2.15 and pandas 3.1 on CPU within its portable tolerance. The
largest strict-tolerance delta is 1.9e-6, below the fork host's 3.8e-6.

### Fixed during extraction

- The DE design matrix treated pandas >= 3 string columns (dtype `str`) as numeric and crashed.
  Every non-numeric column is now dummy-coded (`_stats.py`).
- `get_latent_representation(give_mean=True)` crashed on GPU for `hierarchy_mode="centered_v2"`,
  because the sample index stayed on CPU (`_module.py::_gather_sample`).
- `get_local_sample_distances` built one `(n_cells, S, S, n_latent)` tensor; it now computes
  distances in cell chunks with the same result.
- `get_counterfactual_expression` decoded every draw x cell x target row at full gene and protein
  width in one call, which bypassed its 512 MiB gate. It now decodes in row chunks and keeps
  only the requested features.
- `get_outlier_cell_sample_pairs` scored a subsampled cell against its own posterior component,
  although the thresholds are leave-one-out. The own component is now excluded.
- `learn_z_u_prior_scale=True` ignored `z_u_prior_scale` as the starting value.
- `combine_mrtotalvi_seed_results` reported between-seed means and SDs of latent coordinates,
  which are not aligned across independently trained seeds. Those summaries are now skipped and
  listed in `attrs["between_seed_skipped_latent_variables"]`.

### Known open issues (from the 2026-11-17 Fable and Opus reviews; not fixed)

- Paired DA (`donor_key`) keeps each cell's own component in its own sample's density.
- `centered_v2` weakens the residual prior by about (S - 1) in sample-private regions.
- The protein likelihood is scaled by `kl_weight` (inherited from upstream TOTALVI).
- MrVI's residual-network defaults (`stop_gradients_mlp=True`, `dropout_rate=0.03`) are not applied.
- `train()` overwrites `lr` and KL warm-up passed through `plan_kwargs`; pass them to `train()`.
- Local sample distances are not masked for sample support.
- Outputs assume dataloader row order equals `indices` order; backed AnnData can break this.
- `library_policy="sample_balanced_marginal"` runs one full decode per distinct library value.
- The seed combiner does not compare `specified_panel` or the enrichment contrast settings.
- Test gaps: the hierarchy tests pass on an untrained model, no known-answer test covers DA on the
  default `omit_original_sample=True` path, and the shared training helper passes `lr` through
  `plan_kwargs`, so it is ignored.
