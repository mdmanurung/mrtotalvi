# Changelog

## Unreleased

### Added

- `differential_abundance(density_estimator=..., knn_k=50, scale_floor=None)`: opt-in calibrated
  density estimators. `"knn"` scores cells by neighbour counts in `u` (pseudocount 0.5,
  sample-size corrected, a cell never counts itself or, with `omit_original_sample=True`, its own
  sample). `"scale_floor"` floors the aggregated-posterior kernel scales. The default
  `"posterior"` and its outputs are unchanged. Non-default runs set
  `attrs["density_estimator"]`.

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
