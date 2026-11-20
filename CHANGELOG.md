# Changelog

## Unreleased

### Fixed

- Small input and doc fixes (no numbers change). Counterfactual methods reject a `random_state`
  outside `[0, 2**32)` or of a non-integer type, and an empty `indices` selection, before
  inference. A transfer `adata` missing the protein-names `uns` key now warns that the protein
  order is unchecked. `get_local_sample_distances(groupby=...)` raises on missing labels. The
  seed combiner warns when an entry is NaN in only some seeds. The private registry adapter keeps
  a pandas Categorical's category order, as `setup_anndata` does. Docstrings and the user guide
  now state: uniform LFC weights over every registered batch; NaN (not `-inf`) for a group with
  no reference under `"posterior"`; an analytic `kl_u` for the standard prior; nearly identical
  Vamp components without `init_prior_from_data`; and the kNN small-sample bias.
- `learn_z_u_prior_scale=True` with `use_map=True` now warns at construction: the optimizer
  shrinks `pz_scale` and `eps` together until the `-4` clamp, which removes most of the donor
  residual (`eps` RMS 0.003 against 0.085 by default in one synthetic run). Training is
  unchanged (user decision, 2026-11-20). The test that asserted `pz_scale >= -4` after
  training is fixed; the clamp acts where the scale is used, not on the parameter.
- `get_normalized_expression` on backed AnnData with non-ascending `indices` returned rows in
  sorted order labelled in request order (upstream behaviour the earlier backed fix missed); it
  now returns them in request order. `get_feature_correlation_matrix(log_transform=True)` raised
  `AttributeError` (upstream reads the undefined `self.n_genes`); it now uses the gene count.
- The 512 MiB guard of `get_counterfactual_latent` and `get_counterfactual_expression` sized
  only the output. Centering builds residuals for every registered sample in each minibatch, so
  300 registered samples, one target, 256 cells and 32 draws passed an estimate of 5.5 MiB and
  used about 10 GB. The estimate now adds that working set (attention rows sized from the `qz`
  layers, about 1.6 times the measured peak), so such requests raise before inference.
- Counterfactual expression without a registered panel (the panel is the batch):
  `batch_policy="specified"` decoded in the specified batch but masked proteins by the cell's
  observed batch, so a protein never measured in the specified batch got a value from decoder
  weights the masked likelihood never trained. The mask now follows the specified batch, and a
  conflicting `specified_panel` raises (user decision, 2026-11-20).
- A second `train()` call, or `train()` after `load`, ran with the module in eval mode:
  scvi-tools leaves it there after a fit and after load, and Lightning >= 2.6 no longer
  switches it back. Dropout was off and the `u_invariance_weight` penalty was skipped.
  `train()` now sets train mode first, so every fit runs like the first (user decision,
  2026-11-20). Continued-training results change; stock TOTALVI behaves the old way.
- Private DE path with `donor_key`: the design dropped the first donor's column and has no
  intercept, so that donor's residual level leaked into the fixed-effect `beta` (a pure patient
  offset of 3 gave a time `beta` of 0.9 or -0.3, by donor order). Every donor now gets its own
  offset column, as upstream MrVI adds batch offsets (user decision, 2026-11-20). `beta`,
  `effect_size`, p-values and LFC change whenever `donor_key` is set.
- Regressions from the 2026-11-20 fixes: a boolean `indices` mask on backed AnnData raised
  `IndexError` (it was passed to `np.unique`); a `residual_prior_mode="factual"` checkpoint
  loaded with `hierarchy_mode_override="legacy"` could not be reloaded (the override now resets
  the mode to `"raw_mean"`, with a warning); and an old seed result combined with a new one
  raised "different estimands" (a setting present in only some results is now reported as
  unverifiable).
- The seed combiner also compares the technical context: counterfactual expression records
  `specified_library_size` and a digest of the marginal reference cells, and enrichment records
  a digest of its reference cells and `max_reference_cells_per_sample`. Seeds run with library
  size 1 and 1000 used to pool into one between-seed SD.
- New known-answer donor-recovery test: 8 donors with planted RNA effects (log-ratio
  -2.8 to 2.8), package defaults, 100 epochs. Donor distances must rank the planted effects
  with an exact donor-permutation p <= 0.01 on each of training seeds 0, 1 and 2. Untrained
  models fail it (p = 0.07, 0.23, 0.77); the `*_wiring` tests do not.
- Two opt-in training flags; the defaults are unchanged (user decisions, 2026-11-20).
  `residual_prior_mode="factual"` (centered_v2 only) scores the factual centered residual
  instead of the mean over raw residuals, which made a sample-private residual about `S - 1`
  times cheaper than in `legacy` (ADR-0007 amendment). `anneal_protein_likelihood=False`
  stops scaling the protein likelihood by `kl_weight` during warm-up, as the TOTALVI paper
  describes; upstream TOTALVI scales it, so at `kl_weight=0` the protein term gave no
  gradient. Both are saved with the model; older checkpoints load with the defaults. Neither
  is validated on real data.
- `get_local_sample_representation` and `get_local_sample_distances` take an opt-in
  `admissibility=` boolean `(cell_name, sample)` mask, for example
  `get_outlier_cell_sample_pairs()["is_admissible"]`. Unsupported entries become NaN, group
  means average each sample pair over the cells with support in it, and
  `<column>_n_cells` holds those counts. Without the argument the output is unchanged, as
  upstream MrVI (user decision, 2026-11-20).
- `combine_mrtotalvi_seed_results` now compares `specified_panel` and the enrichment settings
  `group_key`, `contrast` (in numerator, denominator order) and `donor_key` across seeds and
  raises on a mismatch; before, a reversed contrast pooled opposite log-ratios into one mean.
  `local_sample_enrichment` records these settings. Results written before this change lack
  them: they still combine, with a warning that the settings cannot be checked (user decision,
  2026-11-20).
- `train()` warns when it overrides a `plan_kwargs` value (`lr`, `n_steps_kl_warmup`,
  `n_epochs_kl_warmup`, `reduce_lr_on_plateau`, `adversarial_classifier`). The precedence is
  unchanged and follows TOTALVI: the `train()` argument wins.
- `MrTotalVI.load_query_data` and `MrTotalVI.minify_adata` now raise `NotImplementedError`
  pointing to `docs/user_guide.md`. The inherited `load_query_data` loaded a query with new
  samples, froze the sample tables and left the new samples with random embeddings while
  the model reported `is_trained=True`. `minify_adata` failed with `KeyError: 'X_latent_qzm'`.
- `gene_dispersion="gene-label"` is rejected in `MrTotalVI.__init__` with a `ValueError`.
  It used to build and then fail in training with an `AttributeError`.
- `differential_abundance(omit_original_sample=False)` (also `donor_key` and
  `density_estimator="scale_floor"`) now removes each query cell's own Normal component from its
  own sample's aggregated posterior (log space, renormalised by `n_s - 1`), as
  `local_sample_enrichment` and `get_outlier_cell_sample_pairs` already do. Before, every cell
  scored as enriched for its own group: on a 10-D null stub the mean own-group enrichment was
  +89.4 nats (`"posterior"`), +12.3 (`scale_floor=0.4`) and +51.7 (`donor_key`); it is now
  near 0. **This is a deliberate departure from upstream MrVI** on a non-default path. The
  default `omit_original_sample=True` outputs are unchanged (pinned by a golden-file test).
  A one-cell own sample has no leave-one-out density: its per-sample `log_probs` for that cell
  are NaN and it carries no weight in group aggregates.
- `differential_abundance` returns NaN, not `-inf`/`inf`, for group log probabilities and
  enrichments when a cell's group has no reference sample (a group made only of the cell's own
  sample under `omit_original_sample=True`, as `"knn"` already did), and warns once with the
  number of affected cells.
- `local_sample_enrichment`: a target sample with NaN density (a factual singleton) no longer
  makes `group_log_density`, `log_ratio` and the donor mean and median NaN for the cell. The
  group `logmeanexp` runs over the usable samples and divides by their count; the donor
  summaries skip NaN donors. A group or cell with no usable sample stays NaN. All-finite
  outputs are bit-identical to before.
- `differential_abundance(donor_key=...)`: donor centering uses `nanmean`, so a NaN one-cell
  own sample no longer turns every sample of that donor NaN in the covariate aggregates.
- `get_outlier_cell_sample_pairs`: the per-sample subsample is sorted, so leave-one-out
  masks the right component whatever order the loader keeps (same set, same seed). A
  registered sample with no cells in `adata` gets a NaN, inadmissible column instead of
  raising.
- `library_policy="sample_balanced_marginal"` raises a `ValueError` above 1024 distinct
  reference contexts (each is one decode pass) and asks you to subset
  `marginal_reference_indices`. It used to run one decode per distinct library size.
  Marginal scenarios no longer copy four per-cell arrays each (constant fields are
  read-only broadcast views), and the context hash is streamed one scenario at a time;
  `context_table_sha256` and decoded values are unchanged.
- Counterfactual and enrichment `indices` must be integers: a float such as `1.9` was cast to
  cell 1 without an error, and a string such as `"1"` was accepted. Whole-number floats such as
  `1.0` still work.
- With a numpy protein matrix and names in `adata.uns` (`protein_names_uns_key`), a reordered
  protein axis was accepted on transfer and on `MrTotalVI.load` and read against the
  registered order. Both now raise. scvi-tools already checked a DataFrame protein matrix.
- Two inherited TOTALVI methods are overridden to fix scvi-tools 1.5.2 bugs.
  `get_normalized_expression` with several `transform_batch` values and a numeric
  `library_size` multiplied the running RNA sum on every batch (row sums near 500500 at
  1000 for two batches); it now scales once. `get_feature_correlation_matrix` ignored its
  `adata` argument and always sampled the training data; it now uses `adata`. Default calls
  give the same output as before.
- `protein_dispersion="protein-label"` is rejected in `MrTotalVI.__init__` with a `ValueError`,
  like `gene_dispersion="gene-label"`. It built and then failed on the first forward pass.
- `get_outlier_cell_sample_pairs` applies its leave-one-out rule to one-component samples (a
  one-cell sample, or `subsample_size=1`), as decided by the user on 2026-11-20. Before, the
  cell matched its own single component, so it was admissible for its own sample with no
  independent reference, and its self-density entered the pooled threshold. The cell is now
  NaN and inadmissible for its own sample, and the sample adds no threshold score; the pooled
  threshold can change. Samples with 2+ cells in the mixture give the same output as before.
  Upstream MrVI keeps the self-match.
- New warnings, no output changes: the private DE path warns when a fixed covariate is not
  orthogonal to another design column (per-covariate Wald statistics then mix effects) and
  when `filter_inadmissible_samples=True` leaves rejected samples in the eps standardisation;
  `density_estimator="knn"` warns for groups whose expected neighbour count
  `knn_k * n_g / N` is below 1. The `knn` docstring no longer says pooled group scores avoid
  the pseudocount bias. `lfc_std` is documented as an upper bound on the SD of the
  batch-marginal LFC.
- `use_map=False`: the stochastic residual log-scale is clamped below at `log(1e-3)`, the
  scale floor of upstream MrVI (user decision, 2026-11-20). In float32 a log-scale below
  about -52 gave an infinite KL and below about -104 a zero scale that made `Normal` raise.
  Only values below log-scale -6.9 change; `use_map=True` (the default, required by
  `centered_v2`) is unaffected.
- The VampPrior `freeze_prior_after_init` docstring no longer says the prior landscape stays
  fixed: only the pseudo-inputs freeze, and the components move as the shared `qu` trains.
- The private, quarantined batch data module now rejects a validation collection whose gene
  `var_names` differ in order from training; it compared only the protein axis.
- Private `_differential_expression`: `pvalue` is the chi-square survival function in
  float64 (was `1 - cdf` in float32, which gave exactly 0 for large statistics). The
  Benjamini-Hochberg family (all cells times all fixed covariates) is unchanged and now
  documented.
- Counterfactual support and admissibility scoring is vectorised in chunks; scores are
  bit-identical to the per-pair loop.
- Backed AnnData: `get_latent_representation`, the counterfactual and enrichment encoders
  and `collect_u_posterior` returned rows in sorted order (scvi-tools sorts backed
  minibatches) while labelling them in request order, so non-ascending `indices` swapped
  cells. Backed requests now load each cell once in ascending order and map rows back;
  repeated indices also work. In-memory AnnData takes the same path as before.

### Changed (defaults now follow released MrVI in part)

- Defaults switched from candidate p03 to p05 (user, 2026-11-18): the `u` encoder has two layers
  (`DEFAULT_QU_KWARGS = {"n_layers": 2}`, MrVI's `encoder_n_layers`), and `train()` uses
  `lr=2e-3` with `reduce_lr_on_plateau=False`, as MrVI. Benchmark run mrtotalvi-uz-r1: both pass
  all 3 gates; the p03/p05 score gap (3.734 vs 3.721, 6 seeds) is within seed noise. New models
  save the resolved `qu_kwargs`; checkpoints saved with the one-layer default (`qu_kwargs=None`)
  load with one layer. `spec_defaults=False` also skips the two-layer default.
- `MrTotalVI` defaults: `n_latent=30` (was 20) and `n_latent_u=10` (was `None`, isomorphic `u`).
  `n_latent_u=None` still gives the isomorphic hierarchy.
- `qz_kwargs` default merges `stop_gradients_mlp=True` and `dropout_rate=0.03` (MrVI's
  `DEFAULT_QZ_ATTENTION_KWARGS`). Keys you pass override per key. New `spec_defaults` argument:
  `False` restores the old `qz` (and the one-layer `qu`).
- `MrTotalVI.train()` defaults: Adam `eps=1e-8` and `weight_decay=1e-8` (TOTALVI: 0.01 and 1e-6) through `plan_kwargs`
  (your `plan_kwargs` keys win). This also lets the legacy `scale_observations=True` loss train;
  with `eps=0.01` its gradients (divided by cells per sample) fell below Adam's `eps`.
- Old checkpoints load with the values they were saved with: models without `spec_defaults` in
  their saved `init_params_` rebuild the old `qz` settings. Not changed: KL warm-up, early stopping, batch size, gradient clipping, label supervision.
- Class docstring now documents `n_latent_u`, `z_u_prior`, `u_prior_scale`,
  `u_prior_mixture_k`, `use_map`, `scale_observations`, `qz_kwargs` and `qu_kwargs`.

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

- Inherited from upstream TOTALVAE: with `n_samples > 1` the library posterior is drawn with
  `ql.sample()`, so it carries no gradient. Training never passes `n_samples`, so only
  evaluation calls use it.
- Kept for upstream MrVI parity (user, 2026-11-20), now warned or documented: the private DE
  path's per-covariate Wald statistic mixes correlated covariates (sqrt(X^T M X) prefactor),
  rejected samples still enter the eps standardisation, and `lfc_std` averages per-batch
  variances.
- `library_policy="sample_balanced_marginal"` still keys each distinct library value as its own
  context; above 1024 contexts it now fails rather than running (no binning, by design).
- Test gaps: no known-answer test covers DA on the default `omit_original_sample=True` path,
  and the shared training helper passes `lr` through `plan_kwargs`, so it is ignored (now
  with a warning).
