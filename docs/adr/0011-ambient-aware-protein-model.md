# ADR-0011: Opt-in ambient- and isotype-aware protein noise model

## Status

Proposed on 2026-11-22. The number 0011 avoids the fork's ADR-0010 (see the
provenance note in ADR-0009).

## Context

MrTotalVI keeps the stock totalVI protein likelihood. Its background is one
log-normal rate decoded from `z`, with a learnable prior that scvi-tools 1.5.2
seeds from a per-cell GMM fit: one scalar per panel, copied to every protein.
Two facts motivate an extension:

- The background is decoded from `z = z_base + eps`, so a sample counterfactual
  moves it (`_module.py`, `_deterministic_decoder_parameters`).
- Empty droplets measure the free-ambient ADT composition directly (scAR), and
  isotype controls track each cell's nonspecific background (DSB, PMID 35440536).

## Decision

All modes are opt-in. With the defaults below a model is unchanged: no new
parameters, the same loss path, the same outputs.

| Argument | Default | Opt-in values |
|---|---|---|
| `protein_noise_model` | `"totalvi"` | `"ambient"`, `"ambient_isotype"` |
| `protein_background_source` | `"z"` | `"z_base"` |
| `protein_encoder_input` | `"raw"` | `"ambient_zscore"` |
| `seed_background_prior_from_ambient` | `False` | `True` |
| `ambient_profile` | `None` | an `AmbientProfile` from `mrtotalvi.preprocessing.estimate_adt_ambient` |
| `isotype_names` | `None` | isotype features on the protein axis |
| `native_form` | `"scale"` | `"excess"` (`S = exp(alpha) (f - 1)`, isotype mode) |
| `background_mixing_reg` | `1e-3` | any weight for totalVI's mixing pull (opt-in modes) |

`ambient_profile` may also be a dict of one `AmbientProfile` per registered batch.

### Generative model

With `A = c_i a` (free ambient; `a` the pooled profile, `log c_i ~ N(0, 2)`),
`B` the decoded totalVI background, `f >= 1` the decoded foreground scale and
`pi` the decoded background probability:

- `"ambient"`: `mu_back = eff (A + B)`, `mu_fore = eff (A + B f)`. The totalVI
  background prior on `B` is kept.
- `"ambient_isotype"`: `N = softplus(eta_b + lambda h_i)`, `h_i ~ N(0, 1)` (1-D),
  native `S = exp(alpha) f`. `mu_back = eff (A + N)`, `mu_fore = eff (A + N + S)`.
  Isotype features get `S = 0`, so they are fit as `NB(eff (A + N))`. The totalVI
  background prior is dropped, because no term carries its meaning.
- `"z_base"` decodes `B` (its log mean and SD) from `z_base(u)` instead of `z`.

`eff` (`log_per_batch_efficiency`, prior `LogNormal(0, 1)`) multiplies every
component (C7). The prior is applied per cell, as in stock totalVI: each cell adds
`-log p(eff)` for its batch, and the loss takes the mean over cells, so a batch's prior weight
grows with its cell count. It is not counted once per shared parameter. `q(log c)` reads `log1p` of all protein counts and the log total;
`q(h)` reads the isotype block and the log total only.

Posterior component means on the raw count scale come from
`MrTotalVI.get_protein_noise_decomposition`: `native = eff (1 - pi) S`,
`ambient = eff A`, `nonspecific = eff B` or `eff N`, and `observed` is their sum.
Isotypes are dropped from its columns, and unmeasured panel entries are `NaN`.
`observed` is the fitted mixture mean, and `foreground_probability` is the decoder mixture weight `1 - pi`, not P(foreground | y).
In `ambient_isotype` mode the foreground probability of isotypes is unidentified, so the inherited getters return `NaN` for them.

### Calibration (`mrtotalvi.preprocessing.estimate_adt_ambient`)

This follows scAR `setup_anndata`: `prob=0.995`, `min_raw_counts=2`,
`iterations=3`, one pooled profile, no shrinkage. It fails below 50 cell-free
droplets. It also reports DSB step-I statistics on two scales: `log1p` seeds the
background prior, and `log(x + 10)` z-scores the optional encoder input.

### Decisions

| ID | Decision | Source |
|---|---|---|
| Q3 | Opt-in `z_base` background; default unchanged (D1) | user; D1 default |
| C1 | Seed the background prior per protein from empty droplets | user (DSB abstract) |
| C2 | `log1p` scale for the seeded prior | user; scvi `_get_totalvi_protein_priors` |
| C3 | Seeded prior stays learnable | user; scvi 1.5.2 |
| C4 | One pooled ambient profile | user; scAR |
| C5 | scAR selection `prob=0.995`, `min_raw_counts=2`, `iterations=3` | user; scAR |
| C6 | No small-batch shrinkage | user; scAR |
| C7 | Efficiency multiplies every component | user; inherited |
| C8 | Benchmark seeds 0-4 per arm | user |
| C9 | `log c_i ~ N(0, 2)` | user |
| C10 | 1-D `h_i ~ N(0, 1)` | user |
| C11 | `ambient_isotype` without isotypes raises | user |
| C12 | z-score pseudocount 10 | user; dsb `R/dsb.r` |
| C13 | Sparse gate decoded from `z`, no prior | user |
| C14 | Isotypes off the target outputs | user; see deviation 2 |
| D2 | `ambient_zscore` is blocked with `u_prior="vamp"` | default |

### Deviations found during implementation

1. **scAR's cutoff needs the whole transcriptome.** scAR keeps droplets with
   multinomial `log_prob >= log(prob) * n_features`. On an ADT-only matrix of
   about 20 features no droplet passes. `estimate_adt_ambient` therefore selects
   droplets on all features passed (RNA and ADT), as scAR does, and profiles the
   `adt_features` subset. ADT-only input fails with an explicit error.
2. **Isotypes stay on the registered protein axis.** The likelihood treats them
   as background-only controls, and they are dropped from the public component
   outputs. A separate registry field (C14 as approved) would change
   `n_input_proteins` and checkpoint shapes. That is left for a later change.
3. **PR 5 adds no new likelihood.** With `G ~ Bernoulli(omega)` and `omega`
   decoded from `z` (C13), `NB(A + N + G S)` marginalises to
   `(1 - omega) NB(A + N) + omega NB(A + N + S)`. That is the
   `"ambient_isotype"` mixture with `pi = 1 - omega`, so no separate mode is
   added.

### Not yet supported in opt-in modes (fail closed)

`get_counterfactual_expression`, `differential_expression`, the `u` invariance
penalty and any other path through `_deterministic_decoder_parameters` raise
`NotImplementedError`. Counterfactual decoding with `c_i` and `h_i` held fixed
is the next step.

## Simulation benchmark (tusoskill run, 50 iterations)

Run root: `/exports/para-lipg-hpc/mdmanurung/tusoskill_runs/mrtotalvi_ambient_20261122`.
The run used the planted simulator `tests/_simulate_adt.py` with 5 training seeds
per arm. The primary metric is the RMSE of log1p native means against the
planted values. Figures are under `reports/figures/`, and every candidate is in
`experiments/results.jsonl`.

- **Annealing the protein likelihood is the largest single factor.** With the
  repository default `anneal_protein_likelihood=True`, the ambient arms lose a
  low-abundance induced marker in 2 of 5 seeds. With `False`, the leader recovers
  it in most seeds (`benchmark_metrics.png`).
- **The leader is `protein_noise_model="ambient"`,
  `anneal_protein_likelihood=False`, unseeded prior, pooled profile.** It beats
  the matched baseline (A0, no annealing) on native RMSE on all five data seeds:
  0.37 vs 0.54, 0.19 vs 0.41, 0.31 vs 0.39, 0.26 vs 0.36 and 0.25 vs 0.40
  (`benchmark_metrics.png`, `final_checks.png`).
- **It stops ambient from looking like a donor effect.** When stim donors carry
  3x more ambient, the baseline reports a spurious |log2 FC| of 0.19-0.21 on
  null markers; the leader reports 0.02-0.04 (`stress_tests.png`,
  `controls.png`).
- **The gain needs the right profile.** A shuffled or uniform profile, or the
  ambient term switched off, falls back to baseline performance. 300 empty
  droplets are enough (`controls.png`).
- **`z_base` costs nothing but adds no accuracy.** Keep it for the
  counterfactual estimand.
- **Not recommended:** seeding the prior from empty droplets (C1), because
  empties carry far less ambient than cell droplets; the DSB z-score encoder
  input (C12); and a 2-D `h` (C10 alternative). The isotype mode is competitive
  but not better than the ambient mode.
- **Open failure:** one seed in each of two data sets still loses the induced
  marker (`final_checks.png`).

## Consequences

Default models, the legacy oracle and saved checkpoints are unaffected. Opt-in
models save their configuration in `init_params_` and rebuild the same
parameters on load. Promotion of any opt-in mode needs the benchmark gate in
`tusoskill_runs/mrtotalvi_ambient_20261122` and a real-data check on raw
droplets, which is still pending (raw paths not yet supplied). The simulation
results above are not evidence for real data until that check runs.
