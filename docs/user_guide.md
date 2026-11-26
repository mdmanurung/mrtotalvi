# MrTotalVI user guide

`MrTotalVI` extends TotalVI with the MrVI hierarchical sample latent:

1. `u` is the base representation used for integration and aggregated posterior
   statistics. It is sample-conditioned by default.
2. `z = z_base(u) + eps` is the sample-aware representation used for local
   counterfactual donor queries.

```{figure} _static/architecture.svg
:alt: Schematic of the MrTotalVI architecture, from counts and sample label through u and z to the RNA and protein likelihoods.
:width: 100%

MrTotalVI architecture. The encoder reads RNA counts, protein counts and the sample label and gives the cell state
`u`. An attention block over `u` and a learned sample embedding gives the residual `eps`, and `z = z_base(u) + eps`.
The totalVI decoder reads `z`, batch and library size. Batch also enters the encoder (not drawn). The dashed box
applies only with `protein_noise_model="ambient"` or `"ambient_isotype"`. The decoder and the RNA and protein
likelihoods are those of totalVI {cite:p}`Gayoso_2021`. The `u` to `z` hierarchy, with its attention block over a
sample embedding, is that of MrVI {cite:p}`Boyeau_2025`.
```

The model keeps the stock TotalVI decoder unchanged. The hierarchy changes the encoder and
latent prior, not the RNA/protein likelihood definitions.

Engineering coverage does not promote a prior, representation claim, DA/DE inference, streaming,
or new-sample inference.

```python
from mrtotalvi import MrTotalVI
```

## Setup

Register a `sample_key` for the donor or sample axis. RNA (the registered layer or `X`) and protein
(`obsm`) inputs must be finite, non-negative, integer-like **raw counts**. Setup exhaustively checks
all values before mutating AnnData state.

`labels_key` records annotations but does not change the default unsupervised objective. To opt into
label supervision, set `u_prior_supervision="labels"` and a finite positive
`u_prior_label_weight`; disclose that choice in every comparison.

```python
MrTotalVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression",
    sample_key="donor",
    batch_key="batch",
    labels_key="cell_type",
)

model = MrTotalVI(
    adata,
    sample_key="donor",
    n_latent=30,  # package defaults, shown explicitly
    n_latent_u=10,
    u_prior_supervision="none",  # default even when labels_key is registered
)
```

The historical behavior remains the default. Opt into the package-only v2
semantics explicitly:

```python
model_v2 = MrTotalVI(
    adata,
    sample_key="donor",
    n_latent=30,
    n_latent_u=10,
    hierarchy_mode="centered_v2",
    u_encoder_mode="sample_blind",
    use_map=True,
    z_u_prior=True,
)
```

`centered_v2` requires `use_map=True` and `z_u_prior=True`. The
`sample_blind` encoder bypasses biological sample embeddings while retaining
explicitly registered technical covariates. It does not change parameter names
or shapes.

## Prior Options

New MrTotalVI calls resolve `u_prior` to exactly `"standard"`, `"mog"`, or `"vamp"`; the default
is `"mog"` for continuity, not because it is scientifically preferred.

- The defaults are `n_latent=30` and `n_latent_u=10`, as in released MrVI: a lower-dimensional `u`
  is projected into `z`.
- `n_latent_u=None` gives the isomorphic hierarchy (`u` and `z` have the same dimensionality, no
  projection). Checkpoints saved with `None` keep it.
- The residual network follows MrVI: `qz_kwargs={"stop_gradients_mlp": True, "dropout_rate": 0.03}`.
  Keys you pass in `qz_kwargs` or `qu_kwargs` override these per key. `spec_defaults=False`
  restores the `qz` and one-layer `qu` built before `spec_defaults` existed. The `u` encoder has
  two layers, as in MrVI (`qu_kwargs={"n_layers": 2}`); pass `qu_kwargs={"n_layers": 1}` for one.
  Checkpoints saved while the default was one layer load with one layer.
- `u_prior="mog"` uses learned mixture logits, means, and scales.
- `u_prior="standard"` uses an analytic Gaussian prior over `u`.
- `u_prior="vamp"` uses pseudoinputs; data initialization is restricted to frozen training indices
  and records its seed and ordered-index digest.
- `u_prior_supervision="labels"` is the only label-conditioned mode and requires registered labels
  plus a finite positive weight. The default is `"none"` with weight `0.0`.
- `z_u_prior=False` omits the residual `eps` prior penalty while keeping the `u` prior.

Behaviour of the prior options that is easy to miss, all unchanged package defaults:

- **Vamp pseudo-inputs are encoded as donor 0.** In `u_encoder_mode="sample_conditioned"`
  the pseudo-inputs pass through the sample-conditioned `qu` with sample index 0, so the prior
  sits in donor 0's frame and the KL sends gradient to donor 0's conditioning parameters.
  `sample_blind` mode is unaffected. MrTotalVI also feeds batch 0 when `encode_covariates=True`.
  In MrTotalVI, `u_prior="vamp"` with `encode_covariates=True` and registered covariates raises,
  because the pseudo-inputs have no covariate values.
- **Vamp data initialization clusters raw counts.** `init_prior_from_data=True` runs k-means on
  the concatenated, untransformed RNA and protein counts of a random subsample of at most 10,000
  training cells, so centroids are likely dominated by library size and by the higher-count
  modality.
- **Label supervision does not match components to labels at initialization.** With
  `u_prior_supervision="labels"` and Vamp data initialization, k-means cluster `k` is assigned to
  label `k` without checking that the cluster contains that label's cells. An "Unknown" or
  "unassigned" label is treated as one more class and gets its own supervised component, because
  `labels_key` has no `unlabeled_category`. Supervision also replaces `u_prior_mixture_k` by the
  number of labels.

The deprecated `u_prior_mixture` boolean is accepted only for exact historical migration
combinations; contradictory combinations fail before module
construction.

`latent_distribution="ln"` is rejected because the additive `u -> z` hierarchy is Euclidean.

### Choosing a prior: unsettled

No prior is generally recommended. The analysis historically used to justify MoG is unrecoverable
and used non-authoritative geometry. One-dataset Vamp results are insufficient for promotion.
Prespecify prior choice, include MoG/Vamp sensitivity where relevant, and do not choose after seeing
the biological outcome. The following is an experimental sensitivity arm, not a preferred recipe.

The two flags are not gated the same way. `init_prior_from_data` applies only to `u_prior="vamp"`.
When enabled, initialization uses only the frozen training indices and persists the seed and
ordered-index digest; validation/test cells cannot influence pseudoinputs. `freeze_prior_after_init`
also exists for historical sensitivity work, but no combination of these flags is a recommended
scientific default.

```python
model = MrTotalVI(
    adata,
    sample_key="donor",
    u_prior="vamp",
    u_prior_mixture_k=20,
    init_prior_from_data=True,
    freeze_prior_after_init=True,
    use_batch_norm="none",
    use_layer_norm="both",
)
```

This code block is therefore an explicitly prespecified sensitivity arm. It is not evidence that
Vamp is preferable to MoG or standard Gaussian on a new dataset.

## Representations

Use `give_z=False` for the `u` representation and `give_z=True` for the donor-aware `z`
representation.

```python
u = model.get_latent_representation(give_z=False)
z = model.get_latent_representation(give_z=True)
local_z = model.get_local_sample_representation()
local_d = model.get_local_sample_distances()
```

`get_local_sample_distances` returns an `(n_cell, n_sample, n_sample)` array. Both local methods
check the output size first and raise above `max_gb=4.0` GiB (100,000 cells and 100 samples need
4 GB for the distances; 500,000 cells and 200 samples need 80 GB). To avoid the per-cell array,
pass `groupby` (an `adata.obs` column or a list of columns, as in `MRVI.get_local_sample_distances`)
with `keep_cell=False`. Distances are then computed per minibatch and only the group means are
kept, as `<column>_distances`:

```python
by_cluster = model.get_local_sample_distances(groupby="cluster", keep_cell=False)
by_cluster["cluster_distances"].sel(cluster="B")  # (sample_x, sample_y)
```

Both local methods report every cell-sample pair by default, as upstream MrVI. Pass
`admissibility=model.get_outlier_cell_sample_pairs()["is_admissible"]` to set pairs without
sample support to NaN; group means then average each sample pair over the supported cells, with
counts in `<column>_n_cells`.

`model.sample_info` has one row per registered sample, in `sample_order` (the embedding row
order); use `sample_order` when you need the labels.

The default `u` encoder is sample-conditioned, so do not describe legacy `u` as
sample-unaware or donor-neutral. `u_encoder_mode="sample_blind"` removes the
implemented sample-conditioning paths, but this software property alone does
not establish biological disentanglement. Use `z` for registered-sample model
transformations and local sample distances.

### Sample-level embedding (GloScope)

`get_sample_divergence` compares whole samples, in the style of GloScope {cite:p}`Wang_2024`. Each
sample is the distribution of its cells’ posterior-mean `z`. The default fits a Gaussian mixture per
sample and returns the symmetric KL divergence between every pair, as an `(sample_x, sample_y)` array.
`get_sample_embedding` runs classical MDS on that matrix, as GloScope’s `plotMDS` does.

```python
div = model.get_sample_divergence()  # dens="GMM", dist_metric="KL"
emb = model.get_sample_embedding(div, n_components=2)  # DataFrame indexed by sample
```

`z` holds both cell-state composition and within-state sample shifts, so the divergence reflects
both. `dens="KNN"` uses k-nearest-neighbour estimators (`k=50`) instead of a mixture, and
`dist_metric="JS"` gives the Jensen-Shannon divergence. The defaults follow GloScope 2.1.2, with two
departures: the mixture has full covariances (mclust `VVV`; GloScope uses `VVE`, which
scikit-learn lacks), and the `r=10000` Monte Carlo draws are made once per sample, not once per pair.
GloScope warns below 500 cells per sample; so does this method.

## Batch representation

By default the batch covariate is one-hot encoded, so every batch adds a column to the
u-encoder input and to each decoder layer. With many batches that becomes the dominant
input width. `batch_representation="embedding"` replaces the one-hot code with a single
learned embedding table shared by the u-encoder and the decoder:

```python
model = MrTotalVI(
    adata,
    sample_key="donor",
    encode_covariates=True,  # required for batch to reach the u-encoder
    batch_representation="embedding",
    batch_embedding_kwargs={"embedding_dim": 5},
)
model.train()
batch_vectors = model.get_batch_representation()  # (n_cells, embedding_dim); embedding mode only
```

The input width then grows by `embedding_dim` instead of by `n_batch`. Note
`encode_covariates` defaults to `False` on this model, and batch only enters the
u-encoder when it is `True`; the decoder uses the representation either way.

Per-batch *parameter tables* stay one-hot indexed in both modes — gene and protein
dispersion, `log_per_batch_efficiency`, the protein background prior and the library-size
priors. Those are lookup tables rather than network inputs, matching {class}`~scvi.module.VAE`.

`"one-hot"` remains the default and is bit-for-bit unchanged, so existing checkpoints are
unaffected. `"embedding"` changes the architecture, so a model trained with it cannot be
reloaded as `"one-hot"`.

## Opt-in centered counterfactual datasets

The v2 APIs are available only when `hierarchy_mode="centered_v2"`.
Residuals are evaluated for every registered sample, centered over that full
registry, and only then subset to requested targets.

```python
latent = model_v2.get_counterfactual_latent(
    indices=query_indices,
    target_samples=["donor_a", "donor_b"],
    inference_mode="posterior_mc",
    n_draws=32,
    random_state=0,
)

expression = model_v2.get_counterfactual_expression(
    indices=query_indices,
    target_samples=["donor_a", "donor_b"],
    gene_list=["IL7R", "LST1"],
    protein_list=["CD3", "CD14"],
    batch_policy="observed",
    panel_policy="observed",
    library_policy="observed",
)
```

The latent dataset names `u`, `z_base`, `eps_raw`, `eps_centered`, and `z`.
The expression dataset names RNA scale/rate and deterministic protein mixture
components, contributions, totals, batch efficiency, and panel availability.
Posterior Monte Carlo output retains raw `draw` values and adds posterior means
and quantiles.

Technical contexts are explicit:

- `observed` holds each query cell's factual batch, panel, extra covariates,
  and effective library fixed across targets. The effective library is the
  registered size factor when present, otherwise the observed RNA total under
  the default TotalVI setting, or the posterior log-normal mean when
  `use_observed_lib_size=False`.
- `specified` requires registered batch/panel labels and a positive library
  scalar or cell-aligned vector. Without a registered panel the panel is the batch, so a
  specified batch also sets the protein mask; a different `specified_panel` raises.
- `sample_balanced_marginal` weights biological samples equally and empirical
  joint batch/panel contexts within each sample. Batch and panel cannot be
  marginalized separately. Each distinct reference context (batch, panel and,
  for the library policy, each distinct library size) is one decode pass; more
  than 1024 contexts fail, so subset `marginal_reference_indices`.

In-memory output has a hard 512 MiB estimate including summaries and overhead. The estimate also
counts the per-minibatch residual pass, which covers every registered sample (centering needs
all of them) and grows with `batch_size`, draws and `target_chunk_size`; it overestimates the
measured peak by about 1.6 times. Requests above the limit must be reduced through explicit cell,
target, feature, `batch_size` or `target_chunk_size` subsetting. Public `zarr_path` /
`zarr_chunks` streaming is quarantined and fails before filesystem access.

All targets are limited to registered samples. These datasets are model-based,
non-causal transformations. Centering defines the reported decomposition; it
does not prove identifiability, biological neutrality, or superiority over the
legacy model.

For descriptive local densities:

```python
enrichment = model_v2.local_sample_enrichment(
    group_key="condition",
    contrast=("treated", "control"),
    donor_key="patient",
)
```

`donor_key` must name a column that groups the samples of one patient, with exactly one
`treated` and one `control` sample per patient. It cannot be the model’s own `sample_key`
(`"donor"` above), because each registered sample then belongs to its own group and the
pairing check raises.

Group densities use equal-sample `logmeanexp`. Only the factual reference mixture excludes the query cell.
For paired samples, `enrichment["donor_log_ratio"]` gives one log density ratio per donor.
`enrichment["donor_log_ratio_mean"]` gives each usable donor pair equal weight.

`differential_abundance(donor_key=..., omit_original_sample=False)` centers log densities within each donor.
Consider two conditions with one sample per donor per condition and finite densities throughout.
Write `r` for the vector of within-donor log density ratios at one query cell.
The grouped log enrichment equals `logsumexp(r / 2) - logsumexp(-r / 2)`.
Its sign can differ from `mean(r)`. Both summaries describe fitted local densities.

## Ambient protein noise: recommended setup with empty droplets

When the unfiltered droplet matrix is available, fit the ambient protein noise model with one pooled profile. Use `"ambient_isotype"` when the panel has isotype controls and `"ambient"` when it does not, and keep every other argument at its default.

```python
import numpy as np

from mrtotalvi import MrTotalVI
from mrtotalvi.preprocessing import estimate_adt_ambient

# One pooled ambient profile. raw_counts holds every unfiltered droplet with all features
# (RNA and ADT): droplets are selected on the whole matrix, as scAR does.
profile = estimate_adt_ambient(
    raw_counts,  # droplets x features
    raw_barcodes,  # barcodes of the rows
    cell_barcodes,  # every barcode called as a cell; never used as an empty droplet
    feature_names,  # names of the columns
    adt_features=protein_names,  # the ADT columns; the names on adata.obsm["protein"]
)

MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein", sample_key="donor", batch_key="batch")
model = MrTotalVI(
    adata,
    sample_key="donor",
    protein_noise_model="ambient_isotype",  # "ambient" without isotype controls
    ambient_profile=profile,
    isotype_names=isotype_names,  # every isotype control on the panel
)
model.train()

comp = model.get_protein_noise_decomposition(n_samples=25)
dead = comp["native"].max() < 1e-3  # proteins the fit switched off; leave them out
native = np.log1p(comp["native"])  # denoised protein, see "Reading protein values"
```

The evidence for each choice:

- **The ambient model.** On the ADR-0012 simulation, native log1p RMSE fell from 0.77 with stock totalVI to 0.29 with `"ambient"` and a pooled profile. A shuffled or uniform profile still reached 0.38 to 0.39, so most of the gain comes from modelling ambient at all.
- **Isotype controls.** On one macaque PBMC CITE-seq data set (137 ADTs, three training seeds), `"ambient_isotype"` switched off no protein, against 7 to 14 per seed for `"ambient"`. It recovered the published rhesus staining pattern of 18 of 30 antibodies at AUROC >= 0.8, against 7 to 9 for the totalVI protein output.
- **One pooled profile.** Per-well profiles added nothing on the ADR-0012 stress seeds, and per-capture profiles changed no marker in the macaque data.
- **Two public data sets.** On the PBMC data of Mulè et al. 2022 (53,201 cells, 83 antibodies and 4 isotype controls, three seeds, pooled and per-batch profiles), `"ambient_isotype"` switched off no protein, against 8 to 17 for `"ambient"`. On the 10x Genomics PBMC 5k and 10k data (13,112 cells, 14 shared antibodies and 3 isotype controls), no mode switched off a protein. There, `"ambient"` left less CD14 signal on lymphocytes than `"ambient_isotype"` (off-lineage to on-lineage ratio of mean `native` about 0.003 against 0.02). Lineage-marker AUROC against RNA-only labels was about 0.95 for CD4 and CD8 in every mode on both data sets. With a small panel and few isotype controls, compare both modes on a few known markers before choosing.

Leave these off. `seed_background_prior_from_ambient` and `protein_encoder_input="ambient_zscore"` made results worse on the ADR-0011 simulations. `learn_ambient_profile` gave no clear gain (see below). `native_form="excess"` did no better than the default in the macaque data.

One choice is untested on real data. The ADR-0011 simulations used `anneal_protein_likelihood=False`, while the macaque fits kept the default `True`; the two were not compared on real data, so the default stays.

These are results from simulations and three real data sets, not validated defaults: `protein_noise_model` still defaults to `"totalvi"`. The ambient protein noise models are experimental in 1.0.0, so their arguments and outputs can change in a later release.

## Ambient protein noise: isotype outputs

```{figure} _static/architecture_ambient-isotype.svg
:alt: Schematic of the protein noise model with protein_noise_model="ambient_isotype": ambient, nonspecific and native parts of the protein mean.
:width: 100%

Protein noise model with `protein_noise_model="ambient_isotype"`. A burden encoder reads the protein counts and
gives a per-cell ambient burden `c`; the ambient part is `c` times the profile from empty droplets. An isotype
encoder reads only the isotype control counts and gives the factor `h`; the nonspecific part is
`softplus(eta_b + h * Lambda)`, with a per-batch offset and a per-protein loading. The native part comes from the
totalVI decoder and is 0 for isotype controls. The priors shown are the defaults (`ambient_burden_prior=(0.0, 2.0)`
is a mean and a standard deviation on `log c`). `"ambient"` uses only the burden encoder and adds the ambient part
to the totalVI background. Two ideas come from earlier methods. scAR {cite:p}`Sheng_2022` estimates the ambient
profile from empty droplets and scales it per cell. dsb {cite:p}`Mule_2022` uses isotype controls to measure each
cell’s nonspecific background. The mixture and the decoder are those of totalVI {cite:p}`Gayoso_2021`.
```

With `protein_noise_model="ambient_isotype"`, the loss is the same for any mixing logit of an isotype control. Its foreground probability is not identified. `get_protein_foreground_probability` and the protein part of `get_normalized_expression` return `NaN` for isotype columns. `get_protein_noise_decomposition` drops them. Other modes are unchanged.

In `get_protein_noise_decomposition`, `observed` is the fitted expected total count (the mixture mean), not the measured count. `foreground_probability` is the decoder mixture weight `1 - pi`, not the probability of foreground given the observed counts.

### Learnable ambient profile (experimental)

`learn_ambient_profile=True` learns the ambient composition instead of fixing it at `ambient_profile`. It starts at the profile and keeps the cell-free droplets in the loss through the multinomial likelihood of their summed counts, so the profile needs `droplet_counts` (profiles from `estimate_adt_ambient` carry them). With many droplets this term holds the composition close to the droplet estimate. On the ADR-0012 simulation it gave no clear gain (native log1p RMSE 0.267 learned vs 0.282 fixed with per-batch profiles, 0.342 vs 0.341 pooled; 5 seeds), so the default stays off.

### Reading protein values

Read denoised protein from `native` in `get_protein_noise_decomposition`. It is the expected count with the ambient and nonspecific parts removed, on the raw count scale. Do not read expression from `observed` (it still holds ambient and nonspecific counts) or from `foreground_probability` (a mixture weight, see above).

```python
import numpy as np

comp = model.get_protein_noise_decomposition(adata, n_samples=25)
native = np.log1p(comp["native"])  # cells x proteins; isotype controls dropped
dead = comp["native"].max() < 1e-3  # proteins the fit switched off
```

For a depth-normalised readout, pass `size_factor="total_adt"`. Each cell’s raw protein total (all registered proteins, isotype controls included, measured proteins only) is divided by the median total over all cells in `adata`, and `native`, `ambient`, `nonspecific` and `observed` are divided by that factor. `foreground_probability` is unchanged, and a cell with a total of 0 raises `ValueError`. The factor comes from noisy raw totals, so it re-adds per-cell depth noise and does not remove composition-driven differences. In PBMC 5’ it cut a 3.45x run gap on CD3 to 1.54x. The default is `None`.

Three checks come from one real data set: macaque PBMC CITE-seq, 137 ADTs, three training seeds. They are not validated defaults.

- **Dead proteins.** A protein whose `native` is about 0 in every cell carries no signal in that fit. In that data set, `protein_noise_model="ambient"` switched off 7 to 14 of 137 proteins per seed (CD8 among them), and `"ambient_isotype"` none.
- **Rank metrics hide weak signals.** An AUROC on `native` can be high when the values are near 0 everywhere: there, the latent orders near-zero values by cell identity. Before calling a marker, also check its size, for example the fold change of mean `native` between groups or the native share `native / observed`.
- **Display transform.** `log1p` and `arcsinh(x / 1)` give the same picture. The cytometry cofactors 5 (mass) and 150 (flow) fold weak markers toward 0, because `native` values are counts, mostly below 150.

With `protein_noise_model="totalvi"`, `native` is the foreground excess over background, `background * (fore_scale - 1)` times the foreground probability. It is exactly 0 for a protein whose foreground scale the fit left at 1. Stock totalVI’s `get_normalized_expression(include_protein_background=False)` keeps the background rate inside the foreground component, so it is never 0. On the Mulè et al. data, `native` was 0 in every cell for 6 to 11 of 83 proteins per seed, all with low raw counts, and the set changed with the seed. A zero there means the fit learned no foreground for that protein.

In the macaque data set, the protein output of `protein_noise_model="totalvi"` (and of stock totalVI) moved every protein with one per-cell factor: in biaxial plots, unrelated markers rose together along one diagonal.

## Statistical APIs

MrTotalVI exposes aggregated-posterior and outlier utilities and descriptive DA:

```python
ap = model.get_aggregated_posterior()
da = model.differential_abundance(sample_cov_keys=["condition"])
outliers = model.get_outlier_cell_sample_pairs()
```

`differential_abundance()` takes an opt-in `density_estimator`. The default `"posterior"` is the
MrVI aggregated posterior and its output is unchanged. Per-cell posterior scales (about 0.06 per
dimension in one planted-effect test) can be far below the spacing between neighbouring cells
(about 0.5), and then the log densities are noise-dominated: sample pairs with no abundance
change scored between -3.8 and +19.7 nats. Two options change the smoothing scale.

```python
model.differential_abundance(
    sample_cov_keys=["condition"], compute_log_enrichment=True, density_estimator="knn", knn_k=50
)
model.differential_abundance(
    sample_cov_keys=["condition"], compute_log_enrichment=True, density_estimator="scale_floor", scale_floor=0.4
)
```

`"scale_floor"` raises every kernel scale to `max(scale, scale_floor)`. `"knn"` counts, among
each cell’s `knn_k` nearest reference cells in `u`, the cells of each sample or group; the score
is `log(k + 0.5) - log(n_reference)`. In a synthetic 2-D check
(`tests/test_da_density_estimator.py`) both recovered a planted depletion, with a per-cell null sd
of about 0.2-0.3 (`knn_k=50`, floor 0.4); the posterior estimator’s null sd there was about 10.
Under `"knn"` the per-sample
`log_probs` are comparable across samples of similar size within a cell, but not across cells or
with `"posterior"` values, so read the enrichments. Where a sample’s expected neighbour count
`knn_k * n_g / N` is below about 1, the score favours the smaller sample. Group scores share
this bias. The warning identifies groups with low expected counts. The neighbour method pools
reference cells, so samples with higher yields carry more weight within a group. The two kernel
estimators give each included sample equal weight. `donor_key` is available only for the kernel
estimators. See the `differential_abundance` docstring for the full mapping.

With `omit_original_sample=False` (forced by `donor_key`), `differential_abundance()` removes
each cell’s own kernel from its own sample’s density (leave-one-cell-out), unlike upstream MrVI.
Without this, every cell looks enriched for its own group. A group with no reference sample for
a cell (for example one sample per condition under the default `omit_original_sample=True`)
gives NaN, not `inf`, and a warning with the affected cell count.

`MrTotalVI.differential_abundance()` is descriptive and non-inferential. Its public
`differential_expression()` fails closed for both legacy and centered-v2 models. For biological
DE, aggregate raw counts at the donor-by-condition level and use a replicate-aware method such as
PyDESeq2, edgeR, or dreamlet:

```python
model.differential_expression(sample_cov_keys=["condition"])
# RuntimeError (abbreviated): use donor-pseudobulk PyDESeq2, edgeR, or dreamlet.
```

## Sample-embedding shrinkage (opt-in)

`eps` is read from a learned per-sample embedding table (`module.qz.embedding`). Nothing ties
the rows together except an sd-1 prior on `eps` itself, so `eps` can fit an arbitrary offset
per sample. In a planted-effect test, `z` separated two random halves of one sample (null AUC
0.62 to 0.86; stock TOTALVI gave 0.51). The MrVI paper has the same structure.

`sample_embedding_prior_scale=sigma` (default `None`, off) adds a hierarchical Gaussian
log-prior on that table:

```python
model = MrTotalVI(adata, sample_key="sample", sample_embedding_prior_scale=0.5)
```

The penalty is `sum_s ||e_s - mean_s e_s||^2 / (2 sigma^2)`. It centres on the across-sample
mean, so a shared offset costs nothing. A smaller `sigma` shrinks harder. The value is yours to
choose: no sigma has been calibrated, and the default stays off.

`u_encoder_sample_embedding_prior_scale` applies the same penalty, with its own sigma, to the
u-encoder tables when `u_encoder_mode="sample_conditioned"`: `qu.sample_embed` and the gamma and
beta tables of both conditional normalisations. It raises under `sample_blind`.

Scaling: the training loss is a per-cell mean, so the full-data objective is `n_cells * loss`.
The penalty is a global term in that objective, so the loss gains `penalty / n_cells` once per
step, where `n_cells` is the cell count of the AnnData passed to `MrTotalVI`. It is not
multiplied by the batch size, and `kl_weight` warm-up does not scale it. The penalty is also
reported as `kl_global`, which the training plan divides by `n_obs_training`. With a
`train_size` below 1, the penalty is relative to all cells, not the training split.

Four reporting details. First, with the legacy `scale_observations=True` each cell’s data term
is divided by its sample’s cell count while the penalty is not, so one sigma shrinks relatively more
strongly, by about the sample’s cell count (the gradient ratio on `qz.embedding.weight` rose by the sample’s cell count in a toy run).
`centered_v2` weights average to 1 and are unaffected. Second, `elbo_validation` reports the
penalty as `kl_global / n_obs_validation`, about `n_cells / n_obs_validation` times its
size in the loss (0.035 against 0.0035 nats in a toy run), and `get_elbo` leaves the penalty out.
Third, early stopping monitors `elbo_validation`, which is cell-weighted. With
`scale_observations=True` the training loss weights cells by donor, so the monitor is not the
loss being minimised. `model.history` then also holds `donor_balanced_loss_train` and
`donor_balanced_loss_validation`, the weighted loss the module trains on. They are logged only;
the monitor and the stopping rule are unchanged.
Fourth, `MrTotalVI.load` rebuilds the model on the AnnData you pass, so `n_cells` is that
AnnData’s size. A load on a different cell set changes the penalty strength for continued
training, and a `UserWarning` says so when the checkpoint stores the training count (models with
`scale_observations=True`).

Caveats. The table passes through a LayerNorm before attention, and the shared mean is not
penalised. A model can therefore grow the mean to make the shrunk differences matter less after
normalisation. Check the spread of the embeddings after training before relying on the setting.
`get_aggregated_posterior`-based abundance scores are separate and not changed by this option.

## Inherited training defaults

`MrTotalVI.train()` uses MrVI’s Adam `eps=1e-8`, `weight_decay=1e-8`, `lr=2e-3` and no plateau
schedule (`reduce_lr_on_plateau=False`; TOTALVI uses `lr=4e-3` with the schedule).
Pass `eps` and `weight_decay` in `plan_kwargs` (your keys win); pass `lr` and
`reduce_lr_on_plateau` to `train()`. The small `eps` matters for the legacy
`scale_observations=True` loss, whose gradients are divided by the cells per sample. Early
stopping monitors `elbo_validation`, which is cell-weighted even when
`scale_observations=True` weights the training loss by donor. The two can rank fits
differently. Two defaults act on the `u`/`eps` hierarchy, and
`train()` warns when either applies.

- **KL warm-up.** `n_steps_kl_warmup` defaults to `0.75 * n_obs` steps, while the default
  `max_epochs` shrinks as `8e6 / n_obs` above 20,000 cells. Above roughly 40,000 cells (about 37,500 at the default batch size of 256 and train
  fraction of 0.9) the schedule cannot finish, so `kl_weight` stops below 1 and the protein likelihood, `kl_u`,
  `kl_z` and the background KL are down-weighted for the whole run. Pass `max_epochs`,
  `n_steps_kl_warmup` or `n_epochs_kl_warmup` to make it finish. Pass these (and `lr`) to
  `train()` directly: `train()` overwrites the same keys inside `plan_kwargs` with its own
  defaults, so `plan_kwargs={"n_epochs_kl_warmup": 40}` has no effect. A `UserWarning` names
  every such ignored value.
- **Adversarial batch classifier.** It is on whenever a protein has all-zero counts in some batch
  and `override_missing_proteins=False`. It acts on `z = u + eps`, not on `u` alone, so
  it can penalise donor information in `eps` when donors are nested in batches. With the legacy
  `scale_observations=True` and `hierarchy_mode="legacy"` the per-cell ELBO is divided by the
  sample's cell count while the adversarial loss is not, so the adversary carries a much larger relative weight (inferred from the code, not measured). Pass
  `adversarial_classifier=False` to turn it off.
- **Opt-in objectives (unvalidated on real data).** For `centered_v2`, let `a` be the cell's
  factual centered residual and `S > 1` the number of registered samples. At fixed prior scale
  `σ`, the smallest excess `raw_mean` penalty is `||a||² / (2σ²(S - 1))`.
  `residual_prior_mode="factual"` costs `||a||² / (2σ²)`, exactly `S - 1` times that minimum.
  Both costs use zero residuals as the reference. [Derivation](adr/0007-mrtotalvi-v2-centered-counterfactuals.md#centered-hierarchy-and-training-penalty)
  `anneal_protein_likelihood=False` stops scaling the protein likelihood by
  `kl_weight` during warm-up; by default (upstream TOTALVI) it gets no gradient at
  `kl_weight=0`. Both change training; compare matched-seed fits before relying on either.

Protein scale in inherited methods. The loss fits protein counts with means
`per_batch_efficiency * rate_{back,fore}`. `posterior_predictive_sample` now applies that
efficiency (stock TOTALVI does not, and its protein samples are about `1/efficiency` too large).
`get_normalized_expression` still returns upstream’s normalised scale, without the efficiency,
so its protein values sit on a different scale from `get_counterfactual_expression`.

Other inherited TOTALVI methods are not re-derived for `u`/`eps`: `get_marginal_ll` raises, and
`get_normalized_expression` and `get_protein_foreground_probability` decode `z` with each cell’s
own donor `eps` (use the counterfactual APIs to change the donor).

### Empirical protein background prior

`empirical_protein_background_prior` keeps the scvi-tools default: `None`, which
enables the prior above 10 proteins. The model computes it at construction from all
registered cells, including any you hold out later. For held-out evaluation, pass
`empirical_protein_background_prior=False` or construct the model on the training cells
only. The inherited initializer SD shrinks with the number of sampled cells, as
`sqrt(sum(s_i^2)) / n`.

## v1 Limitations

These limitations are intentional cuts documented in ADR-0005.

**Differential expression:**

- MrTotalVI accepts only `use_vmap=False`; `use_vmap=True` raises before inference/statistics.

**Differential expression in one cohort.** The historical `store_lfc=True` path decodes
contrasts in the sample residual `eps = z − z_base(u)`. The default encoder can also carry
donor information in `u`. A historical comparison covered 10 donors and 12 cell types in
one schistosomiasis CITE-seq cohort. Reported Spearman correlations between residual-based
log fold changes and PyDESeq2 pseudobulk estimates ranged from −0.24 to +0.04 across
MRVI, MrTotalVI and MrMultiVI. These values describe rank agreement in that cohort.
The historical method is private. Use a replicate-aware donor-pseudobulk analysis
for biological inference.

**Differential abundance across training seeds.**
In one experiment with 38 donor-timepoint samples (schistosomiasis DTP
cohort, 3 seeds), MrTotalVI DA varied from −9.0 to +9.7 across seeds (per-seed: s0 = +2.74,
s1 = −9.04, s2 = +9.67; std 9.46 >> mean 1.12). Those fits stopped before KL warm-up finished
(`kl_weight` 0.23–0.42). Refits with a completed 40-epoch warm-up and the current unsupervised
default prior gave +0.80, +0.68, +0.40 (mean +0.63 ± 0.20). The joint change leaves
the contribution of each setting unresolved. If using
`differential_abundance` with a `sample_key` that expands the sample count significantly (e.g.
`sample_key="donor_timepoint"`), do not interpret the package output as inferential evidence.
Use an independently validated, replicate-aware compositional method for biological DA.

**ArchesMixin / scArches surgery:**
MrTotalVI does not support reference-query surgery or the `ArchesMixin` protocol.
`MrTotalVI.load_query_data` and `MrTotalVI.minify_adata` raise `NotImplementedError`
(the inherited versions left new samples with random, frozen embeddings, or failed with
a `KeyError`). `gene_dispersion="gene-label"` is rejected at construction with a
`ValueError`; use `"gene"` or `"gene-batch"`. `protein_dispersion="protein-label"` is
rejected the same way; use `"protein"` or `"protein-batch"`.

**Streaming and new-sample inference:**
No data-module class is shipped. The private `_MrTotalVIRegistryAdapter` (`mrtotalvi._datamodule`) is not a
stable public training API, and this package does not claim end-to-end
streaming training or inference for biological samples absent from the registered training set.

**Differential abundance with unequal groups (methods note):**
The default `"posterior"` estimator is noise-dominated (see the `differential_abundance`
docstring), and its group scores use a `logmeanexp` that behaves like a maximum, so a group of
more samples scores higher. With unequal group sizes the sign of the default log-enrichment can
track the number of samples per group (null stub, 2 against 8 samples: mean about -18.7 nats).
This matches the MrVI formula. Use `density_estimator="knn"` or `"scale_floor"` for unbalanced
designs. The `"knn"` per-sample scores are comparable across samples of similar size only
(fixed pseudocount 0.5, see the docstring). With `n_mc_samples > 1` the score is the log expected
density under `q(u|x)`, a smoothed quantity on a different scale from the `n_mc_samples=1`
plug-in; do not mix the two.

**Admissibility (methods note):**
`get_outlier_cell_sample_pairs` (`is_admissible`) uses one global quantile over pooled
leave-one-out scores. `get_counterfactual_latent` (`admissible`) uses one quantile per target
sample. They are different quantities. In both, a cell’s own sample is scored leave-one-out, so
about `quantile_threshold` (default 5%) of own-sample pairs fail by design, and the private DE
filter drops that share of cells from their own sample. A one-cell sample (or
`subsample_size=1`) has no leave-one-out reference: its cell is NaN and inadmissible for its own
sample, and that sample adds nothing to the pooled threshold.

**Objective (methods note):**
The default training objective (`use_map=True`) is a MAP/cross-entropy penalty on the residual
`eps`, not a strict ELBO. For the legacy hierarchy, the analytic ELBO — including the `q(eps)`
entropy term — is available via `use_map=False` at model construction. Centered v2 requires
`use_map=True`.

```python
model = MrTotalVI(adata, sample_key="donor", use_map=False)
```

`get_elbo` is comparable only within one configuration (`use_map`, `hierarchy_mode`, prior
scales and KL weights): with `use_map=True` the `eps` term is `-log p(eps)` at a point and can be
negative. The loss clamps the `eps` prior log-scale below at `-4.0`
(`pz_scale.clamp(min=-4.0)`). A `z_u_prior_scale` under `-4` therefore acts as `-4`, and with
`learn_z_u_prior_scale=True` a scale that reaches `-4` receives no gradient.

The `kl_u` term is analytic for `u_prior="standard"`. For the mixture priors (`"mog"`,
`"vamp"`) it is a single-sample Monte-Carlo estimate of `KL(q(u)‖p(u))` (unbiased, higher
variance than analytic).

`differential_abundance` computes no test. The Wald statistic in `differential_expression` (private; the public method refuses) is compared with a Chi²
distribution whose degrees of freedom equal the number of admissible samples per cell, as in the
MrVI reference. That choice has not been calibrated here and is not shown to be conservative: the
statistic sums squared standardized coefficients over the `n_latent` dimensions, so with fewer
samples than latent dimensions the p-values can be anti-conservative. Treat the p-values as
descriptive until a null simulation supports them.

The design matrix (`_construct_design_matrix`) has no intercept. The response `eps` is mean-centred
per cell across samples before regression, but the design columns are not centred. A constant term
is therefore not unidentifiable. Instead, each coefficient estimates the contrast against the mean
of all samples, not against a reference group: for one binary covariate with `n0` samples at the
reference level and `n1` at the other, beta is `n0/n` times the group difference, so it is half of
the difference in a balanced design and changes with the choice of reference level. This matches
the MrVI reference and is unchanged. The effect sizes, decoded LFCs and Wald statistics inherit
that scale.
