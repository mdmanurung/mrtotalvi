# MrTotalVI review and proposed fixes

The review found that a gene permutation could pass silently. A boolean mask could also return another cell’s posterior. The audit covered `987efdf`, with follow-up ambient tests at `37bd399`. [Full assessment and evidence](docs/reviews/2026-11-23/assessment.md)

## Proposed repair order

1. **Preserve feature and cell identity.** Check ordered RNA and protein names during setup, transfer and load. Verify that masks and disk-backed reads return the requested cells in order. [Reproduced failures and acceptance checks](docs/reviews/2026-11-23/assessment.md#highest-priority-corrections)
2. **Freeze the training split before estimating priors.** Record training cell counts for every donor. Reject unsupported targets. Verify that changing held-out counts leaves the initial priors unchanged. [Training-split evidence](docs/reviews/2026-11-23/assessment.md#highest-priority-corrections)
3. **Match public predictions to the fitted likelihood.** Mark isotype foreground probabilities as undefined. Include structural zeros in expected RNA counts under ZINB. [Protein and RNA predictions](docs/reviews/2026-11-23/assessment.md#protein-noise-and-rna-predictions)
4. **Validate donor-level claims against independent replicates.** Use the chosen donor weights during validation. Score recovery of induced markers separately from average denoising error. Keep abundance scores descriptive until donor-level null experiments establish calibration. [Held-out protein results](docs/reviews/2026-11-23/heldout_protein.png) · [Abundance null experiment](docs/reviews/2026-11-23/abundance_null.png)

The assessment also proposes fixes for inference state, seed comparisons, memory use and checkpoint compatibility. [Engineering findings](docs/reviews/2026-11-23/assessment.md#other-engineering-fixes)
