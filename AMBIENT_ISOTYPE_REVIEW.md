# Ambient-isotype review

Keep `ambient_isotype` experimental for native protein estimates. Ambient-only had lower native error in all 15 matched comparisons against the default isotype arm. On three absent markers, the isotype arm assigned a median 58.6% of fitted signal to native protein. [Held-out benchmark figure](docs/_static/adr0012/ambient_block.png)

One isotype control measures the sum of ambient protein and nonspecific binding. Target counts also enter the nuisance encoder. The scaled-expression getter includes unidentified control weights in the denominator for biological targets.

The full assessment covers the equations, held-out results, reproduced API failures, and 60 passing CPU tests. [Read the full local assessment](docs/reviews/2026-11-24/ambient-isotype-assessment.md)
