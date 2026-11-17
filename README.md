# mrtotalvi

MrTotalVI is TOTALVI (Gayoso et al. 2021) with the sample-aware two-level latent space of
MrVI (Boyeau et al. 2024), for CITE-seq RNA + protein data.

- `u`: per-cell state from an encoder conditioned on sample (`u_encoder_mode="sample_conditioned"`).
- `z = z_base(u) + eps(u, sample)`: sample-specific residual from an attention module.
- The TOTALVI decoder reads `z`, batch and library size.

Built on upstream `scvi-tools` (>= 1.5.2). The code derives from scvi-tools' TOTALVI and MrVI
implementations (BSD-3, see `LICENSE`).

## Migrating from the `mdmanurung/scvi-tools` fork

- Change `from scvi.external import MrTotalVI` to `from mrtotalvi import MrTotalVI`.
- MrMultiVI and CytoANVI are not included; they stay in the fork.
- Checkpoints saved by the fork store only the class name `MrTotalVI`, so they load with
  `mrtotalvi.MrTotalVI.load`. The test suite loads the fork's oracle checkpoint this way.
- `tests/legacy_oracle/` holds a frozen regression oracle produced by the fork (commit `d8c8e997`).
  The test suite replays it under upstream scvi-tools on CPU. Its generator script
  (`generate_legacy_oracle.py`) imports the fork and is kept, checksum-sealed, as provenance only.
- `tests/engineering/run_bounded_human_comparator.py` needs private study data and a git checkout; it
  is not collected by pytest.

## Install

```bash
pip install -e ".[test]"
pytest
```

## Use

```python
from mrtotalvi import MrTotalVI

MrTotalVI.setup_anndata(
    adata,
    layer="counts",
    protein_expression_obsm_key="protein",
    sample_key="donor_timepoint",
    batch_key="batch",
)
model = MrTotalVI(adata, sample_key="donor_timepoint")
model.train(n_epochs_kl_warmup=40)  # pass warm-up to train(); plan_kwargs values are overwritten
u = model.get_latent_representation(give_z=False)
z = model.get_latent_representation(give_z=True)
```

See `docs/user_guide.md` and the design records in `docs/adr/`.

## License

BSD 3-Clause, inherited from scvi-tools.
