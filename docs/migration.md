# Migrating from the scvi-tools fork

MrTotalVI used to live in the `mdmanurung/scvi-tools` fork as `scvi.external.MrTotalVI`.

- Change `from scvi.external import MrTotalVI` to `from mrtotalvi import MrTotalVI`.
- MrMultiVI and CytoANVI are not included; they stay in the fork.
- Checkpoints saved by the fork store only the class name `MrTotalVI`, so they load with
  `mrtotalvi.MrTotalVI.load`. The test suite loads the fork's oracle checkpoint this way.
- `tests/legacy_oracle/` holds a frozen regression oracle produced by the fork (commit `d8c8e997`).
  The test suite replays it under upstream scvi-tools on CPU. Its generator script
  (`generate_legacy_oracle.py`) imports the fork and is kept, checksum-sealed, as provenance only.
- `tests/engineering/run_bounded_human_comparator.py` needs private study data and a git checkout; it
  is not collected by pytest.
