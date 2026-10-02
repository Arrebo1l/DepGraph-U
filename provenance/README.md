# Publication provenance

`artifacts.json` records original and public SHA-256 hashes for copied artifacts.
The submitted PDF is byte-identical to the author's final file. Scientific
records are copied from the compact evidence revision, with the MMStar correction
explicitly identified in `docs/errata.md`.

Public-copy transformations remove private local paths from metadata/logs,
comment nonportable editable-install paths in environment records, and refresh
the affected public hash references. Original hashes remain in `artifacts.json`.
Scientific scalar values, sampled outputs, and timing values are not edited by
these transformations. The newly constructed summary separately selects the
corrected MMStar aggregation.

`release_manifest.json` hashes the reviewed publication files, excluding itself.
It provides accidental-change detection, not an independent signature or proof
that historical measurements occurred. `scripts/verify_release.py` checks it
before recomputing the compact evidence.

For a reviewed maintenance update, stage intended files and run
`python3 scripts/update_release_manifest.py`, then review/stage the new manifest.
Do not regenerate manifests to hide unexplained evidence changes.
