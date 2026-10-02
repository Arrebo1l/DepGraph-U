# Pinned source adaptations

Apply patches only to their recorded source revisions, in a separate checkout.
Do not combine the two Show-o2 variants: they represent different experiment
integrations and modify the same model file.

- `showo2/generation.patch`: uninstrumented generation execution-cost source.
  `showo2/manifest.json` identifies its upstream commit and expected output hashes.
- `showo2/understanding.patch`: KV/logits attribution-study integration, with
  recorded CUDA event instrumentation. `understanding_manifest.json` identifies
  the base commit, target hash, and additional reference-policy module.
- `lumina/lumina_head.patch` and `lumina_generator.patch`: apply both to Lumina-DiMOO
  commit `81344e60545ae8d70f11039d3450717d974f831b`, using
  `git apply --ignore-space-change` after the corresponding `--check`.
  This handles the pinned generator's CRLF context. Both patches were checked
  against the recorded source snapshot.

These patches preserve historical implementation names. Extra opt-in experiments
in the Show-o2 generation snapshot are not part of the main reported method;
leave them disabled. See `docs/reproduction.md` for controls and validation scope.
