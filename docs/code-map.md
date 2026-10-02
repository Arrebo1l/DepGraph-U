# Code and artifact map

| Location | Purpose / status |
|---|---|
| `patches/showo2/generation.patch` | CADD dispatch and logits bypass; uninstrumented execution-cost variant against pinned FlashU; historical opt-in branches retained |
| `patches/showo2/understanding.patch` | Attribution-study KV / projection variant with CUDA probes; separate from generation |
| `implementations/showo2_understanding/understanding.py` | Reconstructed FlashU understanding policy, not official FlashU code |
| `implementations/showo2_understanding/lmms_eval_wrapper.py` | Historical evaluation/calibration integration; requires corresponding eval stack and data |
| `patches/lumina/` | Image vocabulary slice and corresponding local output indices |
| `implementations/omnigen2/cadd_adapter.py` | Guarded refiner dispatch and Taylor controls; C is reference, D adds CADD |
| `implementations/bagel/generation_dispatch.py` | Unused image-CFG prefill removal without TaylorSeer |
| `implementations/bagel/runtime_adapter.py` | Attributed FFN pruning and native-KV append support; requires model integration |
| `examples/omnigen2_pair.py` | New portable C/D paired entry point; clean GPU execution unverified |
| `scripts/prepare_showo2.py` | Fresh local clone plus one pinned variant; verifies output hashes |
| `scripts/verify_release.py` | Release hashes and independent compact evidence checks, CPU only |
| `scripts/summarize_results.py` | Readable tables regenerated from JSON |

## Evidence entry points

| Question | Starting artifact |
|---|---|
| Complete vs. native / fixed-policy generation | `results/summary.json` |
| Understanding responses and 65,828 decode calls | `reproducibility/evidence/table_data.json`, `trace_examples.csv`, `verify.py` |
| MMStar correction | `reproducibility/evidence/mmstar_correction_20260924/` |
| Component transfer | `reproducibility/evidence/main_tables/Tables_5_6_SOURCE_DATA.json` |
| BAGEL no-Taylor diagnostic | `reproducibility/evidence/current_revision/data/bagel_no_taylor/` |
| Guards / execution cost / controlled reuse | `reproducibility/evidence/current_revision/` |
| Figures and source identity | `reproducibility/evidence/figure_provenance.json`, `provenance/` |

`reproducibility/figs/` contains native PDF figures; `source_figures/` contains available archived editable assets, not every figure in every authoring format. `tables/` contains corrected compact-revision TeX fragments. Historical table numbering and internal method names are preserved.
