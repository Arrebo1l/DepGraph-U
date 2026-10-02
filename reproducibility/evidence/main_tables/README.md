# Current main-text evidence

These records accompany six refreshed main tables and Figure 6. They do not overwrite the historical appendix's `../table_data.json`, whose cap-1024 Stable results remain archived.

- `Tables_1_2_DISPLAYED_DATA.json`: reviewed Show-o2 generation values and unrounded means.
- `Tables_3_4_DISPLAYED_DATA.json`: Exact and Stable benchmark values; fresh paired cap-128 MMBench/MMStar for Table 4. Do not compare separate run segments as a controlled Exact-versus-Stable experiment.
- `Tables_5_6_SOURCE_DATA.json`: unrounded auxiliary generation quality/latency values; `DISPLAYED_DATA` holds rounded cells.
- `projection_kv_controls.csv`: all five original controls. The first four fixed-policy controls form Table 3(b), preserving the retained appendix's cross-reference.
- `figure6_mmstar_pairs.csv`: all 1,500 paired cap-128 observations, extracted after matching sample identities and generated token IDs. Reuse calls come from recorded step profiles. Figure footer ratios divide dataset latency statistics, not per-example ratio statistics.
- `SOURCE_INDEX.json`: anonymous upstream-record digests. Full GPU logs, model weights and evaluator outputs are separate preserved archives, not reproduced by this compact package.

Auxiliary generation is evidence for individual rewrites, with Show-o2 the main model. DPG uses one image per prompt and the corrected CSV loader retaining the first data row. Recovered runs span process/hardware segments. OmniGen2 within-process equality does not imply cross-process bitwise reproducibility. The Lumina 2x2 ablation recorded RNG resets, not poststates. These limits remain in the main text.
