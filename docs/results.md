# Results and comparison scopes

Speedup means **reference mean latency / candidate mean latency** within the stated paired experiment. Do not multiply speedups from different controls or present fixed-policy equality as equality to an unmodified model.

## Show-o2 generation

The complete system combines inherited approximation policies with new execution rewrites. Its 2.88–3.21× native-model comparison includes quality changes. The fixed-policy comparison holds the chosen approximation trajectory constant.

| Benchmark | Scale | Fixed-policy reference (s) | + rewrites (s) | Speedup | Paired quality |
|---|---|---:|---:|---:|---:|
| GenEval | 1.5B | 5.783 | 3.312 | 1.746× | 0.69241 → 0.69241 |
| GenEval | 7B | 6.568 | 3.718 | 1.767× | 0.55546 → 0.55546 |
| DPG-Bench | 1.5B | 2.973 | 1.844 | 1.612× | 83.666 → 83.666 |
| DPG-Bench | 7B | 6.575 | 3.723 | 1.766× | 84.355 → 84.355 |

Source: `results/summary.json`, `fixed_policy_generation`. Historical `table` values 1/2 identify GenEval/DPG, not necessarily the submitted PDF's table numbers. Native-model rows are in `complete_system_vs_native`.

GenEval uses a 0–1 score; DPG uses 0–100. Equality is on tested paired outputs, not a proof for new inputs, devices, kernels, or precision settings.

## Show-o2 understanding

The denominator is the **reconstructed FlashU understanding implementation**: the selected FFN / dynamic-layer / visual-token policy, without added KV reuse and last-position projection. We reconstruct the path described by FlashU because it was not included in the pinned public generation implementation. This is not an official released FlashU understanding binary.

Exact-DLS preserves the selected dynamic-layer policy. It matches 24,769 decoded responses; token-sequence comparisons are available for 20,781 examples. Its 1.15–2.34× result includes last-position projection as well as graph-compatible KV reuse. The projection/KV attribution experiment separates these effects.

Stable-Graph disables dynamic graph changes. It can change predictions, quality, and output length. Its paired references, RNG control, and some runs differ from Exact-DLS; do not infer a controlled Exact-versus-Stable comparison by subtracting their rows. Main MMBench-EN/MMStar Stable results use fresh paired cap-128 runs. Old cap-1024 results are diagnostic only.

Main compact MME scores are perception scores; MME latency includes perception and cognition. MMBench uses official circular evaluation, not internal row accuracy. MMStar uses [corrected overall accuracy](errata.md).

## Component transfer

Lumina-DiMOO uses the same ML-Cache policy in both arms. The head patch computes only the image vocabulary slice; global sampled token IDs remain unchanged. OmniGen2 compares CADD to TaylorSeer with last-layer-only history/prediction already enabled in both arms.

| Model | Benchmark | Reference → ours (s) | Quality in both arms | Speedup |
|---|---|---:|---:|---:|
| Lumina-DiMOO | GenEval | 29.311 → 27.599 | 0.87753 | 1.062× |
| Lumina-DiMOO | DPG-Bench | 29.598 → 27.888 | 84.140 | 1.061× |
| OmniGen2 | GenEval | 10.695 → 9.581 | 0.76254 | 1.116× |
| OmniGen2 | DPG-Bench | 10.826 → 9.693 | 84.203 | 1.117× |

GenEval: 553 prompts × 4 seeds = 2,212 pairs. DPG: 1,065 prompts, one image per prompt per arm, not the four-image leaderboard protocol. The first-data-row loader correction and independent scorer records are retained.

BAGEL already omits image-call text projection and supports native KV. A 12-prompt, three-repeat diagnostic separates inherited FFN pruning from unused image-guidance prefill removal. Full-system 1.107× is not the isolated rewrite benefit (~1.0001× alone; ~1.0013× on the same FFN policy). The 0.750 → 0.833 subset score reflects one additional passing image. BAGEL MMStar covers 1,500 inputs: 1.154×, 1,157 matching responses, and corrected quality 67.667% → 66.667%.
