# MMStar aggregation correction

`paper/DepGraph_U.pdf` is preserved unchanged, SHA-256:

```text
8fc51e5b9295686f33af39770054721d67d81fca28a9ea2a3c6c35e067e520f2
```

That PDF reports an unweighted mean over MMStar L2 category accuracies in affected cells. The intended overall metric is correct responses / 1,500 examples. Because L2 categories have unequal sizes, these aggregations differ.

The correction recomputes **12,000 archived answer decisions across eight arms**. No model was rerun. Predictions, answer judgments, timings, equality counts, and speedups are unchanged.

| Arm | Historical L2 macro (%) | Correct / 1,500 | Corrected overall (%) |
|---|---:|---:|---:|
| Exact-DLS reference | 51.73660 | 757 | 50.46667 |
| Exact-DLS | 51.73660 | 757 | 50.46667 |
| Stable cap-128 reference | 51.66549 | 754 | 50.26667 |
| Stable cap-128 | 51.91904 | 759 | 50.60000 |
| Legacy reference | 51.78930 | 758 | 50.53333 |
| Legacy Stable | 51.65821 | 758 | 50.53333 |
| BAGEL native | 68.01347 | 1,015 | 67.66667 |
| BAGEL + adapted system | 66.69248 | 1,000 | 66.66667 |

Source: [RESULTS.json](../reproducibility/evidence/mmstar_correction_20260924/RESULTS.json) and adjacent `responses.jsonl` / `rescore.py`. See [MMStar's official evaluator](https://github.com/MMStar-Benchmark/MMStar/blob/main/eval/vlmeval/evaluate/mmstar.py).

Run `python3 scripts/verify_release.py` to independently recompute the correction. `results/summary.json` uses corrected MMStar scores and retains historical values separately. Files named `*_historical_results.csv` remain historical records.

`reproducibility/` comes from the corrected compact evidence revision. Its table assets are not a source bundle guaranteed to reproduce the submitted 36-page PDF. The PDF and evidence versions are deliberately identified separately.
