# Scope and limitations

## Scientific scope

- Rewrites follow model-specific audited dependencies. This is not an automatic compiler that proves arbitrary computation removable.
- Exactness is relative to the named fixed policy and tested implementation. Generation approximation may already change native-model outputs.
- Exact-DLS speedups include last-position projection. 80.8% of examples end before incremental reuse; cache compatibility alone does not imply large gains.
- Stable-Graph changes the policy and may change predictions, quality, and length. Short output limits, long tails, and baseline choice affect latency.
- Native-KV controls and an unpruned MMMU counterexample remain in the evidence. A weak no-KV reference must not replace a strong native-KV deployment baseline.
- Transfer has architecture-dependent headroom. BAGEL's isolated added generation gain is negligible; its subset cannot establish a full-benchmark quality improvement.
- Most measurements are batch-one A800 experiments. Consumer-GPU fit, serving throughput, training, and universal hardware speedups are not established.

## Measurement and reproduction

- Synchronized timing and instrumented profiling are separate. Nested-exclusive CUDA stream intervals can include launch gaps; they are not pure kernel-time sums.
- Show-o2's 35 Euler timepoints produce 34 model calls: 14 CADD hits and 20 surviving trunk calls.
- Zero-hit measurements cover paired model calls, not a pipeline overhead guarantee. Generation guards do not certify all understanding KV/gather/mask/EOS boundaries.
- OmniGen2 outputs match within recorded pairs. An additional cross-process bitwise check failed after recovery despite matching weights. Hardware/process segments are retained; global cross-process identity is not claimed.
- Lumina's 2×2 ablation records RNG resets but does not export poststates.
- DPG scoring corrects an upstream loader that skipped the first data row. Both arms were rescored over all 1,065 images and 14,392 raw answers per arm. Earlier incomplete scores are not final.
- Early pilot expansion gates and unsuccessful attempts remain in the record. Full benchmark results are not extrapolated from pilots.

## Release completeness

Included: source adaptations, protocol records, checks, selected per-example records, and figure/table assets. Excluded: checkpoints, complete upstream repositories, benchmark image datasets, and most full-resolution generated-image collections.

CPU checks validate supplied records and arithmetic. They cannot recreate omitted images or establish that every historical GPU run was performed exactly as reported. A fresh end-to-end GPU reproduction of every benchmark is not certified for this packaging. The portable example is a new launcher around the recorded adapter; clean-environment GPU execution remains unverified.

Protocols retain historical selection/stopping rules and templated deployment paths. They are records to inspect, not executable configurations for the new example. Environment snapshots may include obsolete packages or commented local installs; they are not a universal installation lock.
