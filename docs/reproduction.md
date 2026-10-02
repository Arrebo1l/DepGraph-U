# Reproduction guide

There are three distinct activities: checking supplied evidence, inspecting/adapting the recorded model code, and running fresh GPU experiments. Only the first is dependency-free and fully verified by release-time CPU checks.

## 1. Audit the compact evidence

From the repository root, with Python 3.10+:

```bash
python3 scripts/verify_release.py
python3 scripts/summarize_results.py
python3 -m unittest discover -s tests -v
```

Do not use `python -O`: historical evidence verifiers use assertions. The bundle includes archived-answer rescoring for all eight MMStar arms. No dataset download or GPU inference is required. Read [errata.md](errata.md) before comparing PDF MMStar values to the JSON summary.

## 2. Prepare a pinned Show-o2 source variant

```bash
git clone https://github.com/Rirayh/FlashU.git third_party/FlashU-upstream
python3 scripts/prepare_showo2.py \
  --upstream third_party/FlashU-upstream \
  --output third_party/showo2-generation \
  --variant generation
```

The tool creates a new local clone, checks out commit `189f877953387f8b7f1e3f4d63eddbc5b597da52`, applies the recorded patch, and verifies modified-file hashes. It does not modify the input checkout, download weights, install dependencies, or execute a benchmark.

Generation controls in the patched FlashU source:

| Variable | Meaning |
|---|---|
| `FLASHU_CACHE_AWARE_BYPASS=1` | Enable CADD dispatch on eligible cache hits |
| `FLASHU_SKIP_UNUSED_LOGITS=1` | Omit unused generation text projection |

Set both to `0` for the fixed-policy execution reference; use the same model, seed, sampling/approximation settings, precision, and evaluator in both arms. Do not enable unrelated historical experimental branches (for example TADA/UQP) when reproducing the main fixed-policy result. Model paths, metadata, device count, batch size, and calibration must be set explicitly in the upstream evaluation configuration. This source preparation is not a claim that its legacy CLI reproduces every final table's timing scope.

For the separate understanding attribution variant, choose a different output directory and `--variant understanding`. The model patch includes CUDA event probes and reads `FLASHU_ATTRIBUTION_LOGITS_MODE` (`all` or `last`) and `FLASHU_UNDERSTANDING_KV_CACHE` (`0` or `1`). `flashu/understanding.py` is installed alongside it. The archived lmms-eval wrapper illustrates calibration and per-sample policy configuration. This requires the corresponding evaluation integration and dataset setup; it is not a turnkey six-benchmark command. Do not stack generation and understanding patches on one checkout.

## 3. Optional fresh OmniGen2 paired example

**Status:** the recorded adapter comes from the GPU experiments; the portable launcher is new and has not been rerun on a clean GPU installation. Expect an A800-class memory footprint rather than assuming a consumer GPU fits the model.

The historical environment uses Python 3.12, PyTorch 2.6.0 + CUDA 12.4, Transformers 4.51.3, Diffusers 0.33.1, and FlashAttention 2.7.4.post1. Inspect `environments/omnigen2-recorded.txt`. Install dependencies in a dedicated environment appropriate for your CUDA/toolchain; the archived freeze is an environment record, not a cross-platform lockfile.

```bash
git clone https://github.com/VectorSpaceLab/OmniGen2.git third_party/OmniGen2
git -C third_party/OmniGen2 checkout 18e6f9d5271b517fcb32e999f10df943ae9b8f20
```

Obtain the exact model snapshot from Hugging Face (requires `huggingface_hub`):

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="OmniGen2/OmniGen2",
    revision="df5dca8a981d74e6c3af214c145f5c735fe72367",
    local_dir="checkpoints/OmniGen2",
)
```

Then run:

```bash
python examples/omnigen2_pair.py \
  --source third_party/OmniGen2 \
  --weights checkpoints/OmniGen2 \
  --output runs/omnigen2-example-001 \
  --prompt "A red apple on a white plate." \
  --seed 42 --repeats 3
```

The example verifies source revision and pinned weight/config hashes, including the scheduler code, then compares **C** (TaylorSeer with last-layer-only history/prediction) to **D** (same + CADD). It checks warmup output/RNG equality and a real CADD hit, alternates arm order, and times CUDA-synchronized pipeline calls with image writing/hashing excluded. All formal pairs and slower outcomes are retained. Output mismatches stop the run after preserving the records. It does not score GenEval/DPG or prove equivalence of every internal state.

The scheduler is loaded from the verified weight repository, whose defaults differ from those in the source checkout. Do not silently replace it with a library scheduler or change the attention backend between arms.

## Other adaptations

- **Lumina-DiMOO:** pinned source `81344e60545ae8d70f11039d3450717d974f831b`. Apply both files in `patches/lumina/` with `git apply --ignore-space-change --check`, then `git apply --ignore-space-change`. The pinned generator uses CRLF line endings; the option is needed for the recorded LF patch context. The head slice is `(126356, 134548)` on `model.model._tg_vocab_slice`; set it to `None` for the reference. Both arms retain the same ML-Cache settings. Global image token IDs are unchanged; only projection/output slicing changes. Keep `_tg_trace_hook=None` during timing.
- **BAGEL generation:** `generation_dispatch.guarded_infer` wraps the loaded official inferencer. It only rewrites supported stateless text-to-image calls with image guidance disabled, text CFG > 1, and TaylorSeer disabled; other calls use the original API. Do not credit inherited FFN pruning to this dispatch change.
- **BAGEL understanding:** native-KV append storage code is provided for inspection, but requires the recorded model-level guards and call-site integration. Do not call `append_kv` as a general cache replacement.

## Full benchmark reruns

`configs/protocols/` records scientific settings, counts, seeds, and controls. It is not a launcher. Use original benchmark licenses and upstream data downloads, retain task-specific caps and score definitions, and independently score both arms. See [results.md](results.md) for denominator and scoring distinctions.

A clean end-to-end reproduction of every full benchmark is still outstanding for this public packaging. The release provides inspectable code and independently checkable archived evidence without claiming that untested installation steps are already validated.
