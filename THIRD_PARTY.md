# Third-party attribution and license boundaries

This project builds on existing models, approximation policies, and evaluators. Their contributions are separate from the execution rewrites studied here.

| Component | Upstream | Recorded code revision / role |
|---|---|---|
| FlashU / Flash-Unified | https://github.com/Rirayh/FlashU | `189f877953387f8b7f1e3f4d63eddbc5b597da52`; generation policy implementation and patch base |
| Show-o / Show-o2 | https://github.com/showlab/Show-o | Model implementation underlying the FlashU integration |
| Lumina-DiMOO | https://github.com/Alpha-VLLM/Lumina-DiMOO | `81344e60545ae8d70f11039d3450717d974f831b`; ML-Cache fixed reference |
| OmniGen2 | https://github.com/VectorSpaceLab/OmniGen2 | `18e6f9d5271b517fcb32e999f10df943ae9b8f20`; TaylorSeer controls |
| BAGEL | https://github.com/ByteDance-Seed/Bagel | `a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f`; native generation/KV and FFN adaptation |
| Transformers / Qwen | https://github.com/huggingface/transformers | Model and cache interfaces; preserved source copyright notices |
| GenEval | https://github.com/djghosh13/geneval | Prompt metadata and independent generation evaluation |
| DPG-Bench / ELLA | https://github.com/TencentQQGYLab/ELLA | Dependency-based scoring; disclosed first-row loader correction |
| MMStar | https://github.com/MMStar-Benchmark/MMStar | Overall accuracy and archived-answer rescoring |

Preserve upstream notices when applying or redistributing patches. The Show-o/Qwen-derived model files carry Apache-2.0 notices. The pinned FlashU tree inspected for this release did not contain a repository-root LICENSE file; this release does not invent or grant rights over that upstream code. Upstream model weights, datasets, and generated-content rights are not relicensed here.

The compact evidence includes original checks and metadata alongside third-party benchmark labels and model-generated outputs. A repository-level code license, if added, does not replace the applicable upstream terms or license the paper/datasets indiscriminately.

Original-code licensing is being selected by the repository owner. Until an explicit license is added, public visibility alone is not a grant of an open-source license. Existing third-party licenses continue to apply to their material.
