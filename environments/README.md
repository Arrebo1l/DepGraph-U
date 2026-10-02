# Environment records

These files document historical runs; they are not one universal requirements file.

- `omnigen2-recorded.txt`: pinned CADD transfer environment (Python 3.12 / CUDA 12.4).
- `lumina-generation-recorded.txt`: Lumina image generation environment.
- `lumina-scoring-recorded.txt`: separate GenEval scoring environment.
- `showo2-understanding-recorded.txt`: recorded evaluation stack; local editable paths are commented in this public copy.
- `bagel-upstream-requirements.txt`: dependency list from the pinned BAGEL source, not a resolved environment lock.

CPU verification needs only Python 3.10+. For GPU work, create separate environments, retain model-specific package versions, and install compatible CUDA/FlashAttention builds. Run `python -m pip check` after installation. Dependency installation and GPU execution of the public packaging have not been certified across platforms.
