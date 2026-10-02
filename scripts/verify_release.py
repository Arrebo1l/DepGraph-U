"""Verify publication hashes, then recompute the bundled scientific evidence."""
from pathlib import Path
import json
import subprocess
import sys
from artifact_utils import verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: archived evidence checks use assertions')
    record = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    count = verify_hashes(ROOT, record['files'])
    print(f'PASS: {count} published artifact hashes', flush=True)
    origins = json.loads((ROOT / 'provenance/artifacts.json').read_text())
    verify_hashes(ROOT, {r['path']: r['published_sha256'] for r in origins['files']})
    subprocess.run([sys.executable, '-B', str(ROOT/'reproducibility/evidence/verify.py')],
                   cwd=ROOT, check=True)
    print('Release verification passed. No model inference was executed.')


if __name__ == '__main__':
    main()
