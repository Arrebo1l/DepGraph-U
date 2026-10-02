"""Maintainer tool: refresh hashes only after reviewing and staging a release.

Uses Git's tracked/staged file list rather than recursively collecting local
files, so checkpoints, secrets, and private outputs cannot enter implicitly.
"""
from pathlib import Path
import json
import subprocess
from artifact_utils import sha256

ROOT=Path(__file__).resolve().parents[1]
TARGET='provenance/release_manifest.json'


def main():
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    files={name:sha256(ROOT/name) for name in sorted(set(paths)) if name and name!=TARGET}
    (ROOT/TARGET).write_text(json.dumps({'schema_version':1,
        'scope':'Public working-tree content of the reviewed tracked files; this manifest excludes itself.',
        'files':files},indent=2)+'\n')
    print(f'Wrote {len(files)} hashes. Review and stage {TARGET} separately.')


if __name__=='__main__':main()
