"""Apply a pinned experiment variant to a new local clone, never the input repo."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
from artifact_utils import verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, destination, variant):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Destination already exists; choose a new directory')
    if destination.is_relative_to(source):
        raise ValueError('Destination must be outside the input repository')
    metadata = 'manifest.json' if variant=='generation' else 'understanding_manifest.json'
    manifest = json.loads((ROOT/'patches/showo2'/metadata).read_text())
    subprocess.run(['git','-C',str(source),'cat-file','-e',manifest['revision']+'^{commit}'], check=True)
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(source),str(destination)], check=True)
    subprocess.run(['git','-C',str(destination),'checkout','--detach',manifest['revision']], check=True)
    patch = ROOT/'patches/showo2'/f'{variant}.patch'
    subprocess.run(['git','-C',str(destination),'apply','--check',str(patch)], check=True)
    subprocess.run(['git','-C',str(destination),'apply',str(patch)], check=True)
    verify_hashes(destination, manifest['target_sha256'])
    for target, release_source in manifest.get('extra_files',{}).items():
        path=destination/target
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/release_source,path)
    print(f'Prepared {variant} source; target hashes match. Dependencies and weights are not installed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True, help='Existing local FlashU Git repository containing the pinned commit')
    p.add_argument('--output', type=Path, required=True, help='New directory outside the input checkout')
    p.add_argument('--variant', choices=['generation','understanding'], required=True)
    a=p.parse_args(); prepare(a.upstream,a.output,a.variant)


if __name__=='__main__': main()
