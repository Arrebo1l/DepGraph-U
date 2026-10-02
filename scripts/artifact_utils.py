"""Small standard-library helpers shared by release tools and examples."""
from pathlib import Path
import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Unsafe artifact path: {name}')
    return path


def verify_hashes(root, expected):
    for name, digest in expected.items():
        path = relative_file(root, name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Artifact missing or changed: {name}')
    return len(expected)


def verify_hf_files(root, manifest):
    """Verify pinned HF LFS SHA-256 or non-LFS Git blob SHA-1, including sizes."""
    for item in manifest['files']:
        if item.get('type') != 'file':
            continue
        path = relative_file(root, item['path'])
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError(f'Weight/config missing or wrong size: {item["path"]}')
        if item.get('lfs'):
            actual, expected = sha256(path), item['lfs']['oid']
        else:
            digest = hashlib.sha1(f'blob {item["size"]}\0'.encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            actual, expected = digest.hexdigest(), item['oid']
        if actual != expected:
            raise ValueError(f'Pinned weight/config hash mismatch: {item["path"]}')


def ratio_of_means(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError('Nonempty, equally sized paired timings required')
    import math
    if not all(math.isfinite(v) and v > 0 for v in [*reference, *candidate]):
        raise ValueError('Latencies must be finite and positive')
    return sum(reference) / sum(candidate)
