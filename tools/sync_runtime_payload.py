"""Update files listed in a JSON manifest, preserving replaced files.

Manifest: {"files": ["api_server.py", "run_inference.py"]}.
Use --dry-run first and stop the target service before applying an update.
Unlisted files are never removed. No game project or Git history is required.
"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def overlaps(a, b):
    return a.is_relative_to(b) or b.is_relative_to(a)


def contained(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes root: {path}')
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink() or (item.exists() and item.resolve() != item.absolute()):
            raise ValueError(f'Linked payload paths are unsupported: {item}')
    return path


def synchronize(source, target, manifest, backup, report, dry_run=False):
    source, target, backup, report = [p.resolve() for p in (source, target, backup, report)]
    if not source.is_dir() or not target.is_dir():
        raise ValueError('Source and target must be existing directories')
    if any(overlaps(a, b) for a, b in ((source, target), (source, backup), (target, backup))):
        raise ValueError('Source, target and backup trees must be disjoint')
    if backup.exists() or report.exists():
        raise FileExistsError('Use new backup and report paths')
    if any(overlaps(report, root) for root in (source, target, backup)):
        raise ValueError('Report must be outside source, target and backup trees')
    files = json.loads(manifest.read_text('utf-8-sig'))['files']
    if not isinstance(files, list) or not files:
        raise ValueError('Manifest files must be a nonempty list of relative paths')
    changes, seen = [], set()
    for relative in files:
        if not isinstance(relative, str):
            raise ValueError('Manifest entries must be strings')
        relative = relative.replace('\\', '/')
        parts = relative.split('/')
        if (PureWindowsPath(relative).drive or relative.startswith('/') or ':' in relative or
                any(p in ('', '.', '..') or p.endswith((' ', '.')) for p in parts)):
            raise ValueError(f'Unsafe manifest path: {relative}')
        if relative.casefold() in seen:
            raise ValueError(f'Duplicate manifest path: {relative}')
        seen.add(relative.casefold())
        src, dst = contained(source, relative), contained(target, relative)
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and not dst.is_file():
            raise ValueError(f'Destination is not a file: {dst}')
        if any(p.exists() and not p.is_dir() for p in dst.parents):
            raise ValueError(f'Destination parent is not a directory: {dst}')
        expected = digest(src)
        old = digest(dst) if dst.is_file() else None
        if old != expected:
            changes.append(dict(path=relative, before_sha256=old, sha256=expected))
    result = dict(source=str(source), target=str(target), backup=str(backup),
                  dry_run=dry_run, verified_files=len(files), changes=changes, applied=[])
    if dry_run:
        return result
    backup.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report.write_text(json.dumps(result, indent=2), 'utf-8')
    result['passed'] = False
    save()
    try:
        for row in changes:
            src, dst = contained(source, row['path']), contained(target, row['path'])
            if digest(src) != row['sha256'] or (digest(dst) if dst.is_file() else None) != row['before_sha256']:
                raise RuntimeError(f'File changed during update: {row["path"]}')
            if dst.exists():
                saved = backup / row['path']
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, saved)
            result['applied'].append(row['path'])
            save()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if digest(dst) != row['sha256']:
                raise RuntimeError(f'Copy verification failed: {row["path"]}')
        result['passed'] = True
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'manifest', 'backup', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    print(json.dumps(synchronize(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()