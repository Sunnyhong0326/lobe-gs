#!/usr/bin/env python3
"""Extract BigCity ZIP/TAR archives and delete successfully extracted TAR inputs."""

import argparse
import io
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_ROOT = Path("/mnt/nfs/sunnyhong/data/LoBE-GS-dataset/original-poses/mc_bigcity")
SPLIT_TAR = re.compile(r"^(.*\.tar)(\d+)$")
TAR_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".tgz")


class JoinedParts(io.RawIOBase):
    """Read split archives as one stream without creating a combined tar file."""

    def __init__(self, parts):
        super().__init__()
        self.parts = iter(parts)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.parts, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def archives_in(directory):
    groups = {}
    archives = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = SPLIT_TAR.match(path.name)
        if match:
            groups.setdefault(match[1], []).append((int(match[2]), path))
        elif path.name.endswith(".zip"):
            archives.append((path.name[:-4], [path], "zip"))
        else:
            for suffix in TAR_SUFFIXES:
                if path.name.endswith(suffix):
                    archives.append((path.name[:-len(suffix)], [path], "tar"))
                    break
    for name, pieces in groups.items():
        pieces.sort()
        indices = [index for index, _ in pieces]
        if indices != list(range(len(indices))):
            raise ValueError(f"Missing or duplicate archive parts: {directory / name}: {indices}")
        if (directory / name).exists():
            raise ValueError(f"Both complete and split archive exist: {directory / name}")
        archives.append((name[:-4], [path for _, path in pieces], "tar"))
    return sorted(archives)


def destination_for(name, target):
    """Strip the archive's enclosing block folder and reject escaping paths."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    parts = relative.parts
    if parts and parts[0] == target.name:
        parts = parts[1:]
    if parts and parts[-1] == "transforms.json":
        return None
    destination = target.joinpath(*parts)
    if not destination.resolve().is_relative_to(target.resolve()):
        raise ValueError(f"Archive destination escapes target: {name}")
    return destination


def copy_member(source, destination, skip_existing):
    if skip_existing and destination.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so an interrupted extraction leaves no partial image.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def extract(parts, kind, target, skip_existing):
    # Block folders may be missing or left empty by a previous cleanup.
    # Neither case means that their archive has already been extracted.
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    if kind == "zip":
        with zipfile.ZipFile(parts[0]) as archive:
            for member in archive.infolist():
                destination = destination_for(member.filename, target)
                if destination is None:
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsupported ZIP symlink: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    with archive.open(member) as source:
                        written += copy_member(source, destination, skip_existing)
    else:
        with io.BufferedReader(JoinedParts(parts)) as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as archive:
                for member in archive:
                    destination = destination_for(member.name, target)
                    if destination is None:
                        continue
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        with archive.extractfile(member) as source:
                            written += copy_member(source, destination, skip_existing)
                    else:
                        raise ValueError(f"Unsupported TAR member: {member.name}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT,
                        help=f"dataset root (default: {DEFAULT_ROOT})")
    parser.add_argument("--dry-run", action="store_true", help="print actions without writing/deleting")
    parser.add_argument("--skip-existing", action="store_true", help="keep existing extracted files")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Dataset directory does not exist: {args.root}")
    jobs = []
    for split in ("train", "test"):
        directory = args.root / split
        if directory.is_dir():
            jobs.extend((directory, name, parts, kind)
                        for name, parts, kind in archives_in(directory))
    for directory, name, parts, kind in jobs:
        target = directory / name
        if target.exists() and not target.is_dir():
            raise ValueError(f"Expected a block directory, found a file: {target}")
        action = "FILL" if target.exists() else "CREATE"
        print(f"[{action} DIRECTORY] {target}", flush=True)
        print(f"[EXTRACT] {', '.join(p.name for p in parts)} -> {target}", flush=True)
        if not args.dry_run:
            written = extract(parts, kind, target, args.skip_existing)
            print(f"[EXTRACTED] {written} files written to {target}", flush=True)
        if kind == "tar":
            for path in parts:
                print(f"[REMOVE ARCHIVE] {path}", flush=True)
                if not args.dry_run:
                    path.unlink()
    # Remove existing transforms as well as omitting archived copies.
    for split in ("train", "test"):
        directory = args.root / split
        for path in sorted(directory.rglob("transforms.json")):
            print(f"[REMOVE] {path}", flush=True)
            if not args.dry_run:
                path.unlink()
    print(f"[DONE] {'Planned' if args.dry_run else 'Extracted'} {len(jobs)} archives; "
          "TAR inputs removed after successful extraction, ZIP inputs retained.")


if __name__ == "__main__":
    main()
