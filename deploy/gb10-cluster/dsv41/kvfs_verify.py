# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect filesystem KV blocks without importing vLLM or modifying the store."""

import argparse
import contextlib
import errno
import json
import os
import stat
import subprocess
import sys
import time
import zlib
from pathlib import Path

XATTR = "user.vllm_kv_crc32"
KINDS = (
    "ok",
    "size_mismatch",
    "no_xattr",
    "crc_mismatch",
    "unknown",
    "vanished",
    "skipped_tmp",
)
BAD = {"size_mismatch", "crc_mismatch"}
NO_XATTR = {errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP}
if hasattr(errno, "ENOATTR"):
    NO_XATTR.add(errno.ENOATTR)


def server_running():
    """Refuse deletion unless both independent server checks succeed."""
    for command in (
        ["systemctl", "--user", "is-active", "dsv41-serve.service"],
        ["pgrep", "-f", "vllm serve"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"server status unavailable: {exc}") from exc
        if command[0] == "systemctl":
            if result.stdout.strip() == "active":
                return True
            if result.returncode != 3 or result.stdout.strip() not in {
                "inactive",
                "failed",
            }:
                raise RuntimeError("cannot establish dsv41-serve.service status")
        elif result.returncode == 0:
            return True
        elif result.returncode != 1:
            raise RuntimeError("cannot establish vllm process status")
    return False


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def hex_part(value, length):
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


@contextlib.contextmanager
def parent_fd(root, relative):
    """Open each directory without following links, including the selected root."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


class Verifier:
    def __init__(self, root, *, fadvise=True, progress=False):
        self.root = root
        self.fadvise = fadvise
        self.progress = progress
        self.counts = dict.fromkeys(KINDS, 0)
        self.sizes = dict.fromkeys(KINDS, 0)
        self.examples = {kind: [] for kind in KINDS}
        self.files = self.read_bytes = self.errors = self.deleted = 0
        self.candidates = []
        self.started = time.monotonic()

    def record(self, kind, path, size=0):
        self.counts[kind] += 1
        self.sizes[kind] += size
        if len(self.examples[kind]) < 5:
            self.examples[kind].append(str(path))

    def error(self, path, exc):
        self.errors += 1
        print(f"error: {path}: {exc}", file=sys.stderr)

    def entries(self, directory):
        try:
            with os.scandir(directory) as entries:
                return sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            print(f"vanished directory: {directory}", file=sys.stderr)
        except OSError as exc:
            self.error(directory, exc)
        return []

    def paths(self, directory, depth=0):
        for entry in self.entries(directory):
            path = Path(entry.path)
            if entry.is_symlink():
                self.record("unknown", path)
            elif entry.is_dir(follow_symlinks=False):
                if depth < 2:
                    yield from self.paths(path, depth + 1)
                else:
                    self.record("unknown", path)
            elif entry.name.endswith(".tmp"):
                self.record("skipped_tmp", path)
            elif entry.name.endswith(".bin"):
                yield path

    def config(self, base):
        path = self.root / base / "config.json"
        try:
            with parent_fd(self.root, path.relative_to(self.root)) as directory:
                fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(fd) as stream:
                config = json.load(stream)
            groups = config["group_bytes"]
            if (
                config.get("format_version") != 2
                or config.get("blocks_per_file") != 1
                or not isinstance(groups, list)
                or not groups
                or any(type(size) is not int or size <= 0 for size in groups)
            ):
                raise ValueError("unsupported config")
            return groups
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            print(f"unknown config: {path}: {exc}", file=sys.stderr)
        except OSError as exc:
            if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
                self.error(path, exc)
        return None

    def inspect(self, path, expected):
        """Return classification, observed size and identity; retry CRC once."""
        for attempt in range(2):
            try:
                with parent_fd(self.root, path.relative_to(self.root)) as directory:
                    fd = os.open(
                        path.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory,
                    )
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    identity = signature(info)
                    size = info.st_size
                    if not stat.S_ISREG(info.st_mode) or expected is None:
                        return "unknown", size, identity
                    if size != expected:
                        return "size_mismatch", size, identity
                    try:
                        recorded = os.getxattr(stream.fileno(), XATTR)
                    except AttributeError:
                        return "no_xattr", size, identity
                    except OSError as exc:
                        if exc.errno not in NO_XATTR:
                            raise
                        if exc.errno != errno.ENODATA:
                            print(f"xattr unavailable: {path}: {exc}", file=sys.stderr)
                        return "no_xattr", size, identity
                    if len(recorded) != 4:
                        return "unknown", size, identity
                    try:
                        data = stream.read()
                        self.read_bytes += len(data)
                    finally:
                        if self.fadvise:
                            try:
                                os.posix_fadvise(
                                    stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                                )
                            except (AttributeError, OSError) as exc:
                                print(f"fadvise disabled: {exc}", file=sys.stderr)
                                self.fadvise = False
                    if signature(os.fstat(stream.fileno())) != identity:
                        if attempt == 0:
                            continue
                        return "unknown", size, identity
                    if zlib.crc32(data) & 0xFFFFFFFF == int.from_bytes(recorded, "big"):
                        return "ok", size, identity
            except FileNotFoundError:
                return "vanished", 0, None
            except OSError as exc:
                if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
                    self.error(path, exc)
                return "unknown", 0, None
        return "crc_mismatch", size, identity

    def scan(self, *, limit=None, sample=1, delete=False):
        seen = 0
        for entry in self.entries(self.root):
            if entry.is_symlink():
                self.record("unknown", Path(entry.path))
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            base, separator, rank = entry.name.rpartition("_r")
            groups = (
                self.config(base)
                if base and separator and rank.isascii() and rank.isdecimal()
                else None
            )
            for path in self.paths(Path(entry.path)):
                seen += 1
                if seen % sample:
                    continue
                relative = path.relative_to(self.root)
                prefix, separator, group = path.parent.name.rpartition("_g")
                expected = None
                if (
                    len(relative.parts) == 4
                    and hex_part(relative.parts[1], 3)
                    and hex_part(prefix, 2)
                    and separator
                    and group.isascii()
                    and group.isdecimal()
                    and groups
                    and int(group) < len(groups)
                ):
                    expected = groups[int(group)]
                kind, size, identity = self.inspect(path, expected)
                self.files += 1
                self.record(kind, path, size)
                if delete and kind in BAD:
                    self.candidates.append((path, identity, expected))
                if self.progress and self.files % 50_000 == 0:
                    print(f"kvfs_verify: inspected={self.files}", file=sys.stderr)
                if limit is not None and self.files >= limit:
                    return

    def delete(self, check_server):
        forbidden = {
            Path("/"),
            Path.home().resolve(),
            Path(__file__).resolve().parents[3],
        }
        if self.root in forbidden or (self.root / ".git").exists():
            raise RuntimeError(f"unsafe deletion root: {self.root}")
        if self.errors:
            raise RuntimeError("deletion refused after incomplete inspection")
        for path, _, _ in self.candidates:
            print(f"delete candidate: {path}")
        sys.stdout.flush()
        for path, identity, expected in self.candidates:
            if check_server():
                raise RuntimeError("server running; deletion refused")
            kind, _, current = self.inspect(path, expected)
            if current != identity or kind not in BAD:
                print(f"delete skipped (changed): {path}")
                continue
            with parent_fd(self.root, path.relative_to(self.root)) as directory:
                info = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or signature(info) != identity:
                    print(f"delete skipped (replaced): {path}")
                    continue
                os.remove(path.name, dir_fd=directory)
                self.deleted += 1
                print(f"deleted: {path}")

    def report(self):
        elapsed = time.monotonic() - self.started
        for kind in KINDS:
            print(f"{kind}: count={self.counts[kind]} bytes={self.sizes[kind]}")
            for path in self.examples[kind]:
                print(f"  {path}")
        print("no_xattr/unknown are unverified; live scans are not snapshots.")
        counts = " ".join(f"{kind}={self.counts[kind]}" for kind in KINDS)
        rate = self.read_bytes / max(elapsed, 1e-9) / (1024 * 1024)
        print(
            f"kvfs_verify: root={self.root} files={self.files} {counts} "
            f"deleted={self.deleted} secs={elapsed:.3f} mib_s={rate:.2f}"
        )


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None, *, check_server=server_running):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", default=os.environ.get("KVFS_DIR", "/mnt/kvdisk/kv/dsv41")
    )
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--limit", type=positive)
    parser.add_argument(
        "--sample",
        type=positive,
        default=1,
        help="inspect every Kth bin in sorted traversal (default: 1)",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--fadvise", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().absolute()
    if not root.is_dir() or root.is_symlink() or root != root.resolve():
        print(
            f"invalid root (requires a real directory without links): {root}",
            file=sys.stderr,
        )
        return 2
    verifier = Verifier(root, fadvise=args.fadvise, progress=args.progress)
    refused = False
    try:
        if check_server():
            raise RuntimeError("server running; report mode only")
    except RuntimeError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        refused = args.delete
    if args.sample != 1 or args.limit is not None:
        print(f"partial scan: sample={args.sample} limit={args.limit}")
    verifier.scan(
        limit=args.limit, sample=args.sample, delete=args.delete and not refused
    )
    if args.delete and not refused:
        try:
            if check_server():
                raise RuntimeError("server running; deletion refused")
            verifier.delete(check_server)
        except (RuntimeError, OSError) as exc:
            print(f"deletion refused: {exc}", file=sys.stderr)
            refused = True
    verifier.report()
    if refused:
        return 3
    if verifier.errors:
        return 2
    return int(any(verifier.counts[kind] for kind in BAD))


if __name__ == "__main__":
    sys.exit(main())
