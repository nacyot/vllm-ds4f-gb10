# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
import mmap
import os
import random
import threading
import time
import zlib
from collections.abc import Collection

try:
    from vllm.fs_io_C import (  # pyright: ignore[reportMissingImports]
        batch_load_block as batch_load_block_C,
    )
    from vllm.fs_io_C import (
        batch_store_block as batch_store_block_C,
    )

    _HAS_FSIO_C = True
except ImportError:
    _HAS_FSIO_C = False

try:
    from vllm.fs_io_C import (  # pyright: ignore[reportMissingImports]
        batch_verify_crc32 as batch_verify_crc32_C,
    )

    _HAS_VERIFY_C = True
except ImportError:
    # An fs_io_C built before batch_verify_crc32 (e.g. the wheel's copy).
    _HAS_VERIFY_C = False

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()

# Pause before the single retry of a block whose read failed.
_RETRY_DELAY_S = 0.01


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def probe_o_direct(directory: str) -> bool:
    """Return whether ``O_DIRECT`` I/O works in *directory*.

    ``O_DIRECT`` is unsupported on some filesystems (e.g. the overlayfs backing
    a container ``/tmp``, older tmpfs, or some NFS mounts), where opening or
    writing a file with it fails with ``EINVAL``. Probe once with an aligned
    single-page write so callers can fall back to buffered I/O instead of
    failing on every block.
    """
    if not O_DIRECT:
        return False
    path = os.path.join(directory, f".o_direct_probe{_get_tmp_suffix()}")
    page = mmap.mmap(-1, mmap.PAGESIZE)
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | O_DIRECT, 0o644)
        try:
            os.write(fd, page)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False
    finally:
        page.close()
        with contextlib.suppress(OSError):
            os.remove(path)


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _validate_offsets(
    view: memoryview, offsets: list[int], block_sizes: list[int]
) -> None:
    """Raise if any block would read/write past the bounds of `view`.

    Without this, an out-of-range offset silently clips to a shorter (or
    empty) slice instead of failing, since memoryview slicing follows
    Python's slice-clamping semantics rather than raising.
    """
    total_len = len(view.cast("B"))
    for offset, block_size in zip(offsets, block_sizes):
        if offset < 0 or offset + block_size > total_len:
            raise ValueError(
                f"block offset {offset} (block_size {block_size}) is out of "
                f"bounds for a buffer of size {total_len}"
            )


_XATTR_CRC = "user.vllm_kv_crc32"


def _crc_of(view: memoryview, offset: int, size: int) -> int:
    return zlib.crc32(view.cast("B")[offset : offset + size]) & 0xFFFFFFFF


def write_checksums(
    paths: list[str], view: memoryview, offsets: list[int], sizes: list[int]
) -> None:
    """Record a CRC32 of each stored block as an extended attribute so a
    byte-for-byte corruption with the right length is caught on load.
    Silently skipped where the filesystem has no xattr support."""
    for path, offset, size in zip(paths, offsets, sizes):
        try:
            os.setxattr(
                path, _XATTR_CRC, _crc_of(view, offset, size).to_bytes(4, "big")
            )
        except OSError:
            return


def _crc_mismatches(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    sizes: list[int],
    indices: list[int],
) -> list[int]:
    """Return the members of ``indices`` whose recorded CRC32 differs from
    the bytes in ``view``. A file without a recorded checksum is not checked.

    With the C extension the whole batch runs under a single GIL release
    (one getxattr and one CRC32 per block); the Python fallback reacquires
    the GIL twice per block, which starves the reader when the engine's
    scheduler thread is busy."""
    if _HAS_VERIFY_C:
        view_B = view.cast("B")
        mismatched = batch_verify_crc32_C(
            [paths[i] for i in indices],
            [view_B[offsets[i] : offsets[i] + sizes[i]] for i in indices],
        )
        return [indices[j] for j in mismatched]
    mismatched = []
    for i in indices:
        try:
            recorded = os.getxattr(paths[i], _XATTR_CRC)
        except OSError:
            continue
        if int.from_bytes(recorded, "big") != _crc_of(view, offsets[i], sizes[i]):
            mismatched.append(i)
    return mismatched


def verify_checksums(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    sizes: list[int],
    skip: Collection[int] = (),
) -> list[int]:
    """Compare each loaded block with its recorded CRC32 and return the indices
    of the blocks that do not match. A mismatching file is removed and the
    check continues with the next block, so one corrupt file costs only its
    own key. Indices in ``skip`` (blocks that were not loaded) are not checked.

    The data and the xattr are read by path at different moments, so a
    concurrent rewrite of the same key (temp file + rename) can pair the old
    bytes with the new checksum. A mismatch is therefore confirmed by reading
    the file again before it is declared corrupt; the re-read bytes replace the
    stale ones."""
    indices = [i for i in range(len(paths)) if i not in skip]
    failed: list[int] = []
    for i in _crc_mismatches(paths, view, offsets, sizes, indices):
        path, offset, size = paths[i], offsets[i], sizes[i]
        try:
            _load_block(path, view, offset, size, use_o_direct=False)
            recorded = os.getxattr(path, _XATTR_CRC)
        except OSError:
            recorded = None
        if recorded is not None and int.from_bytes(recorded, "big") == _crc_of(
            view, offset, size
        ):
            logger.debug("Re-read %s after a concurrent rewrite", path)
            continue
        try:
            os.remove(path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove corrupt file %s: %s", path, cleanup_exc)
        logger.warning("Checksum mismatch for %s; file removed", path)
        failed.append(i)
    return failed


def _block_sizes(block_size: int | list[int], count: int) -> list[int]:
    if isinstance(block_size, int):
        return [block_size] * count
    assert len(block_size) == count
    return list(block_size)


def _store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # Check if block already exists to avoid redundant writes
    if os.path.exists(dest_path):
        return

    tmp_path = dest_path + _get_tmp_suffix()
    # Ensure parent directories exist
    _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    o_direct = O_DIRECT if use_o_direct else 0
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | o_direct,
            0o644,
        )
        try:
            written = os.write(fd, view_slice)
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, wrote {written}"
                )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def _load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """Read one KV block from disk; remove the file only on a provable short
    read (a too-short file is genuine corruption) and leave it untouched on any
    other error."""
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    o_direct = O_DIRECT if use_o_direct else 0

    try:
        fd = os.open(source_path, os.O_RDONLY | o_direct)
        bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            # A failure to remove must not mask the short-read error below.
            try:
                os.remove(source_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to remove short-read file %s: %s",
                    source_path,
                    cleanup_exc,
                )
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    finally:
        if fd is not None:
            os.close(fd)


def _exists_with_size(path: str, size: int) -> bool:
    try:
        return os.stat(path).st_size == size
    except OSError:
        return False


def batch_store_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int | list[int],
    use_o_direct: bool = True,
    checksums: bool = False,
    skip_existing: bool = False,
) -> None:
    """
    Store a batch of KV blocks from a shared buffer to disk in one call.

    Each block buffer[offsets[i] : offsets[i]+size_i] is written atomically
    to dest_paths[i] via a temp-file rename, where size_i is ``block_size``
    or ``block_size[i]`` when a per-block list is given. Raises on first error.
    With ``skip_existing`` a block whose file is already present with the
    expected size is not rewritten: the key is a content hash, and rewriting
    a file that a concurrent load is reading pairs its bytes with the wrong
    checksum.
    """
    sizes = _block_sizes(block_size, len(offsets))
    _validate_offsets(view, offsets, sizes)

    if skip_existing:
        keep = [
            i
            for i, (path, size) in enumerate(zip(paths, sizes))
            if not _exists_with_size(path, size)
        ]
        if len(keep) < len(paths):
            paths = [paths[i] for i in keep]
            offsets = [offsets[i] for i in keep]
            sizes = [sizes[i] for i in keep]
        if not paths:
            return

    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + n] for x, n in zip(offsets, sizes)]
        tmp_paths = [p + _get_tmp_suffix() for p in paths]
        batch_store_block_C(tmp_paths, paths, view_slices, use_o_direct)
    else:
        for path, offset, n in zip(paths, offsets, sizes):
            _store_block(path, view, offset, n, use_o_direct)
    if checksums:
        write_checksums(paths, view, offsets, sizes)


def _retry_block(
    path: str,
    view: memoryview,
    offset: int,
    size: int,
    use_o_direct: bool,
    exc: OSError,
) -> bool:
    """Read one block again after a short pause; True if it loaded."""
    time.sleep(_RETRY_DELAY_S)
    try:
        _load_block(path, view, offset, size, use_o_direct)
    except OSError as retry_exc:
        logger.warning(
            "Block %s failed to load after a retry: %s (first error: %s)",
            path,
            retry_exc,
            exc,
        )
        return False
    logger.debug("Block %s loaded on retry after: %s", path, exc)
    return True


def _load_blocks(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    sizes: list[int],
    use_o_direct: bool,
) -> list[int]:
    """Load every block, retrying each failed one once; return the indices
    that still failed. The C loader stops at its first failure and reports the
    index, so it is resumed after each failed block."""
    failed: list[int] = []
    n = len(paths)
    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + n_] for x, n_ in zip(offsets, sizes)]
        start = 0
        while start < n:
            try:
                batch_load_block_C(paths[start:], view_slices[start:], use_o_direct)
                break
            except OSError as exc:
                i = start + getattr(exc, "num_succeeded", 0)
                if not _retry_block(
                    paths[i], view, offsets[i], sizes[i], use_o_direct, exc
                ):
                    failed.append(i)
                start = i + 1
        return failed
    for i, (path, offset, size) in enumerate(zip(paths, offsets, sizes)):
        try:
            _load_block(path, view, offset, size, use_o_direct)
        except OSError as exc:
            if not _retry_block(path, view, offset, size, use_o_direct, exc):
                failed.append(i)
    return failed


def batch_load_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int | list[int],
    use_o_direct: bool = True,
    checksums: bool = False,
) -> list[int]:
    """
    Load a batch of KV blocks from disk into a shared buffer in one call.

    Block i is read from source_paths[i] into view[offsets[i] : offsets[i]+size_i]
    (``block_size`` or ``block_size[i]``). A block that fails to read is
    retried once; one that still fails (or whose checksum does not match) is
    skipped and the rest of the batch is still loaded. Returns the sorted
    indices of the blocks that were not loaded (empty when all loaded). See
    _load_block for the delete-on-short-read policy.
    """
    sizes = _block_sizes(block_size, len(offsets))
    _validate_offsets(view, offsets, sizes)

    failed = _load_blocks(paths, view, offsets, sizes, use_o_direct)
    if checksums:
        failed = sorted(
            failed + verify_checksums(paths, view, offsets, sizes, skip=set(failed))
        )
    return failed
