# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV inspection and deletion boundaries, using only disposable test stores."""

import errno
import importlib.util
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location(
        "kvfs_verify", Path(__file__).with_name("kvfs_verify.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
    return module


@pytest.fixture
def store(tmp_path, verify):
    root = tmp_path.resolve() / "store"
    config = root / "model" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "format_version": 2,
                "blocks_per_file": 1,
                "group_bytes": [4, 8, 12],
            }
        )
    )
    directory = root / "model_r0" / "abc" / "de_g1"
    directory.mkdir(parents=True)

    def make(name, data=b"abcdefgh", checksum=True):
        path = directory / name
        path.write_bytes(data)
        if checksum:
            if not hasattr(os, "setxattr"):
                pytest.skip("os.setxattr unavailable")
            try:
                os.setxattr(path, verify.XATTR, zlib.crc32(data).to_bytes(4, "big"))
            except OSError as exc:
                if exc.errno in verify.NO_XATTR:
                    pytest.skip(f"test filesystem does not support user xattrs: {exc}")
                raise
        return path

    return root, make


def damaged_store(make):
    good = make("abcde0.bin")
    crc = make("abcde1.bin")
    with crc.open("r+b") as stream:
        stream.write(b"X")
    short = make("abcde2.bin", b"short")
    bare = make("abcde3.bin", checksum=False)
    temporary = make("x_random.tmp", checksum=False)
    return good, crc, short, bare, temporary


@pytest.mark.parametrize(
    "delete,running,code,removed",
    [
        (False, False, 1, 0),
        (False, True, 1, 0),
        (True, True, 3, 0),
        (True, False, 1, 2),
    ],
)
def test_reports_damage_and_only_deletes_confirmed_damage(
    verify,
    store,
    capsys,
    delete,
    running,
    code,
    removed,
):
    root, make = store
    good, crc, short, bare, temporary = damaged_store(make)
    argv = [str(root)] + (["--delete"] if delete else [])
    assert verify.main(argv, check_server=lambda: running) == code
    output = capsys.readouterr().out
    assert "files=4 ok=1 size_mismatch=1 no_xattr=1 crc_mismatch=1" in output
    assert f"skipped_tmp=1 deleted={removed}" in output
    assert good.exists() and bare.exists() and temporary.exists()
    assert crc.exists() == short.exists() == (removed == 0)
    if removed:
        assert output.index(f"delete candidate: {short}") < output.index("deleted:")


def test_crc_retry_accepts_fresh_checksum(verify, store, monkeypatch):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    checksum = zlib.crc32(path.read_bytes()).to_bytes(4, "big")
    calls = []

    def race(fd, attribute):
        calls.append(fd)
        return b"\x00" * 4 if len(calls) == 1 else checksum

    monkeypatch.setattr(verify.os, "getxattr", race, raising=False)
    inspector = verify.Verifier(root)
    assert inspector.inspect(path, 8)[0] == "ok"
    assert len(calls) == 2


def test_persistent_crc_failure_reads_twice_and_releases_pages(
    verify,
    store,
    monkeypatch,
):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    monkeypatch.setattr(
        verify.os,
        "getxattr",
        lambda fd, name: b"\x00" * 4,
        raising=False,
    )
    advised = []
    monkeypatch.setattr(verify.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        verify.os,
        "posix_fadvise",
        lambda *args: advised.append(args),
        raising=False,
    )
    inspector = verify.Verifier(root)
    assert inspector.inspect(path, 8)[0] == "crc_mismatch"
    assert inspector.read_bytes == 16
    assert len(advised) == 2


@pytest.mark.parametrize("directory", ["de", "de_g3", "de_g-1"])
def test_unknown_group_is_never_a_delete_candidate(verify, store, directory):
    root, make = store
    path = make("abcde0.bin", b"bad", checksum=False)
    path.parent.rename(path.parent.with_name(directory))
    inspector = verify.Verifier(root)
    inspector.scan(delete=True)
    assert inspector.counts["unknown"] == 1
    assert inspector.candidates == []


def test_vanished_file_is_not_damage(verify, store, monkeypatch):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    real_open = verify.os.open

    def gone(name, *args, **kwargs):
        if name == path.name:
            raise FileNotFoundError(name)
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(verify.os, "open", gone)
    inspector = verify.Verifier(root)
    inspector.scan()
    assert inspector.counts["vanished"] == 1
    assert not any(inspector.counts[kind] for kind in verify.BAD)


def test_gc_directory_removal_does_not_abort(verify, store, monkeypatch):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    real_scandir = verify.os.scandir

    def gone(directory):
        if Path(directory) == path.parent:
            raise FileNotFoundError(directory)
        return real_scandir(directory)

    monkeypatch.setattr(verify.os, "scandir", gone)
    inspector = verify.Verifier(root)
    inspector.scan()
    assert inspector.files == inspector.errors == 0


@pytest.mark.parametrize("config", [None, {}, [], {"group_bytes": [8]}])
def test_unknown_config_is_report_only(verify, store, config):
    root, make = store
    path = make("abcde0.bin", b"short", checksum=False)
    config_path = root / "model" / "config.json"
    if config is None:
        # Rename this fixture's config to simulate an absent configuration.
        config_path.rename(config_path.with_suffix(".saved"))
    else:
        config_path.write_text(json.dumps(config))
    inspector = verify.Verifier(root)
    inspector.scan(delete=True)
    assert inspector.counts["unknown"] == 1
    inspector.delete(lambda: False)
    assert path.exists()


def test_links_outside_root_are_reported_and_preserved(verify, store, tmp_path):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (path.parent / "linked.bin").symlink_to(outside)
    (root / "linked_r0").symlink_to(tmp_path, target_is_directory=True)
    inspector = verify.Verifier(root)
    inspector.scan(delete=True)
    inspector.delete(lambda: False)
    assert inspector.counts["unknown"] == 2
    assert outside.read_bytes() == b"outside"


def test_changed_candidate_is_preserved(verify, store):
    root, make = store
    path = make("abcde0.bin", b"bad", checksum=False)
    inspector = verify.Verifier(root)
    inspector.scan(delete=True)
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(b"another bad file")
    replacement.replace(path)
    inspector.delete(lambda: False)
    assert path.read_bytes() == b"another bad file"
    assert inspector.deleted == 0


def test_server_start_before_delete_preserves_candidates(verify, store):
    root, make = store
    path = make("abcde0.bin", b"bad", checksum=False)
    checks = iter([False, True])
    assert verify.main([str(root), "--delete"], check_server=lambda: next(checks)) == 3
    assert path.exists()


@pytest.mark.parametrize(
    "service,pgrep,expected",
    [
        ((0, "active"), None, True),
        ((3, "inactive"), (0, "123"), True),
        ((3, "inactive"), (1, ""), False),
        ((1, ""), None, RuntimeError),
        ((3, "inactive"), (2, ""), RuntimeError),
    ],
)
def test_server_checks_fail_closed(verify, monkeypatch, service, pgrep, expected):
    def run(command, **kwargs):
        code, output = service if command[0] == "systemctl" else pgrep
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(verify.subprocess, "run", run)
    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            verify.server_running()
    else:
        assert verify.server_running() is expected


def test_unavailable_server_check_refuses_delete(verify, store):
    root, make = store
    path = make("abcde0.bin", b"bad", checksum=False)

    def unavailable():
        raise RuntimeError("status unavailable")

    assert verify.main([str(root), "--delete"], check_server=unavailable) == 3
    assert path.exists()


def test_sample_uses_sorted_paths_and_limit_counts_inspections(verify, store):
    root, make = store
    for name in ("abcde3.bin", "abcde1.bin", "abcde2.bin", "abcde0.bin"):
        make(name, checksum=False)
    inspector = verify.Verifier(root)
    inspector.scan(sample=2, limit=1)
    assert inspector.files == 1
    assert inspector.examples["no_xattr"][0].endswith("abcde1.bin")


def test_cli_root_precedence_and_missing_root(verify, store, monkeypatch, capsys):
    root, _ = store
    monkeypatch.setenv("KVFS_DIR", str(root))
    assert verify.main([], check_server=lambda: False) == 0
    assert f"root={root} files=0" in capsys.readouterr().out
    assert verify.main([str(root / "missing")], check_server=lambda: False) == 2


def test_io_error_is_incomplete_scan(verify, store, monkeypatch):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    real_open = verify.os.open

    def denied(name, *args, **kwargs):
        if name == path.name:
            raise PermissionError(name)
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(verify.os, "open", denied)
    assert verify.main([str(root)], check_server=lambda: False) == 2
    assert path.exists()


def test_inspection_preserves_gc_access_time(verify, store):
    if not hasattr(os, "O_NOATIME"):
        pytest.skip("O_NOATIME is Linux-specific")
    root, make = store
    path = make("abcde0.bin")
    os.utime(path, ns=(1_000_000_000, path.stat().st_mtime_ns))
    inspector = verify.Verifier(root)
    assert inspector.inspect(path, 8)[0] == "ok"
    assert path.stat().st_atime_ns == 1_000_000_000


def test_noatime_permission_failure_warns_and_falls_back(
    verify,
    store,
    monkeypatch,
    capsys,
):
    root, make = store
    path = make("abcde0.bin", checksum=False)
    real_open = verify.os.open
    noatime = getattr(os, "O_NOATIME", 1 << 29)
    monkeypatch.setattr(verify.os, "O_NOATIME", noatime, raising=False)

    def denied(name, flags, **kwargs):
        if flags & noatime:
            raise PermissionError(errno.EPERM, "not the owner")
        return real_open(name, flags, **kwargs)

    monkeypatch.setattr(verify.os, "open", denied)
    inspector = verify.Verifier(root)
    assert inspector.inspect(path, 8)[0] == "no_xattr"
    assert "reads may update atime" in capsys.readouterr().err
