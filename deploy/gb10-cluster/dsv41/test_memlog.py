# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""memlog.py parsers and the head process chain, without /proc or torch."""

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def memlog():
    spec = importlib.util.spec_from_file_location("memlog", HERE / "memlog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "torch" not in sys.modules, "memlog must stay importable on a live node"
    return module


def test_meminfo_parses_original_and_issue30_keys(memlog):
    text = "MemTotal:  124546000 kB\nSUnreclaim: 2229344 kB\nDirty: 8 kB\n"
    assert memlog.parse_meminfo(text) == {"MemTotal": 121626, "SUnreclaim": 2177}


def test_vmstat_counters_default_to_zero(memlog):
    text = "pswpin 46453038\npswpout 203087277\ncompact_stall 436708\n"
    counters = memlog.parse_vmstat(text)
    assert counters["pswpout"] == 203087277
    assert counters["compact_stall"] == 436708
    assert counters["pgscan_direct"] == 0
    assert list(counters) == list(memlog.VMSTAT)


def test_foll_pin_is_acquired_minus_released(memlog):
    text = "nr_foll_pin_acquired 10\nnr_foll_pin_released 4\n"
    assert memlog.foll_pin(text) == 6


def test_buddyinfo_counts_normal_zone_blocks_from_order_nine(memlog):
    text = (
        "Node 0, zone      DMA    595 384 238 163 93 64 32 16 11 2 3 6 3 12\n"
        "Node 0, zone   Normal    437 2264 2519 3166 263 765 612 1717 594 "
        "5 2 1 0 5\n"
    )
    assert memlog.parse_buddyinfo(text) == 5 + 2 + 1 + 0 + 5
    assert memlog.parse_buddyinfo(text, min_order=13) == 5


def test_status_split_in_mib_with_missing_keys_zero(memlog):
    text = "Name:\tVLLM::Worker_TP\nRssAnon:\t 2904392 kB\nVmSwap:\t 1024 kB\n"
    assert memlog.parse_status(text) == {
        "VmRSS": 0,
        "RssAnon": 2836,
        "RssFile": 0,
        "RssShmem": 0,
        "VmSwap": 1,
    }
    assert memlog.parse_status(text, memlog.PARENT_STATUS) == {
        "RssAnon": 2836,
        "VmSwap": 1,
    }


def test_head_pids_follow_the_worker_parent_chain(memlog, tmp_path):
    def fake(pid, ppid, comm):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 -1\n")
        (d / "status").write_text(f"RssAnon:\t{pid * 1024} kB\nVmSwap:\t0 kB\n")

    fake(300, 200, "VLLM::Worker_TP")
    fake(200, 100, "VLLM::EngineCor")
    fake(100, 1, "vllm")
    assert memlog.head_pids(300, proc=str(tmp_path)) == [300, 200, 100]
    assert memlog.head_pids(None, proc=str(tmp_path)) == [None, None, None]
    assert memlog.status(200, memlog.PARENT_STATUS, proc=str(tmp_path)) == {
        "RssAnon": 200,
        "VmSwap": 0,
    }
    assert memlog.status(999, memlog.PARENT_STATUS, proc=str(tmp_path)) == {
        "RssAnon": 0,
        "VmSwap": 0,
    }


def test_columns_keep_the_original_order_then_append(memlog):
    cols = memlog.columns(2)
    assert cols[:3] == ["ts", "MemTotal", "MemFree"]
    assert cols[cols.index("SwapFree") + 1 :][:2] == ["foll_pin_pages", "worker_pid"]
    assert cols.index("engram1_res_mib") < cols.index("SUnreclaim")
    assert cols[-6:] == [
        "e_pid",
        "a_pid",
        "e_RssAnon",
        "e_VmSwap",
        "a_RssAnon",
        "a_VmSwap",
    ]
    assert "free_order9plus" in cols and "pswpout" in cols
