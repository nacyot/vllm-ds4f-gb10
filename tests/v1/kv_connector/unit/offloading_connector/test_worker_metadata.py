# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    DirectionalTransferStats,
    NodeHeadroomConfig,
    NodeHeadroomSample,
    OffloadingWorkerMetadata,
    read_mem_available_bytes,
    TransferStats,
)

pytestmark = pytest.mark.cpu_test


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 2, 7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={43: 1, 8: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 1, 7: 1, 43: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1, 8: 1})
    meta3 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 8: 1})
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_jobs == {42: 3, 43: 2, 7: 2, 8: 2}


def test_aggregate_failed_jobs_across_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1}, failed_jobs={42: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})

    result = meta1.aggregate(meta2)

    assert result.completed_jobs == {42: 2, 7: 1}
    assert result.failed_jobs == {42: 1}


def test_aggregate_preserves_worker_rank_identity_and_deduplicates():
    meta0 = OffloadingWorkerMetadata(
        worker_rank=0,
        completed_jobs={42: 1},
        completed_job_ranks={42: {0}},
    )
    meta1 = OffloadingWorkerMetadata(
        worker_rank=1,
        completed_jobs={42: 1},
        failed_jobs={42: 1},
        completed_job_ranks={42: {1}},
        failed_job_ranks={42: {1}},
    )

    result = meta0.aggregate(meta1).aggregate(meta1)

    assert result.completed_job_ranks == {42: {0, 1}}
    assert result.failed_job_ranks == {42: {1}}


def test_aggregate_preserves_node_headroom_rank_identity():
    sample0 = NodeHeadroomSample(generation=3, sequence=4, available_bytes=100)
    sample1 = NodeHeadroomSample(generation=3, sequence=9, available_bytes=200)
    meta0 = OffloadingWorkerMetadata(worker_rank=0, node_headroom=sample0)
    meta1 = OffloadingWorkerMetadata(worker_rank=1, node_headroom=sample1)

    result = meta0.aggregate(meta1).aggregate(meta1)

    assert result.node_headroom_by_rank == {0: sample0, 1: sample1}


def test_aggregate_transfer_stats():
    meta1 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=10, time=0.5, sizes=[10])
        )
    )
    meta2 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=20, time=1.0, sizes=[20, 30])
        )
    )

    result = meta1.aggregate(meta2)

    assert result.transfer_stats.load.bytes == 30
    assert result.transfer_stats.load.time == 1.5
    assert result.transfer_stats.load.sizes == [10, 20, 30]


def test_node_headroom_parser_fails_closed(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 123 kB\n")
    assert read_mem_available_bytes(str(meminfo)) == 123 * 1024

    meminfo.write_text("MemAvailable: not-a-number kB\n")
    assert read_mem_available_bytes(str(meminfo)) is None
    meminfo.write_text("MemAvailable: 123 MB\n")
    assert read_mem_available_bytes(str(meminfo)) is None


def test_node_headroom_config_is_strict_restore_opt_in():
    config = {
        "node_headroom_gate": True,
        "node_headroom_reserve_bytes": 10,
        "node_headroom_j_bytes": 5,
        "node_headroom_heartbeat_max_age_ms": 100,
    }
    assert (
        NodeHeadroomConfig.from_extra_config(
            config, strict_restore_enabled=False
        )
        is None
    )
    parsed = NodeHeadroomConfig.from_extra_config(
        config, strict_restore_enabled=True
    )
    assert parsed is not None
    assert parsed.threshold_bytes == 15
