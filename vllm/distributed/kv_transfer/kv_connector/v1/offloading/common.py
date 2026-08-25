# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str
GIB = 1024**3


@dataclass(frozen=True, slots=True)
class NodeHeadroomConfig:
    """Opt-in unified-memory admission settings for bounded restore."""

    reserve_bytes: int
    j_bytes: int
    heartbeat_max_age_ms: int

    @property
    def threshold_bytes(self) -> int:
        return self.reserve_bytes + self.j_bytes

    @classmethod
    def from_extra_config(
        cls,
        extra_config: Mapping[str, object],
        *,
        strict_restore_enabled: bool,
    ) -> "NodeHeadroomConfig | None":
        """Parse the strict-restore-only gate without affecting legacy paths.

        The calibration and freshness window are deliberately required when
        the gate is enabled.  There is no implicit timeout or host-specific
        default that could silently weaken the admission contract.
        """
        if not isinstance(extra_config, Mapping):
            return None
        if not strict_restore_enabled or not extra_config.get(
            "node_headroom_gate", False
        ):
            return None

        reserve_bytes = _required_nonnegative_int(
            extra_config.get("node_headroom_reserve_bytes", 16 * GIB),
            "node_headroom_reserve_bytes",
        )
        j_bytes = _required_nonnegative_int(
            extra_config.get("node_headroom_j_bytes"), "node_headroom_j_bytes"
        )
        heartbeat_max_age_ms = _required_positive_int(
            extra_config.get("node_headroom_heartbeat_max_age_ms"),
            "node_headroom_heartbeat_max_age_ms",
        )
        return cls(
            reserve_bytes=reserve_bytes,
            j_bytes=j_bytes,
            heartbeat_max_age_ms=heartbeat_max_age_ms,
        )


def _required_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_mem_available_bytes(path: str = "/proc/meminfo") -> int | None:
    """Read ``MemAvailable`` from procfs, failing closed on malformed input."""
    try:
        with open(path, encoding="utf-8") as file:
            for line in file:
                fields = line.split()
                if len(fields) < 2 or fields[0] != "MemAvailable:":
                    continue
                value = int(fields[1])
                if value < 0:
                    return None
                if len(fields) == 2:
                    return value
                if len(fields) == 3 and fields[2].lower() == "kb":
                    return value * 1024
                return None
    except (OSError, UnicodeError, ValueError):
        return None
    return None


@dataclass(frozen=True, slots=True)
class NodeHeadroomSample:
    """Rank-local heartbeat carried through the worker metadata aggregate."""

    generation: int
    sequence: int
    available_bytes: int | None


@dataclass(slots=True)
class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)

    def aggregate(
        self, other: "DirectionalTransferStats"
    ) -> "DirectionalTransferStats":
        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )

    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)

    def is_empty(self) -> bool:
        return self.bytes == 0 and self.time == 0.0 and not self.sizes


@dataclass(slots=True)
class TransferStats:
    load: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)
    store: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)

    def aggregate(self, other: "TransferStats") -> "TransferStats":
        return TransferStats(
            load=self.load.aggregate(other.load),
            store=self.store.aggregate(other.store),
        )

    def is_empty(self) -> bool:
        return self.load.is_empty() and self.store.is_empty()


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
    # Propagates the scheduler generation so workers cannot reuse a heartbeat
    # sampled before reset_cache().
    node_headroom_generation: int = 0


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). Failed jobs are also recorded in ``failed_jobs`` so a
    single failed rank makes the aggregated job fail closed. aggregate() sums
    counts across workers within a step.
    The scheduler accumulates across steps and processes a transfer completion
    only when every expected worker rank has acknowledged it.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)
    # Subset of completed_jobs whose transfer did not reach the destination.
    failed_jobs: dict[int, int] = field(default_factory=dict)
    # Identity-preserving acknowledgements.  Counts remain for diagnostics and
    # compatibility, but the scheduler must use these sets for TP completion.
    completed_job_ranks: dict[int, set[int]] = field(default_factory=dict)
    failed_job_ranks: dict[int, set[int]] = field(default_factory=dict)
    # Rank that produced this (pre-aggregation) metadata.  Aggregated
    # metadata leaves this unset and carries the per-job rank sets above.
    # Keep this field after the original positional fields for compatibility.
    worker_rank: int | None = None
    # Pre-aggregation heartbeat from this worker.
    node_headroom: NodeHeadroomSample | None = None
    # Aggregated rank-preserving heartbeats consumed by the scheduler.
    node_headroom_by_rank: dict[int, NodeHeadroomSample] = field(default_factory=dict)

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1
        if self.worker_rank is not None:
            self.completed_job_ranks.setdefault(job_id, set()).add(self.worker_rank)

    def mark_failed(self, job_id: int) -> None:
        """Record a failed transfer while retaining its completion ack."""
        self.mark_completed(job_id)
        self.failed_jobs[job_id] = 1
        if self.worker_rank is not None:
            self.failed_job_ranks.setdefault(job_id, set()).add(self.worker_rank)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        failed = dict(self.failed_jobs)
        for job_id, v in other.failed_jobs.items():
            failed[job_id] = failed.get(job_id, 0) + v

        completed_ranks = {
            job_id: set(ranks)
            for job_id, ranks in self.completed_job_ranks.items()
        }
        failed_ranks = {
            job_id: set(ranks) for job_id, ranks in self.failed_job_ranks.items()
        }
        if self.worker_rank is not None:
            for job_id in self.completed_jobs:
                completed_ranks.setdefault(job_id, set()).add(self.worker_rank)
            for job_id in self.failed_jobs:
                failed_ranks.setdefault(job_id, set()).add(self.worker_rank)
        if other.worker_rank is not None:
            for job_id in other.completed_jobs:
                completed_ranks.setdefault(job_id, set()).add(other.worker_rank)
            for job_id in other.failed_jobs:
                failed_ranks.setdefault(job_id, set()).add(other.worker_rank)
        for job_id, ranks in other.completed_job_ranks.items():
            completed_ranks.setdefault(job_id, set()).update(ranks)
        for job_id, ranks in other.failed_job_ranks.items():
            failed_ranks.setdefault(job_id, set()).update(ranks)

        node_headroom_by_rank = {
            rank: sample for rank, sample in self.node_headroom_by_rank.items()
        }
        if self.worker_rank is not None and self.node_headroom is not None:
            node_headroom_by_rank[self.worker_rank] = self.node_headroom
        if other.worker_rank is not None and other.node_headroom is not None:
            node_headroom_by_rank[other.worker_rank] = other.node_headroom
        node_headroom_by_rank.update(other.node_headroom_by_rank)

        return OffloadingWorkerMetadata(
            worker_rank=None,
            completed_jobs=merged,
            completed_job_ranks=completed_ranks,
            failed_job_ranks=failed_ranks,
            failed_jobs=failed,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            node_headroom_by_rank=node_headroom_by_rank,
        )
