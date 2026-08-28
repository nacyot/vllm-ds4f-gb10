# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    is_kv_cache_tensor_packed,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        # Non-writers still ack: pending_count waits for world_size per job.
        self._is_store_writer = (
            not self.spec.replicated_layout or self.spec.config.parallel.rank == 0
        )

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        # [load-error-recompute] job_id -> dst GPU block ids of in-flight
        # loads, so a failed job can be reported block-by-block.
        self._load_job_block_ids: dict[int, list[int]] = {}
        # [load-error-recompute] GPU block ids whose KV load failed; drained
        # once per step by get_block_ids_with_load_errors().
        self._load_error_block_ids: set[int] = set()
        # [load-error-recompute] req_ids whose load job failed at submit and
        # never reached the backend worker; they must still be reported via
        # get_finished() so the scheduler resumes them.
        self._failed_load_req_ids: set[ReqId] = set()
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        kv_cache_config = self.kv_cache_config
        num_blocks = kv_cache_config.num_blocks
        _ktc = self.vllm_config.kv_transfer_config
        _compact_packed = bool(
            _ktc is not None
            and _ktc.kv_connector_extra_config.get("dspark_compact_packed", False)
        )
        mappings = derive_canonical_mappings(
            self.vllm_config, kv_cache_config, kv_caches
        )

        # Packed layouts (e.g. DSv4) set block_stride > 0; their tensors use
        # stride(0) as the manager-block stride (equals total_num_bytes_per_block).
        # General (non-packed) layouts size the tensor at page_size_bytes per
        # manager block, so page_size_bytes is the correct offloading stride.
        layer_is_packed: dict[str, bool] = {
            ln: is_kv_cache_tensor_packed(kv_tensor)
            for kv_tensor in kv_cache_config.kv_cache_tensors
            for ln in kv_tensor.shared_by
        }

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)

                    page = layer_kv_cache_spec.page_size_bytes
                    elem_size = layer_kv_cache.element_size()
                    byte_offset = layer_kv_cache.storage_offset() * elem_size
                    block_stride_bytes = (
                        layer_kv_cache.stride(0) * elem_size
                        if layer_is_packed[layer_name]
                        else page
                    )
                    raw = torch.empty(
                        0,
                        dtype=torch.int8,
                        device=layer_kv_cache.device,
                    ).set_(layer_kv_cache.untyped_storage())
                    tensors_per_block[layer_name] = (
                        torch.as_strided(
                            raw,
                            (num_blocks, page),
                            (block_stride_bytes, 1),
                            byte_offset,
                        ),
                    )
                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.unpadded_page_size_bytes
                    )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert layer_kv_cache.dtype == torch.int8
                    tensors_per_block[layer_name] = (
                        layer_kv_cache.view(
                            num_blocks, layer_kv_cache_spec.page_size_bytes
                        ),
                    )

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        packed_kv_cache_tensor = next(
            (
                t
                for t in kv_cache_config.kv_cache_tensors
                if is_kv_cache_tensor_packed(t) and t.shared_by
            ),
            None,
        )
        if packed_kv_cache_tensor is not None and not _compact_packed:
            (tensor,) = tensors_per_block[packed_kv_cache_tensor.shared_by[0]]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [
                        [CanonicalKVCacheRef(0, block_stride)]
                        for _ in kv_cache_config.kv_cache_groups
                    ],
                )
            )
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        if _compact_packed and packed_kv_cache_tensor is not None:
            # Packed slab: per-layer strided views already carry each layer's
            # true page; build one canonical tensor + ref per layer instead of
            # charging every group the full slab stride.
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                for layer_name in kv_cache_group.layer_names:
                    if layer_name not in tensors_per_block:
                        continue
                    (tensor,) = tensors_per_block[layer_name]
                    block_tensors.append(
                        CanonicalKVCacheTensor(
                            tensor=tensor,
                            page_size_bytes=page_size_bytes[layer_name],
                        )
                    )
                    mapping = mappings.get(layer_name)
                    assert (
                        mapping is None
                        or mapping.local_page_size_bytes
                        == unpadded_page_size_bytes[layer_name]
                    )
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=len(block_tensors) - 1,
                            page_size_bytes=unpadded_page_size_bytes[layer_name],
                            mapping=mapping,
                        )
                    )
        else:
          for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
              # Filter to layers that were actually processed above.
              # Packed KV allocation emits KVCacheTensor entries for
              # every (tuple_idx, page_size) slot; slots where no group has a
              # layer at that index produce an empty shared_by (reserved memory
              # with no corresponding model layer).
              tensor_layer_names = [
                  n for n in kv_cache_tensor.shared_by if n in tensors_per_block
              ]
              if not tensor_layer_names:
                  continue

              # verify all layers in the group reference the exact same tensors
              assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
              assert (
                  len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                  == 1
              )
              assert (
                  len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
              )

              # pick the first layer to represent the group
              first_layer_name = tensor_layer_names[0]
              for tensor in tensors_per_block[first_layer_name]:
                  block_tensors.append(
                      CanonicalKVCacheTensor(
                          tensor=tensor,
                          page_size_bytes=page_size_bytes[first_layer_name],
                      )
                  )

                  curr_tensor_idx = len(block_tensors) - 1
                  for layer_name in tensor_layer_names:
                      mapping = (
                          mappings.get(layer_name)
                          if len(tensors_per_block[first_layer_name]) == 1
                          else None
                      )
                      assert (
                          mapping is None
                          or mapping.local_page_size_bytes
                          == unpadded_page_size_bytes[layer_name]
                      )
                      block_data_refs[layer_name].append(
                          CanonicalKVCacheRef(
                              tensor_idx=curr_tensor_idx,
                              page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                              mapping=mapping,
                          )
                      )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._init_worker(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            self._load_job_block_ids[job_id] = [
                int(b) for b in entry.dst_spec.block_ids
            ]
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            if not success:
                # [load-error-recompute] The load never reached the backend
                # worker. Report every dst block as failed so the scheduler
                # recomputes them (kv_load_failure_policy=recompute), and
                # synthesize the job completion so the request does not
                # hang in WAITING_FOR_REMOTE_KVS and the manager unpins
                # the staged chunks.
                block_ids = self._load_job_block_ids.pop(job_id)
                logger.error(
                    "[load-error-recompute] submit_load failed: job=%d "
                    "req=%s blocks=%d; scheduling recompute",
                    job_id,
                    entry.req_id,
                    len(block_ids),
                )
                self._load_error_block_ids.update(block_ids)
                self._connector_worker_meta.mark_completed(job_id)
                req_id = self._load_jobs.pop(job_id, None)
                if req_id is not None:
                    self._failed_load_req_ids.add(req_id)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        # [load-error-recompute] loads that failed at submit still resolve
        # the request; their blocks were already reported as failed.
        if self._failed_load_req_ids:
            finished_recving |= self._failed_load_req_ids
            self._failed_load_req_ids = set()
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            if not transfer_result.success:
                if job_id not in self._load_jobs:
                    # [load-error-recompute] Failed *store*: keep the
                    # upstream fail-stop semantics. The current backend
                    # never emits success=False (cpu/gpu_worker.py builds
                    # TransferResult with success=True unconditionally);
                    # if a future backend does, completing the job would
                    # let the scheduler's complete_store mark garbage
                    # staging bytes as stored, and later lookups would
                    # serve them as HITs (silent corruption). Fail loudly
                    # instead (explicit raise, -O safe).
                    raise AssertionError(
                        "[load-error-recompute] store job %d reported "
                        "failure; refusing to complete it as stored"
                        % job_id
                    )
                # [load-error-recompute] Failed *load*: report the dst GPU
                # blocks for recompute instead of asserting. Per the
                # KVConnectorBase_V1 contract the job is still marked
                # completed and the request still resolves below.
                failed_block_ids = self._load_job_block_ids.get(job_id)
                if failed_block_ids:
                    self._load_error_block_ids.update(failed_block_ids)
                logger.error(
                    "[load-error-recompute] load transfer failed: job=%d "
                    "blocks=%d; scheduling recompute",
                    job_id,
                    len(failed_block_ids) if failed_block_ids else 0,
                )
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            self._load_job_block_ids.pop(job_id, None)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """[load-error-recompute] Drain GPU block ids whose KV load failed.

        Two sources are merged:
        - connector-level failures (submit_load refusals and failed
          TransferResults) collected in self._load_error_block_ids, and
        - backend fs pread failures recorded by the offloading spec worker
          (TieringOffloadingSpec zero-fill fallback stashes them on the
          worker object as _dspark_load_error_block_ids).

        Called once per engine step by the model runner after
        get_finished(); the executor unions the sets across TP ranks
        (KVOutputAggregator) and the core scheduler recomputes the
        affected blocks when kv_load_failure_policy=recompute.
        """
        errors = self._load_error_block_ids
        self._load_error_block_ids = set()
        spec_errors = getattr(
            self.worker, "_dspark_load_error_block_ids", None
        )
        if spec_errors:
            errors |= spec_errors
            spec_errors.clear()
        # Never report the shared null placeholder block: sliding-window
        # groups pad with it, and marking it invalid would truncate every
        # request at its first null position.
        errors.discard(0)
        return errors

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._load_job_block_ids.clear()
        self._load_error_block_ids.clear()
        self._failed_load_req_ids.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
