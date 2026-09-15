# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
}


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Each rank offloads its assigned hash heads, so host table storage and
    lookup traffic scale with the number of heads assigned to the rank."""

    mmap: bool = False
    """Leave the fp8 tables in the checkpoint's safetensors shards and map
    them read-only instead of loading them. The lookup kernel reads rows
    through the mapping, so the GPU must share the CPU's address space (GB10
    and GH200 in ATS mode); the page cache is the only copy of the table and
    the OS reclaims it under memory pressure. Before each lookup, CPU threads
    prefault the pages of the rows about to be read so the GPU never blocks
    on a disk read. Overrides `cpu_offload`."""

    mmap_prefault_threads: int = 32
    """Threads issuing madvise(MADV_POPULATE_READ) before an mmap lookup."""

    mmap_release_after_steps: int = 3
    """Drop the pages of the rows prefaulted this many steps ago (madvise
    MADV_DONTNEED, then posix_fadvise DONTNEED on the shard), so the page
    cache holds only the last few steps' rows instead of every row ever
    read; 0 leaves reclaim to the kernel."""

    mmap_prefetch_next_chunk: bool = False
    """Hash the tokens each prefilling request continues with after the
    current step and populate their pages in a background thread while the
    step runs, and run the page releases there too. The step's own
    prefault then finds its rows resident, so the forward no longer waits
    on the disk reads (about 0.3-0.8 s per 4K chunk on GB10)."""

    mmap_decode_async: int = 0
    """Decode-only steps (no prefill in the batch) populate the pages of
    some tables in a background thread and launch the forward without
    waiting for them: 0 waits for every table, 1 waits only for the first
    table (the one the forward reads first) and populates the rest in the
    background, 2 populates every table in the background and the lookup
    faults in place whatever is still cold. Needs `mmap_release_after_steps`
    0; the tables fall back to waiting otherwise."""

    mmap_min_chunk_runs: int = 1
    """Smallest number of page runs per thread-pool chunk of a prefault or
    release. With 1 a decode-sized prefault of ~72 runs dispatches ~72
    one-run chunks."""

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings. Currently only the CUDA DeepSeek V4.1 "
                f"implementation with non-empty {field or 'engram_layer_ids'} "
                "is supported."
            )

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
