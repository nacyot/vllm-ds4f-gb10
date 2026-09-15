# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side token-to-request mapping from upstream PR #56562."""

from vllm.triton_utils import tl, triton


@triton.jit
def _token_request(query_start_loc, token, num_reqs):
    lo = tl.full(token.shape, 0, tl.int32)
    hi = tl.full(token.shape, num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
        right = (lo < hi) & (end <= token)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    return tl.minimum(lo, num_reqs - 1)


@triton.jit(do_not_specialize=["num_reqs", "num_mapped", "num_tokens"])
def _token_request_mapping_kernel(
    query_start_loc, output, num_reqs, num_mapped, num_tokens
):
    token = tl.program_id(0) * 256 + tl.arange(0, 256)
    req = _token_request(query_start_loc, token, num_reqs)
    tl.store(output + token, tl.where(token < num_mapped, req, 0), token < num_tokens)
