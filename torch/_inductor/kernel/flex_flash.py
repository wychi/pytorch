# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V

from .. import ir
from ..ir import FixedLayout, FlexibleLayout
from ..lowering import empty, empty_strided, lowerings
from ..runtime.runtime_utils import is_power_of_2, next_power_of_2
from ..select_algorithm import autotune_select_algorithm, SymbolicGridFn, TritonTemplate
from .flex_attention import (
    compute_forward_block_mn,
    compute_forward_inner,
    compute_next_offset_func,
    create_indices_fake,
    create_num_blocks_fake_generator,
    get_bounded_indices_func,
    get_fwd_subgraph_outputs,
    load_checked_2d,
    load_checked_block,
    maybe_realize,
)


aten = torch.ops.aten
prims = torch.ops.prims


try:
    from flash_attn.cute import flash_attn_func

    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    print(
        "Warning: CUTE implementation not available. Install with: pip install nvidia-cutlass-dsl>=4.1.0.dev0"
    )


from ..select_algorithm import ExternKernelChoice


def flash_attention_forward_kernel(
    query, key, value, scale, causal=False, **kwargs
):
    """Minimal flash attention forward kernel using CUTE implementation."""
    if not CUTE_AVAILABLE:
        raise RuntimeError("CUTE flash attention not available")
    breakpoint()
    # Convert to expected format for flash_attn_func
    # flash_attn_func expects (batch, seqlen, num_heads, head_dim)
    # flex_attention tensors are typically (batch, num_heads, seqlen, head_dim)
    q_transposed = query.transpose(1, 2)  # (batch, seqlen, num_heads, head_dim)
    k_transposed = key.transpose(1, 2)    # (batch, seqlen, num_heads, head_dim)
    v_transposed = value.transpose(1, 2)  # (batch, seqlen, num_heads, head_dim)
    # Call flash_attn_func
    output = flash_attn_func(
        q_transposed,
        k_transposed,
        v_transposed,
        softmax_scale=scale,
        causal=causal,
    )
    # Convert back to flex_attention format (batch, num_heads, seqlen, head_dim)
    return output.transpose(1, 2)


# Create the ExternKernelChoice
FlashAttentionExternKernelChoice = ExternKernelChoice(
    flash_attention_forward_kernel,
    name="flash_attention_forward_kernel",
)
