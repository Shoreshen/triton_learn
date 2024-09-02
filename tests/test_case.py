import torch

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

# @triton_heuristics.pointwise(
#     size_hints=[131072], 
#     filename=__file__,
#     triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
#     inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
#     min_elem_per_thread=0
# )
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-64) + x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tl_math.cos(tmp15)
    tmp17 = tmp0 * tmp16
    tmp18 = tl.load(in_ptr0 + (64 + x2), tmp5, other=0.0)
    tmp19 = -tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = tl.load(in_ptr0 + ((-64) + x2), tmp9, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp9, tmp22, tmp23)
    tmp25 = tl.where(tmp5, tmp21, tmp24)
    tmp26 = tl_math.sin(tmp15)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp17 + tmp27
    tl.store(out_ptr0 + (x2), tmp28, None)


def triton_poi_fused_add_cat_mul_1(in_ptr0, in_ptr1):
    xnumel = 131072
    grid = (128,)
    out_ptr = torch.empty(131072,device=in_ptr0.device)
    triton_[grid](in_ptr0, in_ptr1, out_ptr, xnumel, XBLOCK=1024)
    return out_ptr


def get_inputs():
    bs = 16
    q_len = 1
    hc = 64
    head_dim = 128
    q_shape = (bs, hc, q_len, head_dim)
    freqs_shape = (1, hc, 1)
    torch.manual_seed(0)
    q = torch.rand(q_shape, dtype=torch.float32, device="cuda")
    freqs = torch.rand(freqs_shape, dtype=torch.float32, device="cuda")
    print(freqs.shape)
    return (q, freqs)

q, freqs = get_inputs()

triton_op = triton_poi_fused_add_cat_mul_1(q, freqs)

print("q:")
for num in q[0][0][0][0:8]:
    print(f"\t{num:.{8}f}")
print("freq:")
for num in freqs[0][0:8]:
    print(f"\t{num[0]:.{8}f}")
print("result:")
for num in triton_op[0:8]:
    print(f"\t{num:.{8}f}")

