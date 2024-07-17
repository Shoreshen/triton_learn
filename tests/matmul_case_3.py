import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({}, num_warps=8)],
    key=[],
)
@triton.jit
def matmul_kernel_mnk(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, 
    N: tl.constexpr, 
    K: tl.constexpr
):
    offsets_a = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    offsets_b = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    offsets_c = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    c_ptrs = c_ptr + offsets_c

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    
    c = tl.dot(a, b)
    c = c.to(tl.float16)

    tl.store(c_ptrs, c)

def matmul_mnk(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (1,)
    matmul_kernel_mnk[grid](a, b, c, M, N, K)
    return c

torch.manual_seed(0)
a = torch.randn((512, 64), device='cuda', dtype=torch.float16)
b = torch.randn((64, 256), device='cuda', dtype=torch.float16)

triton_output = matmul_mnk(a, b)
torch_output = torch.matmul(a, b)

rtol = 0 # for nvidia, the tolerance is 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")