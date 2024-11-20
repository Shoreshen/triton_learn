import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_m16n16k64(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr
):
    offsets_a = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
    offsets_b = tl.arange(0, 64)[:, None] * 16 + tl.arange(0, 16)[None, :]
    offsets_c = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    
    c = tl.dot(a, b)
    c = c.to(tl.float16)

    c_ptrs = c_ptr + offsets_c
    tl.store(c_ptrs, c)

def matmul_m16n16k64(a, b):
    c = torch.empty((16, 16), device=a.device, dtype=torch.float16)
    grid = (1,)
    matmul_kernel_m16n16k64[grid](a, b, c)
    return c

torch.manual_seed(0)
a = torch.randn((16, 64), device='cuda', dtype=torch.float16)
b = torch.randn((64, 16), device='cuda', dtype=torch.float16)

triton_output = matmul_m16n16k64(a, b)
torch_output = torch.matmul(a, b)

rtol = 0 # for nvidia, the tolerance is 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
