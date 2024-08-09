import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    SIZE: tl.constexpr
):
    offsets = tl.arange(0, SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def add(x: torch.Tensor, y: torch.Tensor):
    
    output = torch.empty_like(x)
    size = output.numel()
    grid = (1,)
    add_kernel[grid](x, y, output, size)
    return output

size = 1024
torch.manual_seed(0)
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

torch_output = x + y
triton_output = add(x, y)

rtol = 0 # for nvidia, the tolerance is 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
