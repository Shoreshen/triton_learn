import torch

import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output)



def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, )](
        y,
        x,
        num_warps=num_warps,
        BLOCK_SIZE=n_cols,
    )
    return y

def get_inputs():
    x_shape = (4, 32)
    x = torch.rand(x_shape, dtype=torch.float32)
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            x[i][j] = i*32+j
    return x

x = get_inputs()
print(x)
triton_output = softmax(x)
print(triton_output)
