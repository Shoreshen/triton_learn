# Debug Triton

## Installation

### virtualenv

Better to install in python virtual environment

```bash
python -m venv <path/to/virtualenv> --prompt triton
```

Activate virtual environment

```bash
source <path/to/virtualenv>/bin/activate
```

Quit virtual environment

```bash
deactivate
```

### Source Code

#### LLVM

Build LLVM withDebug Symbols：

1. Clone repository：
   ```bash
   gti clone git@github.com:llvm/llvm-project.git <path/to/LLVM>
   cd <path/to/LLVM>
   ```
2. Compile
   ```bash
   cmake -G Ninja -S llvm -B build -DCMAKE_INSTALL_PREFIX=../bin -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_PARALLEL_COMPILE_JOBS=32 -DLLVM_PARALLEL_LINK_JOBS=4
   ninja
   ```

#### Triton

1. Clone repository：
   ```bash
   git clone git@github.com:triton-lang/triton.git <path/to/triton>
   cd <path/to/triton>
   ```
2. Install dependencies：
   ```bash
   pip install ninja cmake wheel
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/121
   ```
	1. Note have to install pre version of `pytorch` release version only support `triton-2.3.0`
3. Deleted installed `triton`：
   ```bash
   pip uninstall triton pytorch-triton
   ```
5. Compile & install：
   ```bash
   cd <path/to/triton>
   LLVM_INCLUDE_DIRS=<path/to/LLVM>/build/include LLVM_LIBRARY_DIR=<path/to/LLVM>/build/lib LLVM_SYSPATH=<path/to/LLVM>/build DEBUG=1 MAX_JOBS=8 pip install -e python
   ```
	1. `DEBUG=1`: Compile in debug mode
	2. `LLVM_INCLUDE_DIRS`, `LLVM_LIBRARY_DIR` and `LLVM_SYSPATH` using [Compiled LLVM](#llvm) to build `triton`
	3. `MAX_JOBS`: Number of jobs to compile `triton`, prevent memory overflow

## Debug with VSCode

### Config file

`launch.json` as follows:

```json
{
"version": "0.2.0",
	"configurations": [
		{
			"name": "Python: triton_compile_test.py",
			"type": "debugpy",
			"request": "launch",
			"program": "${workspaceFolder}/tests/triton_compile_test.py",
			"console": "integratedTerminal",
			"env": {
				// "MLIR_ENABLE_DUMP": "1",
				// "LLVM_IR_ENABLE_DUMP": "1",
				// "TRITON_ENABLE_LLVM_DEBUG": "1"
			}
		},
		{
			"name": "(gdb) Attach",
			"type": "cppdbg",
			"request": "attach",
			"program": "${workspaceFolder}/.venv/bin/python",
			"processId": "${command:pickProcess}",
			"MIMode": "gdb",
			"setupCommands": [
				{
					"description": "Enable pretty-printing for gdb",
					"text": "-enable-pretty-printing",
					"ignoreFailures": true
				}
			]
		}
	]
}
```

### Test case

```py
import torch
import torch.nn as nn

class SiluModel(nn.Module):
	def __init__(self):
		super(SiluModel, self).__init__()
		self.act_fn = nn.functional.sigmoid
	def forward(self, x, up):
		x = self.act_fn(x) * x * up
		return x

def get_input(rank_id):
	input_ids = torch.rand((32, 1, 896), dtype=torch.float32)
	up = torch.rand((32, 1, 896), dtype=torch.float32)
	input_ids = input_ids.to("cuda:" + str(rank_id))
	up = up.to("cuda:" + str(rank_id))
	return input_ids, up

if __name__ == '__main__':
	model = SiluModel().to("cuda")
	model.eval()
	opt_model = torch.compile(model)
	input_ids, up = get_input(0)
	output = opt_model(input_ids, up)
	print("debug output:", output)
```

### Debug steps

1. Select [virtual environment](#virtualenv) python interpreter
2. For any `pybind11` cpp function, e.g. `<path/to/triton>/third_party/nvidia/backend/compiler.py:140~152`
   ```py
   @staticmethod
   def make_ttir(mod, metadata, opt):
		pm = ir.pass_manager(mod.context)
		pm.enable_debug()
		passes.common.add_inliner(pm)
		passes.ttir.add_rewrite_tensor_pointer(pm)
		passes.ttir.add_combine(pm)
		passes.common.add_canonicalizer(pm)
		passes.ttir.add_reorder_broadcast(pm)
		passes.common.add_cse(pm)
		passes.common.add_licm(pm)
		passes.common.add_symbol_dce(pm)
		pm.run(mod)    #<------ Breakpoint here
		return mod
   ```
3. Launch `Python: triton_compile_test.py` and break at the above breakpoint
4. Breakpoint will stop at one of the subprocesses, need to attach to that process
    <img src=./pic/img1.png>
5. Launch `(gdb) Attach` and attach to the target process (`PID: 180302` for the above example)
6. Set breakpoint at the target cpp file and continue `Python: triton_compile_test.py`, if it hit the breakpoint, GDB will stop at the target cpp file
    <img src=./pic/img2.png>