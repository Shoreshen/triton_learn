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
   cd build
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
   pip install ninja cmake wheel setuptools
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
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
                // "TRITON_ENABLE_LLVM_DEBUG": "1",
                "TRITON_CACHE_DIR" : "${workspaceFolder}/.cache" //triton will cache compiled result here and reuse it 
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


# Triton Source Code

## Add New Backend

### Python Runtime Loading of New Backend

During runtime, triton use [`__init__.py`](./triton-project/python/triton/backends/__init__.py) to search all available backends under the same directory:

```py
def _find_concrete_subclasses(module, base_class):
    ret = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: BaseBackend = None
    driver: DriverBase = None


def _discover_backends():
    backends = dict()
    root = os.path.dirname(__file__)
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)): # has to be a directory
            continue
        if name.startswith('__'): # exclude __pycache__
            continue
        compiler = _load_module(name, os.path.join(root, name, 'compiler.py'))  # same as import compiler
        driver = _load_module(name, os.path.join(root, name, 'driver.py'))      # same as import driver
        backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                 _find_concrete_subclasses(driver, DriverBase))
    return backends


backends = _discover_backends()
```

From the above, we know that:

1. A new backend need to be a directory.
2. Under the new directory, we need python files `compiler.py` and `driver.py`.
3. The `compiler.py` must have exactly one subclass of `BaseBackend`.
4. and `driver.py` must have exactly one subclass of `DriverBase`.
5. The search path is `triton/backends/`.

<a id = "get_current_target"></a>

Furthermore, definition of `BaseBackend` follows:

```py
class DriverBase(metaclass=ABCMeta):

    @abstractclassmethod
    def is_active(self):
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    def __init__(self) -> None:
        pass
```

Which indicate that `is_active`, `get_current_target`, `__init__` should be realized in `driver.py`.

### Compilation of New Backend

#### [setup.py](./triton-project/python/setup.py)

<a id = "setuppy"></a>

Using `BackendInstaller` class to initialize backends, its a lise of `Backend` objects (Detailed implementation of `BackendInstaller` and `Backend` class can be found in [Appendix](#definiton-of-backendinstaller-and-backend-class).):

```py
# Initialize both internal ("nvidia", "amd") and external backends
backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()] 
```

Add relative path to `cmake` building variables:

```py
class CMakeBuild(build_ext):
    ...
    def build_extension(self, ext):
        lit_dir = shutil.which('lit')
        ninja_dir = shutil.which('ninja')
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages([get_pybind11_package_info(), get_llvm_package_info()])
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = sysconfig.get_path("platinclude")
        cmake_args = [
            "-G", "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON", "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON", "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS=" + ';'.join([b.name for b in backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS=" + ';'.join([b.src_dir for b in backends if b.is_external])
        ]
    ...
```

Add symbol link:

```py
def add_link_to_backends():
    for backend in backends:
        if os.path.islink(backend.install_dir):
            os.unlink(backend.install_dir)
        if os.path.exists(backend.install_dir):
            shutil.rmtree(backend.install_dir)
        os.symlink(backend.backend_dir, backend.install_dir)
```

With the above steps, it passes necessary parameters to [CmakeLists.txt](#cmakelists.txt) and links backend directories for [run time](#python-runtime-loading-of-new-backend).

#### [CMakeLists.txt](./triton-project/CMakeLists.txt)

Appending `TRITON_CODEGEN_BACKENDS` and `TRITON_PLUGIN_DIRS` passed from [`setup.py`](#setuppy) to `TRITON_BACKENDS_TUPLE` and passing to [C/C++ compilation](#cc-compilation):

```cmake
...
  if (DEFINED TRITON_PLUGIN_DIRS)
    foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
      # Read the plugin name under dir/backend/name.conf
      cmake_path(APPEND PLUGIN_DIR "backend" "name.conf" OUTPUT_VARIABLE PLUGIN_NAME_PATH)
      file(READ ${PLUGIN_NAME_PATH} PLUGIN_NAME)
      string(STRIP ${PLUGIN_NAME} PLUGIN_NAME)

      list(APPEND TRITON_PLUGIN_NAMES ${PLUGIN_NAME})

      # Include the plugin as part of the build, placing the build output under
      # ${TRITON_BINARY_DIR}/third_party/${PLUGIN_NAME}
      cmake_path(APPEND TRITON_BINARY_DIR "third_party" ${PLUGIN_NAME} OUTPUT_VARIABLE PLUGIN_DIR_BUILD_OUTPUT)
      message(STATUS "Building plugin '${PLUGIN_NAME}' from ${PLUGIN_DIR} with output ${PLUGIN_DIR_BUILD_OUTPUT}")
      add_subdirectory(${PLUGIN_DIR} ${PLUGIN_DIR_BUILD_OUTPUT})
    endforeach()
  endif()
...
  # Define triton library
  string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_CODEGEN_BACKENDS})
...
```

#### C/C++ compilation

In [`main.cc`](./triton-project/python/src/main.cc) file, it uses `TRITON_BACKENDS_TUPLE` to:

1. Declare `init_triton_<backend name>` function using macro
    ```cpp
    FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)
    ```
2. Call `init_triton_<backend name>` function using macro
    ```cpp
    FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
    ```

### Summary

With the above illustration, the necessary steps to add a new triton backend are:

1. Create a new directory under `triton-project/third_party` assume the name is `new_backend`
2. Under `triton-project/third_party/new_backend` add:
   1. `compiler.py` import `BaseBackend` from `triton.backends.compiler`, create exactly one subclass of `BaseBackend`
   2. `driver.py` import `GPUDriver` from `triton.backends.driver`, create exactly one subclass of `DriverBase` (since `GPUDriver` is the subclass of `DriverBase`) and realize
3. Create a `cpp/cc` file under `triton-project/third_party/new_backend`, include at least one function with the following signature:
   ```cpp
   #include <pybind11/pybind11.h>
   #include <pybind11/stl.h>
   #include <pybind11/stl_bind.h>
   extern "C" void init_triton_new_backend(py::module &&m) {
       ...
   }
   ```
4. Update [setup.py](./triton-project/python/setup.py) add new backend name to `backends` list
   ```py
   backends = [*BackendInstaller.copy(["nvidia", "amd", "new_backend"]), *BackendInstaller.copy_externals()]
   ```
5. Add `CMakeLists.txt` file under `triton-project/third_party/new_backend`, examples from nvidia as follow:
   ```cmake
   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
   include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
   add_subdirectory(include)
   add_subdirectory(lib)
   if(TRITON_BUILD_PYTHON_MODULE)
     add_triton_plugin(TritonNVIDIA ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc LINK_LIBS TritonNVIDIAGPUToLLVM NVGPUToLLVM)
   endif()
   ```

## Compilation process

### Step 1

Start of compiling process by using `KernelInterface(Generic[T]):` class as `kernel_fun[grid](..., BLOCK_SIZE=1024)` it will jump to [jit.py](./triton-project/python/triton/runtime/jit.py) to `run` function:

```py
class JITFunction(KernelInterface[T]):
    ...
    def run(self, *args, grid, warmup, **kwargs):
        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        ...
        if kernel is None:
            # Kernel is not cached; we have to compile.
            target = driver.active.get_current_target()
            backend = self.make_backend(target)
            options = backend.parse_options(kwargs)
            ...
            # compile the kernel
            src = self.ASTSource(self, signature, constants, configs[0])
            kernel = self.compile(
                src,
                target=target,
                options=options.__dict__,
            )
            ...
```

Note that:

1. `driver.active` calling [_create_driver()](./triton-project/python/triton/runtime/driver.py#L5) function to get the aviliable driver class (which is the only subclass of `DriverBase`)
    ```py
    def _create_driver():
        actives = [x.driver for x in backends.values() if x.driver.is_active()]
        if len(actives) != 1:
            raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
        return actives[0]()
    ```
2. `get_current_target()` is defined in [previous section](#get_current_target)
3. `make_backend(target)` calling the function [`make_backend(target)`](./triton-project/python/triton/compiler/compiler.py#L310):
    ```py
    def make_backend(target):
        actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
        if len(actives) != 1:
            raise RuntimeError(
                f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
        return actives[0](target)
    ```

### Step 2

By calling [self.compile](./triton-project/python/triton/compiler/compiler.py#L226) from [last step](#step-1) function to start compilation of the kernel.

This step will cache compiled code in `TRITON_CACHE_DIR` and reuse it if the same kernel is called again. To make sure debug works as normal, the cached code should be deleted before re-run.

The main compilation process focuses on the following code:
```py
def compile(src, target=None, options=None):
    ...
    # run compilation pipeline  and populate metadata
    ...
    try:
        module = src.make_ir(options, codegen_fns, context)
    except Exception as e:
        filter_traceback(e)
        raise
    ...
```

By calling function [`src.make_ir`](./triton-project/python/triton/runtime/jit.py#L1252) it creates AST (by calling [`JITFunction::parse()`](./triton-project/python/triton/runtime/jit.py#L774)) and first layer of ir object (by calling [`generator.visit(fn.parse())`](./triton-project/python/triton/runtime/jit.py#L1165)):

```py
def ast_to_ttir(fn, specialization, context, options, codegen_fns):
    ...
    generator.visit(fn.parse())

    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret
```

The `fn.parse()` command will call python's parser, and return python ast.

### Step 3

Setting up each lowering stage in the compilation pipeline, the main compilation process focuses on the following code:

```py
def compile(src, target=None, options=None):
    ...
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation()
    try:
        module = src.make_ir(options, codegen_fns, context)
    except Exception as e:
        filter_traceback(e)
        raise
    use_ttgir_loc = os.environ.get("USE_TTGIR_LOC", "0") == "1"
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        if (fn_override_manager is not None and fn_override_manager.has_file(ir_filename)):
            print(f"\nOverriding kernel with file {ir_filename}")
            full_name = fn_override_manager.get_file(ir_filename)
            next_module = parse(full_name, ext, context)
        # use an env variable to parse ttgir from file
        if use_ttgir_loc and ext == "ttgir":
            ttgir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ttgir_full_name)
            print(f"Create new locations for {ttgir_full_name}")
        module = next_module
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)
```

The `backend.load_dialects(context)` will call backend specific (Nvidia, amd, ...). Each stage will try to lower `module` and produce `next_module`.

## Auto tunner

By adding `@triton.autotune` decorator to the function, it allows users to pass in multiple set of configurations, and allow users to set system configurations of the following:

1. `num_warps`: Number of warps per CTA, so the total thread number of a CTA is `32 * num_warps`
2. `num_stages`: Number of wait group can be applied, see  
   1. GPU will provide async instructions, and thread can issue them. 
   2. The issued async instructions than be grouped (so as wait group), and syncronization instructions can be used to wait for specific group to be finished.
   3. See [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async), [`cp.async.commit_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group) and [`cp.async.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group) as example. Also see [Asynchronous Operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#program-order-async-operations) section

# Triton lowing IRs

## Source Python Code

```py
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

## Triton IR

```llvm
module {
  tt.func public @matmul_kernel(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},   ; a_ptr
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},   ; b_ptr
    %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32},   ; c_ptr
    %arg3: i32 {tt.divisibility = 16 : i32},            ; 
    %arg4: i32 {tt.divisibility = 16 : i32},            ; 
    %arg5: i32 {tt.divisibility = 16 : i32},            ; K
    %arg6: i32 {tt.divisibility = 16 : i32},            ; stride_am
    %arg7: i32 {tt.divisibility = 16 : i32},            ; stride_bk
    %arg8: i32 {tt.divisibility = 16 : i32}             ; stride_cm
  ) attributes {noinline = false} {
; First part: Calculating group partition and dimensions in group (Group is in size of [GROUP_SIZE_M, tl.cdiv(M, BLOCK_SIZE_M)])
    ; pid = tl.program_id(axis=0)
    %0 = tt.get_program_id x : i32
    ; num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    %1 = tt.call @cdiv__i32__1cconstexpr_128_(%arg3) : (i32) -> i32
    ; num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    %2 = tt.call @cdiv__i32__1cconstexpr_256_(%arg4) : (i32) -> i32
    ; constant: GROUP_SIZE_M
    %c8_i32 = arith.constant 8 : i32
    ; num_pid_in_group = GROUP_SIZE_M * num_pid_n
    %3 = arith.muli %2, %c8_i32 : i32
    ; group_id = pid // num_pid_in_group
    %4 = arith.divsi %0, %3 : i32
    ; constant: GROUP_SIZE_M
    %c8_i32_0 = arith.constant 8 : i32
    ; first_pid_m = group_id * GROUP_SIZE_M
    %5 = arith.muli %4, %c8_i32_0 : i32
    ; num_pid_m - first_pid_m
    %6 = arith.subi %1, %5 : i32
    ; constant: GROUP_SIZE_M
    %c8_i32_1 = arith.constant 8 : i32
    ; group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    %7 = arith.minsi %6, %c8_i32_1 : i32
    ; pid % num_pid_in_group
    %8 = arith.remsi %0, %3 : i32
    ; ((pid % num_pid_in_group) % group_size_m)
    %9 = arith.remsi %8, %7 : i32
    ; pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    %10 = arith.addi %5, %9 : i32
    ; pid % num_pid_in_group
    %11 = arith.remsi %0, %3 : i32
    ; pid_n = (pid % num_pid_in_group) // group_size_m
    %12 = arith.divsi %11, %7 : i32

; Second part: Calculating data (matrix block) pointers
    ; constant: BLOCK_SIZE_M
    %c128_i32 = arith.constant 128 : i32
    ; pid_m * BLOCK_SIZE_M
    %13 = arith.muli %10, %c128_i32 : i32
    ; tl.arange(0, BLOCK_SIZE_M)
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    ; pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    %15 = tt.splat %13 : i32 -> tensor<128xi32> ; make pid_m * BLOCK_SIZE_M the same dimension as tl.arange(0, BLOCK_SIZE_M)
    %16 = arith.addi %15, %14 : tensor<128xi32>
    ; offs_am = ((pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)) % M
    %17 = tt.splat %arg3 : i32 -> tensor<128xi32> ; Make M the same dimension as ((pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M))
    %18 = arith.remsi %16, %17 : tensor<128xi32>
    ; constant: BLOCK_SIZE_N
    %c256_i32 = arith.constant 256 : i32
    ; pid_n * BLOCK_SIZE_N
    %19 = arith.muli %12, %c256_i32 : i32
    ; tl.arange(0, BLOCK_SIZE_N)
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    ; pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    %21 = tt.splat %19 : i32 -> tensor<256xi32> ; make pid_n * BLOCK_SIZE_N the same dimension as tl.arange(0, BLOCK_SIZE_N)
    %22 = arith.addi %21, %20 : tensor<256xi32>
    ; offs_bn = ((pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)) % N
    %23 = tt.splat %arg4 : i32 -> tensor<256xi32> ; Make N the same dimension as ((pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N))
    %24 = arith.remsi %22, %23 : tensor<256xi32>
    ; offs_k = tl.arange(0, BLOCK_SIZE_K)
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    ; offs_am[:, None]
    %26 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    ; (offs_am[:, None]) * stride_am
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32> ; Make stride_am the same dimension as (offs_am[:, None])
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    ; offs_k[None, :]
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    ; offs_k[None, :] * stride_ak
    %c1_i32 = arith.constant 1 : i32 ; stride_ak optimized to constant
    %cst = arith.constant dense<1> : tensor<1x64xi32> ; expend to the same dimention of offs_k[None, :]
    %30 = arith.muli %29, %cst : tensor<1x64xi32>
    ; ((offs_am[:, None]) * stride_am) + (offs_k[None, :])
    %31 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32> ; expand (offs_am[:, None]) * stride_am) to dimention of [dim(offs_am[:, None]) * stride_am), dim(offs_k[None, :])]
    %32 = tt.broadcast %30 : tensor<1x64xi32> -> tensor<128x64xi32> ; expand (offs_k[None, :]) * stride_ak) to dimention of [dim(offs_am[:, None]) * stride_am), dim(offs_k[None, :])]
    %33 = arith.addi %31, %32 : tensor<128x64xi32>
    ; a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    %34 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>> ; Make a_ptr the same dimension as (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    %35 = tt.addptr %34, %33 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    ; offs_k[:, None]
    %36 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    ; (offs_k[:, None]) * stride_bk
    %37 = tt.splat %arg7 : i32 -> tensor<64x1xi32> ; Make stride_bn the same dimension as (offs_k[:, None])
    %38 = arith.muli %36, %37 : tensor<64x1xi32> 
    ; offs_bn[None, :]
    %39 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    ; (offs_bn[None, :]) * stride_bn
    %c1_i32_2 = arith.constant 1 : i32 ; stride_bn optimized to constant
    %cst_3 = arith.constant dense<1> : tensor<1x256xi32> ; expend to the same dimention of offs_bn[None, :]
    %40 = arith.muli %39, %cst_3 : tensor<1x256xi32>
    ; ((offs_k[:, None]) * stride_bk) + ((offs_bn[None, :]) * stride_bn)
    %41 = tt.broadcast %38 : tensor<64x1xi32> -> tensor<64x256xi32> ; expand (offs_k[:, None]) * stride_bk) to dimention of [dim(offs_k[:, None]) * stride_bk), dim(offs_bn[None, :])]
    %42 = tt.broadcast %40 : tensor<1x256xi32> -> tensor<64x256xi32> ; expand (offs_bn[None, :]) * stride_bn) to dimention of [dim(offs_k[:, None]) * stride_bk), dim(offs_bn[None, :])]
    %43 = arith.addi %41, %42 : tensor<64x256xi32>
    ; b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    %44 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>> ; Make b_ptr the same dimension as (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    %45 = tt.addptr %44, %43 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
    ; accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    %46 = tt.call @"zeros____0cconstexpr_(constexpr_128_, constexpr_256_)__1cconstexpr_fp32_"() : () -> tensor<128x256xf32>
    ; tl.cdiv(K, BLOCK_SIZE_K)
    %47 = tt.call @cdiv__i32__1cconstexpr_64_(%arg5) : (i32) -> i32

; Third part: Main loop
    ; for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))
    %c0_i32 = arith.constant 0 : i32 ; start from 0
    %c1_i32_4 = arith.constant 1 : i32 ; step by 1
    %48 = arith.bitcast %c0_i32 : i32 to i32 ; type convert
    %49 = arith.bitcast %47 : i32 to i32 ; type convert
    %50 = arith.bitcast %c1_i32_4 : i32 to i32 ; type convert
    %51 = llvm.mlir.undef : i32 ; no use
    %52:3 = scf.for %arg9 = %48 to %49 step %50 iter_args(%arg10 = %46, %arg11 = %35, %arg12 = %45) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>)  : i32 {
      ; offs_k[None, :]
      %81 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      ; constant: BLOCK_SIZE_K
      %c64_i32 = arith.constant 64 : i32
      ; k * BLOCK_SIZE_K
      %82 = arith.muli %arg9, %c64_i32 : i32
      ; K - (k * BLOCK_SIZE_K)
      %83 = arith.subi %arg5, %82 : i32
      ; (offs_k[None, :]) < (K - (k * BLOCK_SIZE_K))
      %84 = tt.splat %83 : i32 -> tensor<1x64xi32> ; Make K - (k * BLOCK_SIZE_K) the same dimension as offs_k[None, :]
      %85 = arith.cmpi slt, %81, %84 : tensor<1x64xi32>
      ; a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
      %cst_9 = arith.constant 0.000000e+00 : f32 ; constant: 0.0
      %86 = tt.broadcast %85 : tensor<1x64xi1> -> tensor<128x64xi1> ; expand (offs_k[None, :]) < (K - (k * BLOCK_SIZE_K)) to dimention of [dim(offs_k[None, :]) < (K - (k * BLOCK_SIZE_K))]
      %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x64xf32> ; other=0.0
      %87 = arith.truncf %cst_10 : tensor<128x64xf32> to tensor<128x64xf16> ; type convertion
      %88 = tt.load %arg11, %86, %87 : tensor<128x64x!tt.ptr<f16>>
      ; offs_k[:, None]
      %89 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      ; constant: BLOCK_SIZE_K
      %c64_i32_11 = arith.constant 64 : i32
      ; k * BLOCK_SIZE_K
      %90 = arith.muli %arg9, %c64_i32_11 : i32
      ; K - k * BLOCK_SIZE_K
      %91 = arith.subi %arg5, %90 : i32
      ; offs_k[:, None] < K - k * BLOCK_SIZE_K
      %92 = tt.splat %91 : i32 -> tensor<64x1xi32>
      %93 = arith.cmpi slt, %89, %92 : tensor<64x1xi32>
      ; b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
      %cst_12 = arith.constant 0.000000e+00 : f32
      %94 = tt.broadcast %93 : tensor<64x1xi1> -> tensor<64x256xi1>
      %cst_13 = arith.constant dense<0.000000e+00> : tensor<64x256xf32>
      %95 = arith.truncf %cst_13 : tensor<64x256xf32> to tensor<64x256xf16>
      %96 = tt.load %arg12, %94, %95 : tensor<64x256x!tt.ptr<f16>>
      ; c = tl.matmul(a, b, c, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K)
      %cst_14 = arith.constant 0.000000e+00 : f32 ; no use
      %97 = tt.dot %88, %96, %arg10, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
      ; constant: BLOCK_SIZE_K
      %c64_i32_15 = arith.constant 64 : i32
      ; a_ptrs += BLOCK_SIZE_K * stride_ak
      %cst_16 = arith.constant dense<64> : tensor<128x64xi32> ; stride_ak optimized to 1
      %98 = tt.addptr %arg11, %cst_16 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      ; constant: BLOCK_SIZE_K
      %c64_i32_17 = arith.constant 64 : i32
      ; BLOCK_SIZE_K * stride_bk
      %99 = arith.muli %arg7, %c64_i32_17 : i32
      ; b_ptrs += BLOCK_SIZE_K * stride_bk
      %100 = tt.splat %99 : i32 -> tensor<64x256xi32>
      %101 = tt.addptr %arg12, %100 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
      ; Update loop variables: %arg10 = %97(new accumulator), %arg11 = %98(new a_ptrs), %arg12 = %101(new b_ptrs)
      scf.yield %97, %98, %101 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
    }

; Fourth part: Store the result back to c_ptr
    ; %52#0 = %arg10 = accumulator, type convertion to f16
    %53 = arith.truncf %52#0 : tensor<128x256xf32> to tensor<128x256xf16>
    ; pid_m * BLOCK_SIZE_M
    %c128_i32_5 = arith.constant 128 : i32 ; constant: BLOCK_SIZE_M
    %54 = arith.muli %10, %c128_i32_5 : i32
    ; tl.arange(0, BLOCK_SIZE_M)
    %55 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    ; offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    %56 = tt.splat %54 : i32 -> tensor<128xi32>
    %57 = arith.addi %56, %55 : tensor<128xi32>
    ; pid_n * BLOCK_SIZE_N
    %c256_i32_6 = arith.constant 256 : i32 ; constant: BLOCK_SIZE_N
    %58 = arith.muli %12, %c256_i32_6 : i32
    ; offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    %59 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %60 = tt.splat %58 : i32 -> tensor<256xi32>
    %61 = arith.addi %60, %59 : tensor<256xi32>
    ; offs_cm[:, None]
    %62 = tt.expand_dims %57 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    ; (offs_cm[:, None]) * stride_cm
    %63 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %64 = arith.muli %63, %62 : tensor<128x1xi32>
    ; c_ptr + stride_cm * offs_cm[:, None]
    %65 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %66 = tt.addptr %65, %64 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    ; offs_cn[None, :]
    %67 = tt.expand_dims %61 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    ; stride_cn * offs_cn[None, :]
    %c1_i32_7 = arith.constant 1 : i32 ; stride_cn optimized to 1
    %cst_8 = arith.constant dense<1> : tensor<1x256xi32>
    %68 = arith.muli %67, %cst_8 : tensor<1x256xi32>
    ; c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    %69 = tt.broadcast %66 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x256x!tt.ptr<f16>>
    %70 = tt.broadcast %68 : tensor<1x256xi32> -> tensor<128x256xi32>
    %71 = tt.addptr %69, %70 : tensor<128x256x!tt.ptr<f16>>, tensor<128x256xi32>
    ; offs_cm[:, None]
    %72 = tt.expand_dims %57 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    ; offs_cm[:, None] < M
    %73 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %74 = arith.cmpi slt, %72, %73 : tensor<128x1xi32>
    ; offs_cn[None, :]
    %75 = tt.expand_dims %61 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    ; offs_cn[None, :] < N
    %76 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %77 = arith.cmpi slt, %75, %76 : tensor<1x256xi32>
    ; c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    %78 = tt.broadcast %74 : tensor<128x1xi1> -> tensor<128x256xi1>
    %79 = tt.broadcast %77 : tensor<1x256xi1> -> tensor<128x256xi1>
    %80 = arith.andi %78, %79 : tensor<128x256xi1>
    ; tl.store(c_ptrs, c, mask=c_mask)
    tt.store %71, %53, %80 : tensor<128x256x!tt.ptr<f16>>
    tt.return
  }
  tt.func private @cdiv__i32__1cconstexpr_128_(%arg0: i32) -> i32 attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.addi %arg0, %c128_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c128_i32_0 = arith.constant 128 : i32
    %2 = arith.divsi %1, %c128_i32_0 : i32
    tt.return %2 : i32
  }
  tt.func private @cdiv__i32__1cconstexpr_256_(%arg0: i32) -> i32 attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.addi %arg0, %c256_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c256_i32_0 = arith.constant 256 : i32
    %2 = arith.divsi %1, %c256_i32_0 : i32
    tt.return %2 : i32
  }
  tt.func private @"zeros____0cconstexpr_(constexpr_128_, constexpr_256_)__1cconstexpr_fp32_"() -> tensor<128x256xf32> attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    tt.return %cst_0 : tensor<128x256xf32>
  }
  tt.func private @cdiv__i32__1cconstexpr_64_(%arg0: i32) -> i32 attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg0, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %2 = arith.divsi %1, %c64_i32_0 : i32
    tt.return %2 : i32
  }
}
```

## Triton GPU IR

```llvm
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:89", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_m16n16k16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<16x16xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %7 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %12 = tt.load %9 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %13 = triton_gpu.local_alloc %12 : (tensor<16x16xf16, #blocked>) -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory>
    %14 = tt.load %11 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %15 = triton_gpu.local_alloc %14 : (tensor<16x16xf16, #blocked>) -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory>
    %16 = triton_gpu.local_load %13 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %17 = triton_gpu.local_load %15 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %18 = tt.dot %16, %17, %cst, inputPrecision = tf32 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %19 = triton_gpu.convert_layout %18 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    %20 = arith.truncf %19 : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %21 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %22 = tt.addptr %21, %7 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %22, %20 : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
```

### Memory layout

#### Distributed

##### Block

Distributed memory layout of tensor with the following parameters:

1. `sizePerThread`: The size of tensor owned by a thread
2. `threadPerWarp`: Number of `sizePerThread` owned by a warp
3. `warpsPerCTA`: Number of `threadPerWarp` owned by a CTA
4. `CTAsPerCGA`: number of `CTA`s in a `CTA` group
5. `CTASplitNum`: `CTASplitNum[d]` means dimension `d` will be splite to `CTASplitNum[d]` `CTA`s.

Note: `CTAsPerCGA` and `CTASplitNum` are newly imported feature, defaultly not used and will remian `CTAsPerCGA = 1` for the program and `CTASplitNum` as $1^{rand}$ for each tensor

For a 2D tensor `A` with the offset address of `A[i,j]` can be calculated as `offset = (i * num_col + j) * byte_per_element` the thread that owns `A[i,j]` can be calculated as follow:

```c
struct {
    int CTAnum;
    int warpNum;
    int threadNum;
} threadLoc;

threadLoc calcLaneID(int i, int j, int sizePerThread[2], int threadPerWarp[2], int warpsPerCTA[2]) {
    threadLoc loc;
    int sizePerWarp[2], sizePerCTA[2];
    
    sizePerWarp[0] = sizePerThread[0] * threadPerWarp[0];
    sizePerWarp[1] = sizePerThread[1] * threadPerWarp[1];
    sizePerCTA[0] = sizePerWarp[0] * warpsPerCTA[0];
    sizePerCTA[1] = sizePerWarp[1] * warpsPerCTA[1];
    
    int CTA_row = i / sizePerCTA[0];
    int CTA_col = j / sizePerCTA[1];
    loc.CTAnum = CTA_row * warpsPerCTA[1] + CTA_col;

    int warp_row = (i % sizePerCTA[0]) / sizePerWarp[0];
    int warp_col = (j % sizePerCTA[1]) / sizePerWarp[1];
    loc.warpNum = warp_row * warpsPerCTA[1] + warp_col;

    int lane_row = (i % sizePerWarp[0]) / sizePerThread[0];
    int lane_col = (j % sizePerWarp[1]) / sizePerThread[1];
    loc.threadNum = lane_row * threadPerWarp[1] + lane_col;

    return loc;
}
```

For nD tensor, `A` with the offset address of `A[i_1,...,i_n]` can be calculated as `offset = (i_1 * stride_1 + ... + i_{n-1} * stride_{n-1} + ... + i_n) * byte_per_element` the thread that owns `A[i_1,...,i_n]` can be calculated as follow:

```c
struct {
    int CTAnum;
    int warpNum;
    int threadNum;
} threadLoc;

threadLoc calcLaneID(int i, int j, int sizePerThread[n], int threadPerWarp[n], int warpsPerCTA[n]) {
    threadLoc loc;
    int sizePerWarp[n], sizePerCTA[n], CTA[n], warp[n], lane[n];
    
    for (int i = 0; i < n; i++) {
        sizePerWarp[i] = sizePerThread[i] * threadPerWarp[i];
    }

    for (int i = 0; i < n; i++) {
        sizePerCTA[i] = sizePerWarp[i] * warpsPerCTA[i];
    }

    // CTA offset
    for (int i = 0; i < n; i++) {
        CTA[i] = i / sizePerCTA[i];
    }
    int stride = 1;
    for (int i = n - 1; i >= 0; i--) {
        loc.CTAnum += CTA[i] * stride;
        stride *= sizePerCTA[i];
    }

    // warp offset
    for (int i = 0; i < n; i++) {
        warp[i] = (i % sizePerCTA[i]) / sizePerWarp[i];
    }
    stride = 1;
    for (int i = n - 1; i >= 0; i--) {
        loc.warpNum += warp[i] * stride;
        stride *= sizePerWarp[i];
    }

    // thread offset
    for (int i = 0; i < n; i++) {
        lane[i] = (i % sizePerWarp[i]) / sizePerThread[i];
    }
    stride = 1;
    for (int i = n - 1; i >= 0; i--) {
        loc.threadNum += lane[i] * stride;
        stride *= sizePerThread[i];
    }

    return loc;
}
```

##### nvidia_mma

Memory layout specifically designed for Nvidia's [mma](#mma-1) operation. It has the following parameters:

1. `versionMajor` & `versionMinor`: The version of the mma operation
2. `warpsPerCTA`: The number of warps per CTA
3. `CTAlayout`: Not used for now
4. `instrShape`: `m` and `n` of [`mma`](#nvidia_mma) instruction. Example of `mma.sync.aligned.m16n8k16` has the shape of `[16, 8]`

The detailed position mapping need further investigation.

##### Slice

##### DotOperand

Memory layout of matrix `A` and `B` for matrix produce of `D = AB + C`, having the following  parameters:

1. `opIdx`: The index of the operand in the dot operation. (A = 0, B = 1)
2. `parent`: A [Distributed layout](#distributed) attribute. Indicating detailed layout of the memory
3. `kWidth`: The number of consecutive elements stored by one thread along k dimension

Note that for Nvidia's [mma](#mma-1) operation for `D = AB + C` all of `A`, `B`, `C` and `D` should would be stored in thread's register across one warp. Thus all of them need [distributed](#distributed) memory layout.

This is the reason of adding [mma](#nvidia_mma) layout.

#### Shared 

Meaning memory that can be accessed by all threads in CTA (or block). One of the example would be shared memory in CUDA.

##### Swizzling

Pre-definitions:

1. A 2D tensor `A` with the offset address of `A[i,j]` can be calculated as `offset = (i * num_col + j) * byte_per_element`
2. `triton.shared` attribute with parameters `vec`, `perPhase` and `maxPhase`

Then we can calculate the offset address of elment `A[i,j]` as follow:

```c
int offset (int i, int j, int num_col, int byte_per_element, int vec, int perPhase, int maxPhase) {
    int phase_grp = i / perPhase;
    int phase = phase_grp % maxPhase;
    int line_grp = j / vec;
    int phased_line_grp = line_grp ^ perPhase; //xor
    int phased_j = phased_line_grp * vec

    return (i * num_col + phased_j) * byte_per_element;
}
```

The reason of swizzling is to avoid bank conflict, see [this post](https://github.com/triton-lang/triton/discussions/2026).

For Nvidia device, the `vec` should be 8, since [`ldmatrix`](#ldmatrix) instruction can switch rows of 8 element of f16 by adjusting `%lane.s0` register.

## PTX

```ptx
.version 8.4
.target sm_89
.address_size 64

	// .globl	matmul_kernel_m16n16k16
.extern .shared .align 16 .b8 global_smem[];

.visible .entry matmul_kernel_m16n16k16(
	.param .u64 matmul_kernel_m16n16k16_param_0,
	.param .u64 matmul_kernel_m16n16k16_param_1,
	.param .u64 matmul_kernel_m16n16k16_param_2
)
.maxntid 128, 1, 1
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<62>;
	.reg .f32 	%f<11>;
	.reg .b64 	%rd<8>;
$L__func_begin0:

	ld.param.u64 	%rd4, [matmul_kernel_m16n16k16_param_0];
	ld.param.u64 	%rd5, [matmul_kernel_m16n16k16_param_1];
$L__tmp0:
	mov.u32 	%r20, %tid.x;
	ld.param.u64 	%rd6, [matmul_kernel_m16n16k16_param_2];
	shl.b32 	%r21, %r20, 1;
	and.b32  	%r22, %r21, 240;
	and.b32  	%r23, %r21, 14;
	and.b32  	%r24, %r21, 254;
	mul.wide.u32 	%rd7, %r24, 2;
	add.s64 	%rd1, %rd4, %rd7;
	add.s64 	%rd2, %rd5, %rd7;
	mov.pred 	%p1, -1;
	// begin inline asm
	mov.u32 %r1, 0x0;
	@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
	// end inline asm
	shr.u32 	%r25, %r20, 2;
	and.b32  	%r26, %r25, 8;
	xor.b32  	%r27, %r23, %r26;
	or.b32  	%r28, %r27, %r22;
	shl.b32 	%r29, %r28, 1;
	mov.u32 	%r30, global_smem;
	add.s32 	%r31, %r30, %r29;
	st.shared.u32 	[%r31], %r1;
	// begin inline asm
	mov.u32 %r2, 0x0;
	@%p1 ld.global.b32 { %r2 }, [ %rd2 + 0 ];
	// end inline asm
	add.s32 	%r32, %r30, 1024;
	add.s32 	%r33, %r32, %r29;
	st.shared.u32 	[%r33], %r2;
	bar.sync 	0;
	and.b32  	%r34, %r20, 7;
	bfe.u32 	%r35, %r20, 3, 2;
	bfe.u32 	%r36, %r20, 4, 1;
	bfe.u32 	%r37, %r20, 2, 1;
	xor.b32  	%r38, %r36, %r37;
	shl.b32 	%r39, %r38, 4;
	shl.b32 	%r40, %r20, 5;
	and.b32  	%r41, %r40, 480;
	or.b32  	%r42, %r39, %r41;
	add.s32 	%r7, %r30, %r42;
	// begin inline asm
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %r13, %r14, %r15, %r16 }, [ %r7 + 0 ];
	// end inline asm
	bfe.u32 	%r43, %r20, 5, 1;
	xor.b32  	%r44, %r43, %r37;
	shl.b32 	%r45, %r44, 4;
	or.b32  	%r46, %r45, %r41;
	add.s32 	%r12, %r32, %r46;
	// begin inline asm
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 { %r17, %r18, %r10, %r11 }, [ %r12 + 0 ];
	// end inline asm
	mov.f32 	%f1, 0f00000000;1
	mov.f32 	%f2, %f1;
	mov.f32 	%f3, %f1;
	mov.f32 	%f4, %f1;
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %f1, %f2, %f3, %f4 }, { %r13, %r14, %r15, %r16 }, { %r17, %r18 }, { %f1, %f2, %f3, %f4 };
	// end inline asm
	bar.sync 	0;
	bfe.u32 	%r47, %r20, 2, 3;
	and.b32  	%r48, %r21, 6;
	shl.b32 	%r49, %r43, 3;
	or.b32  	%r50, %r49, %r48;
	mad.lo.s32 	%r51, %r47, 18, %r50;
	shl.b32 	%r52, %r51, 2;
	add.s32 	%r53, %r30, %r52;
	st.shared.v2.f32 	[%r53], {%f1, %f2};
	st.shared.v2.f32 	[%r53+576], {%f3, %f4};
	bar.sync 	0;
	shr.u32 	%r54, %r20, 3;
	and.b32  	%r55, %r54, 12;
	or.b32  	%r56, %r55, %r35;
	shl.b32 	%r57, %r34, 1;
	mad.lo.s32 	%r58, %r56, 18, %r57;
	shl.b32 	%r59, %r58, 2;
	add.s32 	%r60, %r30, %r59;
	ld.shared.v2.f32 	{%f9, %f10}, [%r60];
	cvt.rn.f16.f32 	%rs1, %f10;
	cvt.rn.f16.f32 	%rs2, %f9;
	mov.b32 	%r61, {%rs2, %rs1};
	add.s64 	%rd3, %rd6, %rd7;
	// begin inline asm
	@%p1 st.global.b32 [ %rd3 + 0 ], { %r61 };
	// end inline asm
	ret;
$L__tmp1:
$L__func_end0:
}
```

### Matrix multiplication with nvidia_mma

This is a warp level operation, which means each warp would finish one `mma` instruction, to perform the `mma` operation, we need:

1. Load both matrixs of specific shape to thread's register (Based ont he shape of Matrix, each thread would hold a determined part of elements of the matrix)
2. perform `mma` and the result will be saved in registers spreaded in each threads
3. Store the result back

For better illustration, we use `mma.sync.aligned.m16n8k16` as example in the following sections.

### ldmatrix

Official document is [here](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix).

On warp level, it load 1, 2, or 4 matrix of shape `[8, 8]` with size of element are 2 byte.

#### Example of `ldmatrix.sync.aligned.m8n8.x4.shared.b16 { d0, d1, d2, d3 }, [ s0 + 0 ]`

Donate `%lane.s0` as `%lane`'th thread's `s0` register. same notation applies to `d0`, `d1`, `d2`, `d3`.

The element of memory address `%lane.s0 + 2j` to `%land.s0 + 2j + 1` (2 byte), where `j` is less than 8 and greater or equal to 0 will be loaded to:

```c
struct {
    int laneID;
    int dstIdx;
    int high; // 0 in the low 16 bit, 1 in the high 16 bit
} ThreadLoc;

ThreadLoc calcTheadLoc(int srcLaneId, int s0, int j) 
{
    ThreadLoc loc;
    loc.dstIdx = srcLaneId / 8;
    int row_in_mat = srcLaneId % 8;
    int col_in_mat = j;
    // if col_in_mat is odd, then the element will be stored in high 16bit of target register, otherwise low 16 bitxx
    loc.high = col_in_mat % 2; 
    // one dst register will store 2 elements (since each elements tooks 2 byte)
    loc.laneID = row_in_mat * 4 + (col_in_mat / 2);

    return loc;
}
```

Illustration:

1. Basically, the `i`'th `[8, 8]` matrix will be stored in destination register `di` of each thread
2. The part that each register holds are same as the figure in official document:
   <img src=https://docs.nvidia.com/cuda/parallel-thread-execution/_images/mma-ldmatrix-fragments.png>
3. The start address of `i`'th matrix's `n`'th row of 8 elements (2 byte) will be stored at the address of `%(i * 8 + j).s0`. Thus the matrix do not be continues in memory. This provide the possibility of [Swizzling](#swizzling)

### mma

A warp would use this instruction to execute matrix multiplication.

We use example of `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { d1, d2, d3, d4 }, { a13, a14, a15, a16 }, { b17, b18 }, { d1, d2, d3, d4 };` as example to illustrate the usage of `mma` instruction.

#### matrix A

By definition of the instruction, matrix A has the shape of `[16, 16]` with each element tooks 2 byte

Element of `A[i,j]` will be stored in:

```c
struct {
    int laneID;
    int aIdx;
    int high; // 0 in the low 16 bit, 1 in the high 16 bit
} ALoc;

ALoc calcTheadLoc(int i, int j) 
{
    ALoc loc;
    loc.aIdx = i / 8 + j / 8;
    int row_in_sub_mat = i % 8;
    int col_in_sub_mat = j % 8;
    // if col_in_mat is odd, then the element will be stored in high 16bit of target register, otherwise low 16 bitxx
    loc.high = col_in_sub_mat % 2;
    // one dst register will store 2 elements (since each elements tooks 2 byte)
    loc.laneID = row_in_sub_mat * 4 + (col_in_sub_mat) / 2; 

    return loc;
}
```

#### matrix B

By definition of the instruction, matrix A has the shape of `[16, 8]` with each element tooks 2 byte

Element of `B[i,j]` will be stored in:

```c
struct {
    int laneID;
    int dIdx;
    int high; // 0 in the low 16 bit, 1 in the high 16 bit
} BLoc;

BLoc calcTheadLoc(int i, int j) 
{
    BLoc loc;
    int sub_mat = i / 8;
    int row_in_sub_mat = i % 8;
    int col_in_sub_mat = j;
    loc.high = i % 2;
    loc.aIdx = sub_mat + i % 2;
    // two dst registers of same thread will store 2 elements (since each elements tooks 2 byte)
    loc.landID = j * 4 + i / 2;
    return loc;
}
```

#### matrix D

By definition of the instruction, matrix A has the shape of `[16, 8]` with each element tooks 4 byte

Element of `D[i,j]` will be stored in:

```c
struct {
    int laneID;
    int dIdx;
} DLoc;

DLoc calcTheadLoc(int i, int j) 
{
    DLoc loc;
    int sub_mat = i / 8;
    int row_in_sub_mat = i % 8;
    int col_in_sub_mat = j;
    loc.aIdx = sub_mat + j % 2;
    // two dst registers of same thread will store 2 elements (since each elements tooks 4 byte)
    loc.laneID = row_in_sub_mat * 4 + (col_in_mat / 2); 

    return loc;
}
```

# MLIR

## Conversion between Dialects

Referencing to [toy example](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) from the [official document](#https://mlir.llvm.org/docs/) of MLIR.

The example we used is conversion of `tritonIR` to `tritonGPUIR`, which is pass [ConvertTritonToTritonGPU](triton-project/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp#L746) pass.

It is useful to understand this since almost all of the conversion pass in Triton is similar to this example.

### Components

#### TypeConverter

For our example, defined as follow:

```cpp
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    // types with encoding are already in the right format
    // TODO: check for layout encodings more specifically
    if (tensorType.getEncoding())
      return tensorType;
    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::gpu::BlockedEncodingAttr encoding =
        getDefaultBlockedEncoding(this->context, shape, this->numWarps,
                                  this->threadsPerWarp, this->numCTAs);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  // Add encoding for tensor pointer
  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    // Check whether tensor pointer `tt.ptr<tensor<>>`
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (pointeeTensorType == nullptr)
      return ptrType;

    // Add layout into the tensor
    auto convertedTensorType = convertType(pointeeTensorType);
    return triton::PointerType::get(convertedTensorType,
                                    ptrType.getAddressSpace());
  });

  //
  // Materializations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
    llvm_unreachable("Argument rematerialization should not happen in Triton "
                     "-> TritonGPU conversion");
    return std::nullopt;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonGPU Conversion");
    return std::nullopt;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    auto cast =
        builder.create<triton::gpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return std::optional<Value>(cast.getResult());
  });
}
```

1. `addConversion([](Type type) { return type; });`: Defaultly return the same type
2. `addConversion([this](RankedTensorType tensorType) -> RankedTensorType{...}`: Logic to convert `RankedTensorType`, in here it returns the same type but added extra attribute ([memory layout](#block))
3. `addArgumentMaterialization(...)`: lambda function to handle the case when the argument is not the same type as the original one. Here it indicate that this should not be happennig when converting `tritonIR` to `tritonGPUIR`
4. `addTargetMaterialization(...)`: lambda function to handle the case when the target is not the same type as the original one

Other functions need to be further investigaged.

#### Target

For our example, defined as follow:

```cpp
TritonGPUConversionTarget::TritonGPUConversionTarget(
    MLIRContext &context, TritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of TritonGPUDialect
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    if (hasLegalRegions && typeConverter.isLegal(op)) {
      return true;
    }
    return false;
  });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        cast<RankedTensorType>(dotOp.getA().getType()).getEncoding();
    Attribute bEncoding =
        cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
    if (aEncoding && isa<triton::gpu::DotOperandEncodingAttr>(aEncoding) &&
        bEncoding && isa<triton::gpu::DotOperandEncodingAttr>(bEncoding))
      return true;
    return false;
  });
}
```

1. `addLegalDialect<triton::gpu::TritonGPUDialect>();`: All operations in `triton::gpu::TritonGPUDialect` are legal
2. `addIllegalOp<...>`: All operations in `<...>` are illegal. Actually it will remove the patterns that applies on legal operations at beginging.
3. `addDynamicallyLegalDialect<arith::ArithDialect, ...>([&](Operation *op) {...})` for operations from listed dialects in `<arith::ArithDialect, ...>`, used the passed in lambda function to determine if the operation is legal.
4. `addDynamicallyLegalOp<triton::DotOp>(...)` for `triton::DotOp` operation, used the passed in lambda function to determine if the operation is legal.

#### Pattern

Re-write patterns for conversion. Usually sub-class of [`OpConversionPattern`](llvm-project/mlir/include/mlir/Transforms/DialectConversion.h#L515)

### applyPartialConversion

#### Basic process

By the following to jump calls:

```cpp
LogicalResult mlir::applyPartialConversion(
    ArrayRef<Operation *> ops, const ConversionTarget &target,
    const FrozenRewritePatternSet &patterns, ConversionConfig config) {
  OperationConverter opConverter(target, patterns, config,
                                 OpConversionMode::Partial);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyPartialConversion(Operation *op, const ConversionTarget &target,
                             const FrozenRewritePatternSet &patterns,
                             ConversionConfig config) {
  return applyPartialConversion(llvm::ArrayRef(op), target, patterns, config);
}
```

MLIR will fall into the main logic of applyPartialConversion:

```cpp
LogicalResult OperationConverter::convertOperations(ArrayRef<Operation *> ops) {
  if (ops.empty())
    return success();
  const ConversionTarget &target = opLegalizer.getTarget();

  // Compute the set of operations and blocks to convert.
  SmallVector<Operation *> toConvert;
  for (auto *op : ops) {
    op->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&](Operation *op) {
          toConvert.push_back(op);
          // Don't check this operation's children for conversion if the
          // operation is recursively legal.
          auto legalityInfo = target.isLegal(op);
          if (legalityInfo && legalityInfo->isRecursivelyLegal)
            return WalkResult::skip();
          return WalkResult::advance();
        });
  }

  // Convert each operation and discard rewrites on failure.
  ConversionPatternRewriter rewriter(ops.front()->getContext(), config);
  ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();

  for (auto *op : toConvert)
    if (failed(convert(rewriter, op)))
      return rewriterImpl.undoRewrites(), failure();

  // Now that all of the operations have been converted, finalize the conversion
  // process to ensure any lingering conversion artifacts are cleaned up and
  // legalized.
  if (failed(finalize(rewriter)))
    return rewriterImpl.undoRewrites(), failure();

  // After a successful conversion, apply rewrites if this is not an analysis
  // conversion.
  if (mode == OpConversionMode::Analysis) {
    rewriterImpl.undoRewrites();
  } else {
    rewriterImpl.applyRewrites();
  }
  return success();
}
```

1. It can be seen that a failure of one pattern will cause the failure of total conversion process and result in undo of all applied conversions:
   ```cpp
    // Convert each operation and discard rewrites on failure.
    ConversionPatternRewriter rewriter(ops.front()->getContext(), config);
    ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();

    for (auto *op : toConvert)
        if (failed(convert(rewriter, op)))
        return rewriterImpl.undoRewrites(), failure();
   ```
2. After all the patterns has been applied, there is still check of legality of the whole module
3. By calling `convert` it will call `opLegalizer.legalize(op, rewriter)`, and for partial convertions, if legalization failed, it will no return error unless the operation is [illegal for the target](#target):
   ```cpp
    LogicalResult OperationConverter::convert(ConversionPatternRewriter &rewriter,
                                            Operation *op) {
        // Legalize the given operation.
        if (failed(opLegalizer.legalize(op, rewriter))) {
            // Handle the case of a failed conversion for each of the different modes.
            // Full conversions expect all operations to be converted.
            if (mode == OpConversionMode::Full)
            return op->emitError()
                    << "failed to legalize operation '" << op->getName() << "'";
            // Partial conversions allow conversions to fail iff the operation was not
            // explicitly marked as illegal. If the user provided a `unlegalizedOps`
            // set, non-legalizable ops are added to that set.
            if (mode == OpConversionMode::Partial) {
            if (opLegalizer.isIllegal(op))
                return op->emitError()
                    << "failed to legalize operation '" << op->getName()
                    << "' that was explicitly marked illegal";
            if (config.unlegalizedOps)
                config.unlegalizedOps->insert(op);
            }
        }
        ...
    }
   ```
4. If operation is [legal for the target](#target), than return. Otherwise will call target patterns to lower the operation:
   ```cpp
    LogicalResult
    OperationLegalizer::legalize(Operation *op,
                                ConversionPatternRewriter &rewriter) {
        ...
        // Check if this operation is legal on the target.
        if (auto legalityInfo = target.isLegal(op)) {
            LLVM_DEBUG({
            logSuccess(
                logger, "operation marked legal by the target{0}",
                legalityInfo->isRecursivelyLegal
                    ? "; NOTE: operation is recursively legal; skipping internals"
                    : "");
            logger.startLine() << logLineComment;
            });

            // If this operation is recursively legal, mark its children as ignored so
            // that we don't consider them for legalization.
            if (legalityInfo->isRecursivelyLegal) {
            op->walk([&](Operation *nested) {
                if (op != nested)
                rewriter.getImpl().ignoredOps.insert(nested);
            });
            }

            return success();
        }

        ...
        // Otherwise, we need to apply a legalization pattern to this operation.
        if (succeeded(legalizeWithPattern(op, rewriter))) {
            LLVM_DEBUG({
            logSuccess(logger, "");
            logger.startLine() << logLineComment;
            });
            return success();
        }
        ...
    }
   ```

#### Properties

1. From the [above section](#basic-process), if a pattern failed, it will continue to apply the rest of other patterns
2. If [`convert`](llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp#L2551) function failed, it will stop and undo all the applied patterns
3. During the conversion, if a re-writed operations type does not fit it's user, it will create a new operation, and will save the new operand in `OpAdaptor` of user operation. After all conversion has been done, it will clean the un-needed operations by  calling `rewrite->cleanup`. see Appendix for example

## CallGraph

Defined [here](triton-project/include/triton/Analysis/Utility.h#L227) and build up 2 structures:

1. `graph`: is a map that key is `FuncOp` and value is a pair of `callOp` within the key and target `FuncOp` of the call
2. `roots`: those `FuncOp` that are not called by any other `FuncOp`

# Appendix

## Definiton of `BackendInstaller` and `Backend` class

```py
@dataclass
class Backend:
    name: str
    package_data: dict
    src_dir: str
    backend_dir: str
    install_dir: str
    is_external: bool


class BackendInstaller:

    @staticmethod
    def prepare(backend_name: str, backend_src_dir: str = None, is_external: bool = False):
        # Initialize submodule if there is one for in-tree backends.
        if not is_external:
            root_dir = os.path.join(os.pardir, "third_party")
            assert backend_name in os.listdir(
                root_dir), f"{backend_name} is requested for install but not present in {root_dir}"

            try:
                subprocess.run(["git", "submodule", "update", "--init", f"{backend_name}"], check=True,
                               stdout=subprocess.DEVNULL, cwd=root_dir)
            except subprocess.CalledProcessError:
                pass
            except FileNotFoundError:
                pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.abspath(os.path.join(backend_src_dir, "backend"))
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(os.path.join(backend_path, file)), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(os.path.dirname(__file__), "triton", "backends", backend_name)
        package_data = [f"{os.path.relpath(p, backend_path)}/*" for p, _, _, in os.walk(backend_path)]
        return Backend(name=backend_name, package_data=package_data, src_dir=backend_src_dir, backend_dir=backend_path,
                       install_dir=install_dir, is_external=is_external)

    # Copy all in-tree backends under triton/third_party.
    @staticmethod
    def copy(active):
        return [BackendInstaller.prepare(backend) for backend in active]

    # Copy all external plugins provided by the `TRITON_PLUGIN_DIRS` env var.
    # TRITON_PLUGIN_DIRS is a semicolon-separated list of paths to the plugins.
    # Expect to find the name of the backend under dir/backend/name.conf
    @staticmethod
    def copy_externals():
        backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
        if backend_dirs is None:
            return []
        backend_dirs = backend_dirs.strip().split(";")
        backend_names = [Path(os.path.join(dir, "backend", "name.conf")).read_text().strip() for dir in backend_dirs]
        return [
            BackendInstaller.prepare(backend_name, backend_src_dir=backend_src_dir, is_external=True)
            for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
        ]
```

## Example for partial conversion not unaligned type

The incoming IR is:

```llvm
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:89", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_m16n16k16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
    %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x16xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %10 = tt.addptr %9, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %11 = tt.load %8 : tensor<16x16x!tt.ptr<f16>>
    %12 = tt.load %10 : tensor<16x16x!tt.ptr<f16>>
    %13 = tt.dot %11, %12, %cst, inputPrecision = tf32 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %14 = arith.truncf %13 : tensor<16x16xf32> to tensor<16x16xf16>
    %15 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %16 = tt.addptr %15, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    tt.store %16, %14 : tensor<16x16x!tt.ptr<f16>>
    tt.return
  }
}
```

After applying `ArithConstantPattern`:

```llvm
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:89", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_m16n16k16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ; new operation for cst_0 since tt.dot uses cst_0 and the type does not aligne with conveted one
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_1 = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst_1 : tensor<16x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
    %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x16xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %10 = tt.addptr %9, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %11 = tt.load %8 : tensor<16x16x!tt.ptr<f16>>
    %12 = tt.load %10 : tensor<16x16x!tt.ptr<f16>>
    ; This should be using he new operation cst, but due to the type inconsistency, the new operation is stored in ``
    %13 = tt.dot %11, %12, %cst_0, inputPrecision = tf32 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %14 = arith.truncf %13 : tensor<16x16xf32> to tensor<16x16xf16>
    %15 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %16 = tt.addptr %15, %6 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    tt.store %16, %14 : tensor<16x16x!tt.ptr<f16>>
    tt.return
  }
}
```

After `rewrite->cleanup`:

```llvm
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:89", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_m16n16k16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %1 = triton_gpu.convert_layout %0 : tensor<16xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %3 = triton_gpu.convert_layout %2 : tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %4 = arith.muli %3, %cst_0 : tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %5 = triton_gpu.convert_layout %0 : tensor<16xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %7 = triton_gpu.convert_layout %6 : tensor<1x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %8 = tt.broadcast %4 : tensor<16x1xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %9 = triton_gpu.convert_layout %8 : tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %10 = tt.broadcast %7 : tensor<1x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %11 = arith.addi %9, %10 : tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %12 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %13 = tt.addptr %12, %11 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %15 = tt.addptr %14, %11 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %16 = tt.load %13 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %17 = tt.load %15 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %18 = triton_gpu.convert_layout %16 : tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %19 = triton_gpu.convert_layout %17 : tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %20 = triton_gpu.convert_layout %cst : tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %21 = tt.dot %18, %19, %20, inputPrecision = tf32 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = arith.truncf %21 : tensor<16x16xf32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %23 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.addptr %23, %11 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<16x16xi32, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %24, %22 : tensor<16x16x!tt.ptr<f16>, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}
```