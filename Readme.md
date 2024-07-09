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
   2. `driver.py` import `GPUDriver` from `triton.backends.driver`, create exactly one subclass of `DriverBase` (since `GPUDriver` is the subclass of `DriverBase`)
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

## Compiling process



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

