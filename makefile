# $@  表示目标文件
# $^  表示所有的依赖文件
# $<  表示第一个依赖文件
# $?  表示比目标还要新的依赖文件列表
BRANCH = $(shell git symbolic-ref --short HEAD)
LLVM_HASH = $(shell cat ./triton/cmake/llvm-hash.txt)
# LLVM Build =============================================================================
checkout_hash:
	cd llvm-project && git checkout $(LLVM_HASH)
config_llvm:
	cd llvm-project && cmake -G Ninja -S llvm -B build -DCMAKE_INSTALL_PREFIX=../bin -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_PARALLEL_COMPILE_JOBS=32 -DLLVM_PARALLEL_LINK_JOBS=4
build_llvm:
	cd llvm-project/build && ninja

PHONY += checkout_hash config_llvm build_llvm
# Triton Build ===========================================================================
config_triton:
	cd triton && cmake -B build -G Ninja -DTRITON_BUILD_PYTHON_MODULE=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../bin -DPYBIND11_INCLUDE_DIR="/usr/lib/python3.12/site-packages/pybind11/include" -DLLVM_LIBRARY_DIR="../llvm-project/build/lib" -DTRITON_BUILD_PROTON=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1
build_triton:
	cd triton/build && LLVM_INCLUDE_DIRS=../llvm-project/build/include LLVM_LIBRARY_DIR=../llvm-project/build/lib LLVM_SYSPATH=../llvm-project/build CMAKE_BUILD_TYPE=Debug MAX_JOBS=8 pip install -e python

PHONY += config_triton build_triton
# python virtualenv ======================================================================
.venv:
	python -m venv .venv --prompt triton
active_venv: .venv
	source .venv/bin/activate
deactive_venv:
	- deactivate

PHONY += active_venv deactive_venv
# git ====================================================================================
sub_pull:
	git submodule foreach --recursive 'git pull'
commit:
	git add -A
	@echo "Please type in commit comment: "; \
	read comment; \
	git commit -m"$$comment"
sync: commit
	git push -u triton_learn $(BRANCH)
reset_hard:
	git fetch && git reset --hard origin/$(BRANCH)

PHONY += commit sync sub_pull