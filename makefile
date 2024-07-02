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
	cd triton && cmake -B build -G Ninja -DTRITON_BUILD_PYTHON_MODULE=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../bin -DCMAKE_INSTALL_PREFIX="/usr/lib/python3.12/site-packages/pybind11/include" -DLLVM_LIBRARY_DIR="../llvm-project/build/lib" -DCUPTI_INCLUDE_DIR="/opt/cuda/extras/CUPTI/include;/opt/cuda/targets/x86_64-linux/include" -DROCTRACER_INCLUDE_DIR="/opt/rocm/include/" -DJSON_INCLUDE_DIR="/usr/include/" -DCMAKE_EXPORT_COMPILE_COMMANDS=1
build_triton:
	cd triton/build && ninja

PHONY += config_triton build_triton
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