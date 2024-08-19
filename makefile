# $@  表示目标文件
# $^  表示所有的依赖文件
# $<  表示第一个依赖文件
# $?  表示比目标还要新的依赖文件列表
BRANCH = $(shell git symbolic-ref --short HEAD)
LLVM_HASH = $(shell cat ./triton-project/cmake/llvm-hash.txt)
PW = $(shell cat ~/文档/PW)
TEXT =
# tools ==================================================================================
search_text:
	grep -rn --exclude="*.o" --exclude="*.so" --exclude="*.a" "$(TEXT)" .
# LLVM ===================================================================================
checkout_hash:
	cd llvm-project && git checkout $(LLVM_HASH)
config_llvm:
	cd llvm-project && cmake -G Ninja -S llvm -B build -DCMAKE_INSTALL_PREFIX=../bin -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_PARALLEL_COMPILE_JOBS=32 -DLLVM_PARALLEL_LINK_JOBS=4
build_llvm:
	cd llvm-project/build && ninja
clear_loc:
	@sed -i '/^#loc[0-9]* =/d' *.ll
	sed -i 's/ loc(\([^)]*\))//g' *.ll

PHONY += checkout_hash config_llvm build_llvm
# Triton Build ===========================================================================
build_triton:
	cd triton/build && LLVM_INCLUDE_DIRS=../llvm-project/build/include LLVM_LIBRARY_DIR=../llvm-project/build/lib LLVM_SYSPATH=../llvm-project/build DEBUG=1 MAX_JOBS=6 pip install -e python

PHONY += config_triton build_triton
# python virtualenv ======================================================================
.venv:
	python -m venv .venv --prompt triton
active_venv: .venv
	source .venv/bin/activate
deactive_venv:
	- deactivate
clear_cache:
	find . -type d -name "__pycache__" -exec rm -r {} +
	- rm -rf /tmp/torchinductor_shore
	- rm -rf ./.cache/*
	- rm -rf ./.tmp/*
allow_attach:
	echo $(PW) | sudo -S sh -c 'echo 0 | tee /proc/sys/kernel/yama/ptrace_scope'

PHONY += active_venv deactive_venv clear_cache
# git ====================================================================================
triton_pull:
	cd triton-project && git pull
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
