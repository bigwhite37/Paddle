#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

XTRANS_URL="https://klx-sdk-release-public.su.bcebos.com/mars_release/XTRANSCUDA/dev/latest/xtrans_cuda_12.8_ubuntu2004_x86_64_mars.tar.gz"
SIMULATOR_URL="https://klx-sdk-release-public.su.bcebos.com/xse/mars/release/latest/output.tar.gz"
NCCL_URL="https://irepo.baidu-int.com/rest/prod/v3/baidu/xpu/bkcl-ci/nodes/96914472/files"

XCUDA_PATH="${PARENT_DIR}/xtrans_cuda_12.8_ubuntu2004_x86_64_mars"
SIMULATOR_DIR="${PARENT_DIR}/xse_simulator"
NCCL_DIR="${PARENT_DIR}/nccl"
NCCL_ROOT="${NCCL_DIR}/output/xccl_Linux_x86_64_nccl_cuda12"

# --- Color definitions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
step()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; \
           echo -e "${BOLD}${CYAN}  [$1/8] $2${NC}"; \
           echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   M100 Paddle Build & Verification      ║"
echo "  ║   Branch: m100 | Target: XCN (KL004)    ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================
step 1 "Download xtrans SDK"
# ============================================================
if [ -d "$XCUDA_PATH" ]; then
    ok "Already exists: ${DIM}$XCUDA_PATH${NC}"
else
    info "Downloading xtrans SDK (~4.8GB, direct connection)..."
    info "URL: ${DIM}$XTRANS_URL${NC}"
    wget --no-proxy -q --show-progress -O /tmp/xtrans_sdk.tar.gz "$XTRANS_URL"
    info "Extracting to ${DIM}$PARENT_DIR${NC} ..."
    tar -xzf /tmp/xtrans_sdk.tar.gz -C "$PARENT_DIR"
    rm -f /tmp/xtrans_sdk.tar.gz
    ok "xtrans SDK installed"
fi

CUDNN_LIB_DIR="${XCUDA_PATH}/targets/x86_64-linux/lib"
if [ -f "${CUDNN_LIB_DIR}/libcudnn.so.8.9" ] && [ ! -e "${CUDNN_LIB_DIR}/libcudnn.so.9" ]; then
    ln -sf libcudnn.so.8.9 "${CUDNN_LIB_DIR}/libcudnn.so.9"
    ok "Created libcudnn.so.9 symlink"
fi

# ============================================================
step 2 "Download M100 Simulator"
# ============================================================
if [ -d "$SIMULATOR_DIR" ]; then
    ok "Already exists: ${DIM}$SIMULATOR_DIR${NC}"
else
    info "Downloading M100 simulator..."
    info "URL: ${DIM}$SIMULATOR_URL${NC}"
    mkdir -p "$SIMULATOR_DIR"
    wget --no-proxy -q --show-progress -O /tmp/xse_simulator.tar.gz "$SIMULATOR_URL"
    info "Extracting to ${DIM}$SIMULATOR_DIR${NC} ..."
    tar -xzf /tmp/xse_simulator.tar.gz -C "$SIMULATOR_DIR"
    rm -f /tmp/xse_simulator.tar.gz
    ok "M100 simulator installed"
fi

# ============================================================
step 3 "Download XNCCL"
# ============================================================
if [ -d "$NCCL_ROOT" ]; then
    ok "Already exists: ${DIM}$NCCL_ROOT${NC}"
else
    info "Downloading XNCCL..."
    info "URL: ${DIM}$NCCL_URL${NC}"
    mkdir -p "$NCCL_DIR"
    wget --no-check-certificate --header "IREPO-TOKEN:b7bcb630-febc-4c17-8fad-ccbce41b6931" -q --show-progress -O /tmp/xnccl_output.tar.gz "$NCCL_URL"
    info "Extracting to ${DIM}$NCCL_DIR${NC} ..."
    tar -xzf /tmp/xnccl_output.tar.gz -C "$NCCL_DIR"
    rm -f /tmp/xnccl_output.tar.gz
    ok "XNCCL installed"
fi

# ============================================================
step 4 "Set Environment Variables"
# ============================================================
export PATH="${XCUDA_PATH}/bin:${PATH}"
export CUDA_PATH="${XCUDA_PATH}"
export LD_LIBRARY_PATH="${XCUDA_PATH}/lib64:${XCUDA_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L${XCUDA_PATH}/lib64/"
export XTRANS_DIR="${XCUDA_PATH}"
export CXX="${XCUDA_PATH}/bin/clang++"
export CUDNN_ROOT="${XCUDA_PATH}/targets/x86_64-linux"
export CUPTI_ROOT="${XCUDA_PATH}/targets/x86_64-linux"
export NCCL_ROOT
export XMLIR_CUDNN_ENABLED=true
export CMAKE_CUDA_ARCHITECTURES=80

# Ensure git trusts this repo and all submodules (needed when running as root)
git config --global --add safe.directory '*'

# Create a git tag if none exist — setup.py uses `git describe` for version detection.
# Without any tag, it prints "fatal: No names found" to stderr.
if ! git -C "$SCRIPT_DIR" describe --tags HEAD >/dev/null 2>&1; then
    git -C "$SCRIPT_DIR" tag -f v0.0.0-m100-dev HEAD 2>/dev/null || true
fi

# Find Python >= 3.9 (with numpy installed)
PYTHON_BIN=""
# Check conda envs first (most likely to have numpy)
for p in /root/miniconda/envs/python310_torch25_cuda/bin/python3 \
         /root/miniconda/envs/*/bin/python3; do
    if [ -x "$p" ] 2>/dev/null; then
        if "$p" -c "import sys; assert sys.version_info >= (3,9); import numpy" 2>/dev/null; then
            PYTHON_BIN="$p"
            break
        fi
    fi
done
# Fallback: search PATH
if [ -z "$PYTHON_BIN" ]; then
    for candidate in python3.10 python3.11 python3.12 python3; do
        p=$(command -v "$candidate" 2>/dev/null)
        if [ -n "$p" ] && "$p" -c "import sys; assert sys.version_info >= (3,9); import numpy" 2>/dev/null; then
            PYTHON_BIN="$p"
            break
        fi
    done
fi
if [ -z "$PYTHON_BIN" ]; then
    fail "No Python >= 3.9 found"
    exit 1
fi
ok "Python: ${GREEN}$PYTHON_BIN${NC} ($($PYTHON_BIN --version))"

# Put Python's bin dir first on PATH so cmake finds matching pip/numpy
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

info "XCUDA_PATH  = ${GREEN}$XCUDA_PATH${NC}"
info "CXX         = ${GREEN}$CXX${NC}"
info "CUDA_PATH   = ${GREEN}$CUDA_PATH${NC}"
info "CUDNN_ROOT  = ${GREEN}$CUDNN_ROOT${NC}"
info "NCCL_ROOT   = ${GREEN}$NCCL_ROOT${NC}"

if command -v nvcc &>/dev/null; then
    ok "nvcc found: $(nvcc --version 2>&1 | grep -o 'release.*' | head -1)"
else
    fail "nvcc not found in PATH"
    exit 1
fi

# Set proxy for PyPI access (cmake installs pybind11-stubgen, pip install later)
# BOS domains (*.bcebos.com) must bypass proxy — proxy returns 403 for them
export http_proxy=http://10.162.37.16:8128
export https_proxy=http://10.162.37.16:8128
export no_proxy="*.bcebos.com,.bcebos.com,bcebos.com,mirrors.baidu.com"

# ============================================================
step 5 "CMake Configure"
# ============================================================
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clear stale CMake cache to avoid architecture/python detection conflicts
if [ -f CMakeCache.txt ]; then
    CACHED_COMPILER=$(grep "CMAKE_CUDA_COMPILER:STRING=" CMakeCache.txt 2>/dev/null | cut -d= -f2)
    CACHED_PYTHON=$(grep "PYTHON_EXECUTABLE:FILEPATH=" CMakeCache.txt 2>/dev/null | cut -d= -f2)
    if [ "$CACHED_COMPILER" != "${XCUDA_PATH}/bin/nvcc" ] || [ "$CACHED_PYTHON" != "$PYTHON_BIN" ]; then
        warn "Stale CMakeCache (compiler or python mismatch), removing..."
        rm -f CMakeCache.txt
        rm -rf CMakeFiles/
    fi
fi

info "Build directory: ${DIM}$BUILD_DIR${NC}"
info "CMake options:"
echo -e "  ${DIM}CMAKE_BUILD_TYPE       = Release"
echo -e "  CMAKE_CUDA_COMPILER   = ${XCUDA_PATH}/bin/nvcc"
echo -e "  CMAKE_CXX_COMPILER    = ${XCUDA_PATH}/bin/clang++"
echo -e "  CMAKE_CUDA_ARCHITECTURES = 80 (skip auto-detect)"
echo -e "  WITH_GPU=ON  WITH_TESTING=OFF  WITH_NCCL=ON"
echo -e "  WITH_DISTRIBUTE=ON  WITH_CUDNN_FRONTEND=OFF  WITH_CINN=OFF${NC}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${XCUDA_PATH}/bin/nvcc" \
    -DCMAKE_CXX_COMPILER="${XCUDA_PATH}/bin/clang++" \
    -DCMAKE_C_COMPILER="${XCUDA_PATH}/bin/clang" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDNN_ROOT="${XCUDA_PATH}/targets/x86_64-linux" \
    -DCUPTI_ROOT="${XCUDA_PATH}/targets/x86_64-linux" \
    -DNCCL_ROOT="${NCCL_ROOT}" \
    -DWITH_GPU=ON \
    -DWITH_PYTHON=ON \
    -DWITH_TESTING=OFF \
    -DWITH_NCCL=ON \
    -DWITH_MPI=OFF \
    -DWITH_DISTRIBUTE=ON \
    -DWITH_CUDNN_FRONTEND=OFF \
    -DWITH_CINN=OFF \
    -DWITH_TENSORRT=OFF \
    -DWITH_PROFILER=OFF \
    -DWITH_XBYAK=ON \
    -DON_INFER=OFF \
    -DWITH_PIP_CUDA_LIBRARIES=OFF \
    -DPY_VERSION=3.10 \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN"

ok "CMake configure complete"

# ============================================================
step 6 "Build (make -j$(nproc))"
# ============================================================
info "Starting parallel build with ${GREEN}$(nproc)${NC} jobs..."
BUILD_START=$(date +%s)

# Build libraries only (skip cmake's wheel packaging target — it has a known patchelf
# race condition with large .so files on patchelf 0.10, see python/CMakeLists.txt:103-108).
make -j"$(nproc)" copy_libpaddle paddle_python

BUILD_END=$(date +%s)
BUILD_ELAPSED=$(( BUILD_END - BUILD_START ))
ok "Build complete in ${GREEN}${BUILD_ELAPSED}s${NC} ($((BUILD_ELAPSED/60))m $((BUILD_ELAPSED%60))s)"

# ============================================================
step 7 "Build & Install Wheel"
# ============================================================
info "Packaging wheel (setup.py bdist_wheel)..."
cd "$BUILD_DIR/python"

# Install Python build dependencies (opt-einsum, astor, etc.)
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/python/requirements.txt" -q

# Set rpath for key .so files before packaging.
# Using simple $ORIGIN-relative paths (not nvidia pip paths — M100 uses xtrans SDK).
patchelf --set-rpath '$ORIGIN/../libs/' "$BUILD_DIR/python/paddle/base/libpaddle.so"
for lib in libphi.so libphi_core.so libphi_gpu.so libpir.so; do
    if [ -f "$BUILD_DIR/python/paddle/libs/$lib" ]; then
        patchelf --set-rpath '$ORIGIN' "$BUILD_DIR/python/paddle/libs/$lib"
    fi
done
ok "rpath set on shared libraries"

# Build wheel (skip setup.py's own patchelf by setting rpath beforehand — it's idempotent)
"$PYTHON_BIN" setup.py bdist_wheel 2>&1 | tee /tmp/wheel_build.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    warn "First wheel build attempt failed, retrying..."
    "$PYTHON_BIN" setup.py bdist_wheel
fi
WHEEL=$(find "${BUILD_DIR}/python/dist" -name "*.whl" | head -1)
if [ -z "$WHEEL" ]; then
    fail "No wheel found in ${DIM}build/python/dist/${NC}"
    exit 1
fi
info "Found wheel: ${DIM}$(basename "$WHEEL")${NC}"
info "Installing with pip..."
"$PYTHON_BIN" -m pip install --force-reinstall "$WHEEL"
ok "Wheel installed: ${GREEN}$(basename "$WHEEL")${NC}"

# ============================================================
step 8 "Run Demo on M100 Simulator"
# ============================================================
SIMULATOR_SO="${SIMULATOR_DIR}/output/xse-ubuntu_2004_x86_64/so/libxpusim.so"
if [ ! -f "$SIMULATOR_SO" ]; then
    fail "Simulator library not found: ${DIM}$SIMULATOR_SO${NC}"
    exit 1
fi

info "Configuring simulator environment..."
export XPU_SIMULATOR_MODE=1
export CUDA_AMODEL_DLL="$SIMULATOR_SO"
export CUDA_AMODEL_GPU=KL004
export XPUSIM_DEVICE_MODEL=XCN
export LD_LIBRARY_PATH="${SIMULATOR_DIR}/output/xse-ubuntu_2004_x86_64/so:${LD_LIBRARY_PATH}"

echo -e "  ${DIM}XPU_SIMULATOR_MODE  = 1"
echo -e "  CUDA_AMODEL_GPU   = KL004"
echo -e "  XPUSIM_DEVICE_MODEL = XCN"
echo -e "  CUDA_AMODEL_DLL   = ...libxpusim.so${NC}"
echo ""

info "Running inline M100 simulator verification ..."
echo ""

cd "$SCRIPT_DIR"
if "$PYTHON_BIN" - <<'PY'
import numpy as np
import paddle

checks = []
paddle.set_device('gpu:0')
checks.append(('device_setup', True))

x = paddle.randn([2, 3])
checks.append(('randn', tuple(x.shape) == (2, 3)))

a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
b = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float32')
checks.append(('matmul_gemm', np.allclose(paddle.matmul(a, b).numpy(), [[19, 22], [43, 50]])))

checks.append(('elementwise', np.allclose((a + b).numpy(), [[6, 8], [10, 12]])))
checks.append(('reduce', np.allclose(paddle.sum(a).numpy(), 10.0)))

batch_a = paddle.reshape(paddle.arange(8, dtype='float32'), [2, 2, 2])
batch_b = paddle.ones([2, 2, 2], dtype='float32')
checks.append(('batch_matmul', tuple(paddle.matmul(batch_a, batch_b).shape) == (2, 2, 2)))

clone = a.clone()
checks.append(('clone_copy', np.allclose(clone.numpy(), a.numpy())))

passed = 0
print('=' * 50)
print('M100 Simulator Verification')
print('=' * 50)
for name, ok in checks:
    print(f'  [{name}] {"PASS" if ok else "FAIL"}')
    passed += int(ok)
print('=' * 50)
print(f'Results: {passed}/{len(checks)} passed')
if passed != len(checks):
    raise SystemExit(1)
print('ALL TESTS PASSED')
PY
then
    echo ""
    ok "All verification passed!"
    echo -e "${GREEN}${BOLD}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║         BUILD & TEST SUCCEEDED           ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
else
    echo ""
    fail "Some tests failed"
    echo -e "${RED}${BOLD}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║          VERIFICATION FAILED             ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
    exit 1
fi
