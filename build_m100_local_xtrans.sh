#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SIMULATOR_URL="https://klx-sdk-release-public.su.bcebos.com/xse/mars/release/latest/output.tar.gz"

XCUDA_PATH="${PARENT_DIR}/xtrans/output"
SIMULATOR_DIR="${PARENT_DIR}/xse_simulator"

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
           echo -e "${BOLD}${CYAN}  [$1/7] $2${NC}"; \
           echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   M100 Paddle Build & Verification      ║"
echo "  ║   Branch: m100 | Local xtrans output    ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================
step 1 "Use local xtrans build output"
# ============================================================
if [ ! -d "$XCUDA_PATH" ]; then
    fail "Local xtrans output not found: ${DIM}$XCUDA_PATH${NC}"
    fail "Build xtrans first: ${DIM}bash ${PARENT_DIR}/xtrans/script/build.sh -c xpu4${NC}"
    exit 1
fi

required_paths=(
    "$XCUDA_PATH/bin/nvcc"
    "$XCUDA_PATH/bin/clang++"
    "$XCUDA_PATH/bin/clang"
    "$XCUDA_PATH/lib64/libcudnn.so"
    "$XCUDA_PATH/targets/x86_64-linux/include/cudnn_api/cudnn.h"
)

for p in "${required_paths[@]}"; do
    if [ ! -e "$p" ]; then
        fail "Required local xtrans artifact missing: ${DIM}$p${NC}"
        exit 1
    fi
done
ok "Using local xtrans output: ${DIM}$XCUDA_PATH${NC}"

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
step 3 "Set Environment Variables"
# ============================================================
export PATH="${XCUDA_PATH}/bin:${PATH}"
export CUDA_PATH="${XCUDA_PATH}"
export LD_LIBRARY_PATH="${XCUDA_PATH}/lib64:${XCUDA_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L${XCUDA_PATH}/lib64/"
export XTRANS_DIR="${XCUDA_PATH}"
export CXX="${XCUDA_PATH}/bin/clang++"
export CUDNN_ROOT="${XCUDA_PATH}/targets/x86_64-linux"
export CUPTI_ROOT="${XCUDA_PATH}/targets/x86_64-linux"
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
step 4 "CMake Configure"
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
echo -e "  WITH_GPU=ON  WITH_TESTING=OFF  WITH_NCCL=OFF"
echo -e "  WITH_CUDNN_FRONTEND=OFF  WITH_CINN=OFF${NC}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${XCUDA_PATH}/bin/nvcc" \
    -DCMAKE_CXX_COMPILER="${XCUDA_PATH}/bin/clang++" \
    -DCMAKE_C_COMPILER="${XCUDA_PATH}/bin/clang" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDNN_ROOT="${XCUDA_PATH}/targets/x86_64-linux" \
    -DCUPTI_ROOT="${XCUDA_PATH}/targets/x86_64-linux" \
    -DWITH_GPU=ON \
    -DWITH_PYTHON=ON \
    -DWITH_TESTING=OFF \
    -DWITH_NCCL=OFF \
    -DWITH_MPI=OFF \
    -DWITH_DISTRIBUTE=OFF \
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
step 5 "Build (make -j$(nproc))"
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
step 6 "Build & Install Wheel"
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
WHEEL_LOG=$(mktemp /tmp/m100_wheel_build.XXXXXX.log)
"$PYTHON_BIN" setup.py bdist_wheel 2>&1 | tee "$WHEEL_LOG"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    warn "First wheel build attempt failed, retrying..."
    "$PYTHON_BIN" setup.py bdist_wheel
fi
WHEEL=$(find "${BUILD_DIR}/python/dist" -name "*.whl" | head -1)
if [ -z "$WHEEL" ]; then
    fail "No wheel found in ${DIM}build/python/dist/${NC}"
    fail "Wheel build log: ${DIM}$WHEEL_LOG${NC}"
    exit 1
fi
info "Wheel build log: ${DIM}$WHEEL_LOG${NC}"
info "Found wheel: ${DIM}$(basename "$WHEEL")${NC}"
info "Installing with pip..."
"$PYTHON_BIN" -m pip install --force-reinstall "$WHEEL"
ok "Wheel installed: ${GREEN}$(basename "$WHEEL")${NC}"

# ============================================================
step 7 "Run Demo on M100 Simulator"
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

info "Running ${GREEN}demo_m100.py${NC} ..."
echo ""

cd "$SCRIPT_DIR"
if "$PYTHON_BIN" demo_m100.py; then
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
