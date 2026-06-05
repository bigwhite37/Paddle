# M100 `USE_CUDA` 全局定义必要性验证

## 目标

验证 `cmake/configure.cmake` 中 `WITH_GPU` 下的全局定义：

```cmake
add_definitions(-DUSE_CUDA)
```

是否属于 M100/xtrans 适配的最小必要改动，以及如果确实需要，应如何收敛它的作用范围。

## 背景

`847020a` 在 `cmake/configure.cmake` 的 `WITH_GPU` 分支中增加了 `-DUSE_CUDA`。这个宏不是 Paddle 既有的 GPU 构建语义宏，名称也过于泛化，因此需要验证它是否真的必须全局保留。

## 静态检查

### Paddle 仓库内的 `USE_CUDA`

命令：

```bash
git grep -n "USE_CUDA" -- .
```

关键结果：

```text
cmake/configure.cmake:112:  add_definitions(-DUSE_CUDA)
```

除历史文档、测试环境变量、以及名称不同的 `USE_CUDA_ONLY_OP` / `USE_CUDA_ATOMIC` 外，Paddle 源码本身没有消费裸 `USE_CUDA` 宏。

### xtrans 头文件中的 `USE_CUDA`

命令：

```bash
grep -R -n "USE_CUDA" \
  /home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include
```

关键结果：

```text
.../include/xpudnn/xpudnn.h:4:#if !defined(USE_CUDA)
.../include/xpudnn/xpudnn.h:18:#if defined(__KL3_XRE__) || defined(USE_CUDA)
.../include/cudnn_api/cudnn.h:6:#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
```

相关头文件逻辑：

```cpp
// cudnn_api/cudnn.h
#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"

#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

```cpp
// xpudnn/xpudnn.h
#if !defined(USE_CUDA)
#include "xpu/refactor/context/context.h"
namespace xdnn = baidu::xpu::api;
#include "xpudnn/ops/xpudnn_api.h"
#endif

#if defined(__KL3_XRE__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#include "library_types.h"
#else
#include "xpu/runtime.h"
#endif
```

`USE_CUDA` 的实际作用主要在 xtrans 侧：

1. 避免 `cudnn_api/cudnn.h` 继续包含 `xpudnn_cuda_patch.h`。
2. 让 `xpudnn/xpudnn.h` 选择 CUDA runtime 头，而不是 XPU runtime 头。

### Paddle 中 cuDNN 头的扩散路径

命令：

```bash
git grep -n "backends/dynload/cudnn.h\|<cudnn.h>\|\"cudnn.h\"" -- paddle | head -80
```

关键结果包括：

```text
paddle/phi/backends/dynload/cudnn.h:17:#include <cudnn.h>
paddle/phi/backends/gpu/gpu_types.h:29:#include "paddle/phi/backends/dynload/cudnn.h"
paddle/phi/core/enforce.h:19:#include <cudnn.h>
paddle/phi/core/enforce.h:52:#include "paddle/phi/backends/dynload/cudnn.h"
paddle/fluid/platform/enforce.h:34:#include <cudnn.h>
paddle/fluid/platform/enforce.h:76:#include "paddle/phi/backends/dynload/cudnn.h"
paddle/phi/backends/gpu/gpu_context.cc:39:#include "paddle/phi/backends/dynload/cudnn.h"
paddle/phi/backends/gpu/gpu_resources.cc:29:#include "paddle/phi/backends/dynload/cudnn.h"
paddle/phi/core/platform/device_context.h:32:#include "paddle/phi/backends/dynload/cudnn.h"
```

判断：`<cudnn.h>` / `paddle/phi/backends/dynload/cudnn.h` 不是只出现在单个 dynload 源文件中，它会通过 GPU 公共类型、enforce、device context 等公共头进入大量 C++ / CUDA 编译单元。因此 `USE_CUDA` 如果需要，不能只看 `cudnn.cc` 单个对象能否编译。

### `xpudnn_cuda_patch.h` 的影响

相关片段：

```cpp
#define cudaStream_t XPUStream
#define cudaFree(A) ;
#define cudaMalloc(A,B) ;
#define cudaMemcpy(A,B,C,D) 0
#define __half        float16
#define __nv_bfloat16 bfloat16
```

这说明如果不定义 `USE_CUDA`，包含 `cudnn_api/cudnn.h` 的编译单元可能受到 xtrans CUDA patch 的宏替换影响，典型风险包括：

- `cudaStream_t` 被宏替换成 `XPUStream`；
- `cudaMalloc` / `cudaFree` / `cudaMemcpy` 被替换为空实现或常量；
- `__half` / `__nv_bfloat16` 被替换成 xtrans 类型。

## 构建验证

### 验证步骤

1. 临时移除 `cmake/configure.cmake` 中的全局：

```cmake
add_definitions(-DUSE_CUDA)
```

2. 重新生成现有 build 目录：

```bash
cmake -S . -B build
```

结果：CMake configure/generate 成功。

3. 确认核心目标生成 flags 中不再包含 `-DUSE_CUDA`：

```bash
grep -R -n -- "-DUSE_CUDA" "build/paddle/phi" "build/paddle/fluid" | head -20
```

结果：无输出，说明 `phi` / `fluid` 相关生成 flags 已不再带 `-DUSE_CUDA`。

4. 在不带全局 `-DUSE_CUDA` 的情况下重建：

```bash
cmake --build build --target phi_core phi_gpu -- -j$(nproc)
```

结果：通过。

关键输出末尾：

```text
254 warnings generated when compiling for host.
Linking CXX shared library libphi_gpu.so
Built target phi_gpu
```

判断：当前 build 配置下 `WITH_CUDNN_FRONTEND=OFF`，`phi_core` / `phi_gpu` targeted build 在移除全局 `-DUSE_CUDA` 后仍可编译通过。因此，这个 targeted build 不能证明 `cmake/configure.cmake` 中的全局 `add_definitions(-DUSE_CUDA)` 对当前默认 `phi_core` / `phi_gpu` 构建是必需的。

## 最小编译对照验证

除了 Paddle target build，还用 `ku-docs/tests/test_cudnn_dynload_xtrans.cu` 做了独立编译对照。这个程序同时包含 CUDA runtime 和 xtrans cuDNN 头，并会使用 `cudaStreamCreate` / `cudaStreamDestroy`，可以直接暴露 `xpudnn_cuda_patch.h` 对 CUDA runtime 类型的影响。

### 不加 `-DUSE_CUDA`

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda
```

结果：失败。

关键输出：

```text
ku-docs/tests/test_cudnn_dynload_xtrans.cu:41:30: error: no matching function for call to 'cudaStreamCreate'
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/include/cuda_runtime_api.h:2284:42: note: candidate function not viable: no known conversion from 'XPUStream *' (aka 'void **') to 'cudaStream_t *' (aka 'CUstream_st **') for 1st argument
ku-docs/tests/test_cudnn_dynload_xtrans.cu:74:30: error: no matching function for call to 'cudaStreamDestroy'
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/include/cuda_runtime_api.h:2528:74: note: candidate function not viable: cannot convert argument of incomplete type 'XPUStream' (aka 'void *') to 'cudaStream_t' (aka 'CUstream_st *') for 1st argument
2 errors generated when compiling for xcn.
```

判断：不定义 `USE_CUDA` 时，`cudnn_api/cudnn.h` 会包含 `xpudnn_cuda_patch.h`，其中 `#define cudaStream_t XPUStream` 会污染后续 CUDA runtime API 的类型检查。

### 加 `-DUSE_CUDA`

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -DUSE_CUDA \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_with_use_cuda
```

结果：通过编译。

判断：`USE_CUDA` 对 xtrans cuDNN 头的影响是真实存在的；它能避开 `xpudnn_cuda_patch.h` 的 CUDA runtime 类型污染。

## 恢复验证态

完成移除验证后，已把 `cmake/configure.cmake` 中的：

```cmake
add_definitions(-DUSE_CUDA)
```

恢复，并重新执行：

```bash
cmake -S . -B build
```

恢复后确认 `phi_core` / `phi_gpu` 的 flags 重新包含 `-DUSE_CUDA`：

```text
build/paddle/phi/CMakeFiles/phi_core.dir/flags.make: CUDA_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_core.dir/flags.make: CXX_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_gpu.dir/flags.make: CUDA_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_gpu.dir/flags.make: CXX_DEFINES ... -DUSE_CUDA ...
```

## 当前结论

1. `USE_CUDA` 的技术作用是真实存在的，但责任边界在 xtrans 头文件侧，不是 Paddle 源码自身语义。
   - Paddle 源码没有消费裸 `USE_CUDA` 宏。
   - xtrans 的 `cudnn_api/cudnn.h` / `xpudnn/xpudnn.h` 会根据 `USE_CUDA` 选择是否启用 `xpudnn_cuda_patch.h` 和是否走 CUDA runtime 头。

2. 最小编译对照证明：如果某个编译单元同时包含 xtrans cuDNN 头和 CUDA runtime API，不定义 `USE_CUDA` 会触发类型污染并编译失败。
   - 失败根因是 `xpudnn_cuda_patch.h` 中的 `#define cudaStream_t XPUStream`。
   - 加 `-DUSE_CUDA` 后同一验证程序可编译通过。

3. 当前 Paddle targeted build 结果证明：在当前配置 `WITH_CUDNN_FRONTEND=OFF` 下，移除全局 `-DUSE_CUDA` 后 `phi_core` / `phi_gpu` 仍可构建通过。
   - 因此，这次 targeted build 不能证明 `cmake/configure.cmake` 的全局 `add_definitions(-DUSE_CUDA)` 是当前默认构建的必要最小改动。
   - 这也解释了为什么仅以 `phi_core` / `phi_gpu` 默认构建通过与否来判断会不充分：当前配置下很多 cuDNN include 路径被 `WITH_CUDNN_FRONTEND` 条件挡住。

4. 最小化建议：
   - 不建议把 `add_definitions(-DUSE_CUDA)` 作为默认最小 Paddle functional patch 保留。
   - 如果后续开启 `WITH_CUDNN_FRONTEND` 或其他目标确实需要 xtrans cuDNN 头与 CUDA runtime API 混用，应优先推动 xtrans 用更准确的宏或头文件策略解决，而不是让 Paddle 全局定义一个泛化的 `USE_CUDA`。
   - 如果短期必须在 Paddle 构建侧兜底，也应尽量缩小作用域，例如只加到实际包含 xtrans cuDNN 头且会受 `xpudnn_cuda_patch.h` 影响的 target / source 编译参数，而不是所有 `WITH_GPU` 编译单元。
