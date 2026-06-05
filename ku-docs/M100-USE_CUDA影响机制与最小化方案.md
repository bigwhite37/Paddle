# M100 `-DUSE_CUDA` 影响机制与最小化方案

## 目标

说明 `cmake/configure.cmake` 中：

```cmake
if(WITH_GPU)
  add_definitions(-DUSE_CUDA)
endif()
```

到底影响了什么、为什么会影响 M100 构建、为什么当前写法不是最小影响写法，以及后续如何收敛。

## 一句话结论

`-DUSE_CUDA` 本身是有技术作用的：它告诉 xtrans 的 cuDNN 头文件“当前是 CUDA-style 编译环境，不要启用会把 CUDA 类型替换成 XPU 类型的 patch”。

但当前把它放在 `if(WITH_GPU)` 下用 `add_definitions()` 全局加给整个 GPU 构建，作用域过大，不是最小化影响的做法。

更准确地说：

- **需要解决的问题**：xtrans 的 `cudnn_api/cudnn.h` 在未定义 `USE_CUDA` 时会包含 `xpudnn_cuda_patch.h`，这个 patch 会污染 CUDA runtime 类型和函数宏。
- **当前解决方式**：给所有 `WITH_GPU` 编译单元都定义 `USE_CUDA`。
- **最小化方向**：不要全局定义；如果短期必须在 Paddle 侧兜底，应只作用到确实会包含 xtrans cuDNN 头、且会受 `xpudnn_cuda_patch.h` 影响的编译单元或 target。

## `-DUSE_CUDA` 是什么

编译参数：

```bash
-DUSE_CUDA
```

等价于在每个被编译的 `.cc` / `.cu` 文件最前面写：

```cpp
#define USE_CUDA
```

所以它会影响所有后续被包含头文件中的条件判断，例如：

```cpp
#if defined(USE_CUDA)
// 走 CUDA 分支
#else
// 走非 CUDA 或兼容 patch 分支
#endif
```

在当前问题里，`USE_CUDA` 主要不是 Paddle 源码自己消费，而是 xtrans 头文件消费。

## xtrans cuDNN 头为什么关心 `USE_CUDA`

xtrans 的 cuDNN 入口头大致是：

```cpp
#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"

#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

也就是说：

- 定义了 `USE_CUDA`：不包含 `xpudnn_cuda_patch.h`。
- 没定义 `USE_CUDA`：会包含 `xpudnn_cuda_patch.h`。

`xpudnn/xpudnn.h` 中也有类似分支：

```cpp
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

直观理解：

- 有 `USE_CUDA`：xtrans cuDNN 头按 CUDA runtime 语义展开。
- 没有 `USE_CUDA`：xtrans cuDNN 头会进入 XPU patch / XPU runtime 兼容语义。

## 不定义 `USE_CUDA` 时的实际影响

`xpudnn_cuda_patch.h` 里有宏替换：

```cpp
#define cudaStream_t XPUStream

#define cudaFree(A) ;
#define cudaMalloc(A,B) ;
#define cudaMemcpy(A,B,C,D) 0

#define __half        float16
#define __nv_bfloat16 bfloat16
```

这不是普通类型声明，而是预处理器级别的文本替换。

例如 Paddle 或验证程序本来写：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

如果 `xpudnn_cuda_patch.h` 生效，预处理后会变成接近：

```cpp
XPUStream stream;
cudaStreamCreate(&stream);
```

但 `cudaStreamCreate` 是 CUDA runtime API，它需要的是 `cudaStream_t*`，不是 `XPUStream*`，于是会报类型不匹配。

实际验证中，不加 `-DUSE_CUDA` 编译 `ku-docs/tests/test_cudnn_dynload_xtrans.cu` 会失败：

```text
error: no matching function for call to 'cudaStreamCreate'
 candidate function not viable: no known conversion from 'XPUStream *' (aka 'void **') to 'cudaStream_t *' (aka 'CUstream_st **') for 1st argument

error: no matching function for call to 'cudaStreamDestroy'
 candidate function not viable: cannot convert argument of incomplete type 'XPUStream' (aka 'void *') to 'cudaStream_t' (aka 'CUstream_st *') for 1st argument
```

同一个程序加上 `-DUSE_CUDA` 后可以编译通过。

这说明：`-DUSE_CUDA` 的作用不是虚的，它确实能阻止 xtrans cuDNN 头启用会污染 CUDA runtime API 的 patch。

## 为什么 `if(WITH_GPU)` 会影响 M100

当前 M100 适配不是 Paddle 里的独立后端，不是：

```cmake
WITH_M100=ON
```

而是继续走 Paddle 原来的 CUDA/GPU 后端：

```cmake
WITH_GPU=ON
```

差异在于底层工具链和库路径换成了 xtrans：

```text
Paddle 构建语义：WITH_GPU=ON
CUDA compiler：xtrans nvcc
CXX compiler：xtrans clang++
cuDNN include：xtrans cudnn_api/cudnn.h
cuBLAS include：xtrans cublas_v2.h / cublas_api.h
```

所以 `WITH_GPU` 对 M100 构建同样成立。

在 Paddle 语义里，`WITH_GPU` 表示“构建 CUDA-like GPU 后端”，不等于“只构建 NVIDIA 真 CUDA”。M100/xtrans 当前就是复用这条 CUDA-like GPU 后端路径。

因此：

```cmake
if(WITH_GPU)
  add_definitions(-DUSE_CUDA)
endif()
```

会进入 M100 构建，并影响 M100 编译命令。

## 当前全局写法的影响范围

`add_definitions(-DUSE_CUDA)` 是目录级 / 全局式写法。放在 `cmake/configure.cmake` 的 `WITH_GPU` 分支下后，它会进入大量 C / C++ / CUDA 编译命令。

恢复当前写法后，`phi_core` / `phi_gpu` 的 flags 中可见：

```text
build/paddle/phi/CMakeFiles/phi_core.dir/flags.make: CUDA_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_core.dir/flags.make: CXX_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_gpu.dir/flags.make: CUDA_DEFINES ... -DUSE_CUDA ...
build/paddle/phi/CMakeFiles/phi_gpu.dir/flags.make: CXX_DEFINES ... -DUSE_CUDA ...
```

这意味着它不是只影响 `cudnn.cc` 或某几个 cuDNN kernel，而是影响整个 `WITH_GPU` 构建下的大量目标。

## 为什么说它不是最小影响写法

真正需要避免的是这个路径：

```text
包含 xtrans cudnn_api/cudnn.h
  -> 未定义 USE_CUDA
    -> 包含 xpudnn_cuda_patch.h
      -> cudaStream_t / cudaMalloc / cudaMemcpy / __nv_bfloat16 等被宏替换
```

也就是说，真正需要保护的是“会包含 xtrans cuDNN 头且会继续使用 CUDA runtime 语义”的那批编译单元。

但当前写法保护的是：

```text
所有 WITH_GPU 编译单元
```

这两者范围不一致。当前范围明显更大。

它的问题包括：

1. **名字过于泛化**
   - `USE_CUDA` 不是 Paddle 原生语义宏。
   - 未来任何头文件只要也判断 `USE_CUDA`，都会被这个全局定义影响。

2. **作用域过大**
   - 它进入所有 `WITH_GPU` 编译单元，不只是 cuDNN 相关编译单元。

3. **把 xtrans 兼容细节扩散到 Paddle 全局构建语义**
   - 用户目标是 GPU/M100 构建尽量无感。
   - 全局 `USE_CUDA` 会让 Paddle CMake 层背上 xtrans 头文件策略的细节。

4. **不解决 cuDNN dynload 的运行时符号问题**
   - `-DUSE_CUDA` 只影响编译期头文件分支。
   - 它不能让 xtrans `libcudnn.so` 额外导出 `cudnn*` 符号。
   - Paddle dynload 的 `dlsym("cudnn...")` 风险仍然需要 xtrans ABI alias/wrapper 或其他方案解决。

## 为什么移除全局 `-DUSE_CUDA` 后当前 targeted build 还能过

验证过：临时移除 `cmake/configure.cmake` 中的全局 `add_definitions(-DUSE_CUDA)` 后，重新生成并构建：

```bash
cmake --build build --target phi_core phi_gpu -- -j$(nproc)
```

当前配置下通过。

关键原因是当前 build 配置里：

```text
WITH_CUDNN_FRONTEND=OFF
```

很多 cuDNN include 路径被条件挡住，例如：

```cpp
#ifdef WITH_CUDNN_FRONTEND
#include <cudnn.h>
#endif
```

所以这次 `phi_core` / `phi_gpu` targeted build 通过只能说明：

```text
当前默认配置下，phi_core / phi_gpu 没有充分触发会被 xpudnn_cuda_patch.h 搞坏的路径。
```

不能说明：

```text
所有 M100/Paddle 构建都不需要 USE_CUDA。
```

最小验证程序已经证明，只要确实进入“xtrans cuDNN 头 + CUDA runtime API”混用路径，不定义 `USE_CUDA` 就会失败。

## Paddle 当前 CMake 结构对局部化的限制

局部化 `USE_CUDA` 时需要注意 Paddle 的 target 组织方式。

`paddle/phi/backends/dynload/CMakeLists.txt` 里并没有单独生成 Linux 下的 `dynload_cudnn` target，而是把 `cudnn.cc` 收集进 `backends_srcs`：

```cmake
elseif(WITH_GPU)
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${CUDA_SRCS})
endif()
```

`collect_srcs()` 的实现只是把源码路径追加到缓存变量：

```cmake
function(collect_srcs SRC_GROUP)
  ...
  foreach(src ${prefix_SRCS})
    set(${SRC_GROUP}
        "${${SRC_GROUP}};${CMAKE_CURRENT_SOURCE_DIR}/${src}"
        CACHE INTERNAL "")
  endforeach()
endfunction()
```

最后 `paddle/phi/CMakeLists.txt` 把这些源码统一放进大 target：

```cmake
nv_library(
  phi_core ${PHI_BUILD_TYPE}
  SRCS ${PHI_CORE_SRCS}
  DEPS ${PHI_DEPS})

nv_library(
  phi_gpu ${PHI_BUILD_TYPE}
  SRCS ${PHI_GPU_SRCS}
  DEPS ${PHI_DEPS} phi_core)
```

因此，不能简单说“把 `USE_CUDA` 加到 dynload 子目录 target 上”就能解决，因为 Linux 下这里没有独立的 `dynload_cudnn` target；相关源码最后进入的是 `phi_core` / `phi_gpu` 这样的大 target。

这会影响最小化方案的可行性排序。

## 最小化方案分析

### 方案 0：在 xtrans 侧修头文件策略

这是最干净的方向。

理想状态是 xtrans 不需要 Paddle 全局定义 `USE_CUDA`，也不会在 CUDA-style 编译环境里默认启用 `xpudnn_cuda_patch.h`。

可以考虑的 xtrans 侧方向：

1. 使用更准确的内部宏名，而不是要求上层项目定义泛化的 `USE_CUDA`。
2. 在 `cudnn_api/cudnn.h` 中根据实际 include/runtime 条件决定是否需要 `xpudnn_cuda_patch.h`。
3. 把 `xpudnn_cuda_patch.h` 中危险的宏替换改成更窄的声明或适配，不要全局替换 `cudaStream_t`、`cudaMalloc`、`cudaMemcpy`。
4. 对 Paddle dynload 场景补齐 `libcudnn.so` 的 `cudnn*` ABI alias/wrapper，避免只靠头文件宏映射。

优点：

- Paddle 侧最小。
- 不污染 Paddle CMake 语义。
- 对其他使用 xtrans cuDNN 的项目也更稳。

缺点：

- 需要 xtrans SDK 修改和发布。
- 如果当前只能改 Paddle 源码，短期不可直接落地。

推荐级别：最高，作为最终收敛方向。

### 方案 1：完全移除 Paddle 侧全局 `-DUSE_CUDA`

这就是把 `cmake/configure.cmake` 中的：

```cmake
add_definitions(-DUSE_CUDA)
```

直接移除。

当前验证结果：

- `WITH_CUDNN_FRONTEND=OFF` 下，`phi_core` / `phi_gpu` targeted build 可以通过。
- 但最小验证程序显示，只要进入 xtrans cuDNN 头和 CUDA runtime API 混用路径，不加 `USE_CUDA` 会失败。

优点：

- Paddle 侧最小，没有额外 CMake 宏。
- 当前默认 targeted build 可通过。

风险：

- 开启 `WITH_CUDNN_FRONTEND` 后可能失败。
- 其他 cuDNN 相关 target / kernel 如果包含 xtrans `cudnn.h` 并继续使用 CUDA runtime API，也可能失败。
- 风险会在更完整构建或功能覆盖时暴露。

适用条件：

- 只要求当前默认 `WITH_CUDNN_FRONTEND=OFF` 构建通过。
- 同时明确记录：cuDNN frontend / dynload 相关路径尚未完全覆盖。

推荐级别：可作为最小补丁候选，但需要明确验证边界，不能宣称覆盖所有 cuDNN 场景。

### 方案 2：只在开启 `WITH_CUDNN_FRONTEND` 时定义 `USE_CUDA`

思路：

```cmake
if(WITH_GPU AND WITH_CUDNN_FRONTEND)
  add_definitions(-DUSE_CUDA)
endif()
```

优点：

- 比所有 `WITH_GPU` 构建都定义更窄。
- 与当前观察相符：很多 cuDNN include 路径受 `WITH_CUDNN_FRONTEND` 控制。

问题：

- 并不是所有 cuDNN 头包含都一定只受 `WITH_CUDNN_FRONTEND` 控制。
- 例如部分 kernel/helper 可能直接或间接包含 `paddle/phi/backends/dynload/cudnn.h`。
- 如果某些非 frontend 路径也包含 xtrans cuDNN 头，这个条件不够。

适用条件：

- 需要进一步 grep 和 targeted build 证明 cuDNN 头污染只在 `WITH_CUDNN_FRONTEND` 打开时会出现。

推荐级别：中等。比当前全局 `WITH_GPU` 更窄，但仍然是 CMake 全局定义，且边界可能不完整。

### 方案 3：target 级别定义到 `phi_core` / `phi_gpu`

思路：移除 `configure.cmake` 的全局定义，改成：

```cmake
target_compile_definitions(phi_core PRIVATE USE_CUDA)
target_compile_definitions(phi_gpu PRIVATE USE_CUDA)
```

优点：

- 比 `add_definitions()` 更现代，作用域至少落在明确 target 上。
- 不会影响配置点之后所有目录和无关 target。

问题：

- `phi_core` / `phi_gpu` 本身非常大。
- 这仍然会影响大量源码，和真正需要的 cuDNN 相关编译单元相比还是过宽。
- 其他 target 如果也包含 xtrans cuDNN 头，可能还需要额外补。

适用条件：

- 想快速从目录级全局定义收敛到 target 级定义。
- 可以接受 `phi_core` / `phi_gpu` 内部仍然较宽。

推荐级别：中等偏低。比当前写法工程上更可控，但从“最小功能补丁”角度仍偏大。

### 方案 4：source file 级别定义到具体受影响文件

思路：对确实需要的源码加 `COMPILE_DEFINITIONS USE_CUDA`：

```cmake
set_source_files_properties(
  backends/dynload/cudnn.cc
  PROPERTIES COMPILE_DEFINITIONS USE_CUDA)
```

或者对少数 cuDNN kernel/helper 对应源文件设置。

优点：

- 作用域最小。
- 符合“只有受 xtrans cuDNN patch 影响的编译单元才定义”的原则。

问题：

- 当前 include 污染可能通过公共头扩散，受影响文件不一定只有 `cudnn.cc`。
- `phi_core` / `phi_gpu` 是大 target，源码很多，必须先准确找出所有会触发 xtrans cuDNN 头 + CUDA runtime API 混用的编译单元。
- 列表可能随 `WITH_CUDNN_FRONTEND`、kernel 开关、平台开关变化，维护成本高。

适用条件：

- 已经通过失败日志明确定位到少数具体 `.cc` / `.cu` 文件。
- 愿意为最小作用域承担文件列表维护成本。

推荐级别：理论上最小，但当前不建议直接上来做。应该等实际失败点明确后再按失败文件补。

### 方案 5：在 Paddle 的局部 wrapper 头里临时定义 / undef `USE_CUDA`

思路：在 `paddle/phi/backends/dynload/cudnn.h` 包含 `<cudnn.h>` 前局部定义：

```cpp
#ifndef USE_CUDA
#define USE_CUDA
#define PADDLE_DEFINED_USE_CUDA_FOR_CUDNN
#endif

#include <cudnn.h>

#ifdef PADDLE_DEFINED_USE_CUDA_FOR_CUDNN
#undef USE_CUDA
#undef PADDLE_DEFINED_USE_CUDA_FOR_CUDNN
#endif
```

优点：

- 作用域看起来只包住 `<cudnn.h>`。
- 不需要 CMake 全局定义。

问题：

- 这是头文件宏技巧，容易让行为变得隐蔽。
- 如果 `<cudnn.h>` 内部包含的其他头依赖 `USE_CUDA` 后续继续存在，`#undef` 可能引入新问题。
- 如果其他源码直接 `#include <cudnn.h>`，绕过 Paddle wrapper，这个方案无效。
- 用户之前已经明确不喜欢这类奇怪的宏 workaround；也不符合用户无感和最小清晰语义。

推荐级别：低。不建议作为首选。

### 方案 6：检测 xtrans 头后才定义 `USE_CUDA`

思路：CMake 检测当前 `CUDNN_INCLUDE_DIR` 是否是 xtrans 的 `cudnn_api`，只有命中时定义 `USE_CUDA`。

优点：

- 比普通 GPU 构建全局定义更精准，不影响 NVIDIA CUDA 构建。

问题：

- 会把 xtrans 路径 / 头文件布局写进 Paddle CMake。
- 用户目标是 M100/GPU 尽量无感，这会显式引入 xtrans 兼容概念。
- 之前已经明确不建议引入 `PADDLE_WITH_CUDA_COMPAT_LAYER` 这类 CMake 侧兼容概念。

推荐级别：低。除非短期必须规避 NVIDIA 构建影响，否则不建议。

## 推荐收敛顺序

### 首选：xtrans 侧修复

最终目标应是 Paddle 不需要定义 `USE_CUDA`。

xtrans 应该保证：当用户使用 xtrans CUDA SDK 以 CUDA-style 方式编译 Paddle 时，`cudnn_api/cudnn.h` 不会默认把 CUDA runtime 类型替换成 XPU 类型。

这也是最符合责任边界的方案。

### Paddle 侧短期策略

如果短期必须在 Paddle 侧保留兜底，建议按以下顺序推进：

1. **先移除全局 `add_definitions(-DUSE_CUDA)`，跑当前默认 targeted build。**
   - 这一步已验证：`WITH_CUDNN_FRONTEND=OFF` 下 `phi_core` / `phi_gpu` 可通过。

2. **打开或覆盖更可能触发 cuDNN 头的目标，记录实际失败文件。**
   - 重点是 `WITH_CUDNN_FRONTEND=ON`、cuDNN fusion、LSTM、conv cudnn helper 等路径。
   - 不要预先猜测所有文件。

3. **如果失败点集中在少数源码，优先 source file 级别加 `COMPILE_DEFINITIONS USE_CUDA`。**
   - 这是真正最小作用域。

4. **如果失败点通过公共头扩散到大量 `phi_core` / `phi_gpu` 源码，再考虑 target 级别定义。**
   - 这比 `add_definitions()` 更可控，但仍不算最小。

5. **不要新增 Paddle CMake 兼容概念。**
   - 不建议引入 `PADDLE_WITH_CUDA_COMPAT_LAYER`。
   - 不建议把 xtrans 检测逻辑扩散到 Paddle 主 CMake 语义中。

## 建议的下一步验证

为了判断能否完全移除或局部化 `USE_CUDA`，建议按以下顺序继续验证：

1. 保持移除全局 `add_definitions(-DUSE_CUDA)`。
2. 重新配置 `WITH_CUDNN_FRONTEND=ON` 的 build，或者在当前 build 中找到可单独启用 cuDNN frontend 的最小目标。
3. 构建以下类别目标：
   - `cudnn_frontend_test`；
   - 包含 `paddle/phi/backends/dynload/cudnn_frontend.h` 的目标；
   - cuDNN fusion / LSTM / conv frontend 相关 CUDA kernel。
4. 如果失败，记录第一个失败编译单元和具体错误。
5. 只对该失败编译单元尝试 source file 级 `COMPILE_DEFINITIONS USE_CUDA`。
6. 重复直到 targeted build 通过。
7. 如果失败面过大，再退而求其次考虑 `phi_core` / `phi_gpu` target 级定义。

## 当前建议写入最小化审查的结论

`-DUSE_CUDA` 不应作为默认最小 Paddle functional patch 的全局保留项。

更准确的表述应是：

- xtrans 当前 cuDNN 头确实存在 `USE_CUDA` 控制的 CUDA/XPU patch 分支。
- 不定义 `USE_CUDA` 时，`xpudnn_cuda_patch.h` 会污染 CUDA runtime 类型，这是实际问题。
- 但当前 `add_definitions(-DUSE_CUDA)` 作用域过大；它解决的是 xtrans 头文件策略问题，却扩散到了 Paddle 全局 GPU 构建。
- 最小化路径应优先推动 xtrans 修头；Paddle 侧如果短期兜底，应从 source file 级或 target 级局部定义开始，而不是 `WITH_GPU` 全局定义。
