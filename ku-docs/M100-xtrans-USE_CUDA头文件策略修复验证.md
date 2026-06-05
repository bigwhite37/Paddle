# M100 xtrans `USE_CUDA` 头文件策略修复验证

## 目标

验证并修复 xtrans cuDNN public 头文件依赖上层工程全局定义 `USE_CUDA` 的问题。

当前待验证假设：Paddle 不应该通过 `cmake/configure.cmake` 全局 `add_definitions(-DUSE_CUDA)` 来规避 xtrans cuDNN 头文件污染；更合理的修复点在 xtrans 侧，使 `cudnn_api/cudnn.h` 默认以 CUDA-compatible 方式展开，不默认包含会污染 CUDA runtime 类型的 `xpudnn_cuda_patch.h`。

## 修改前状态

当前 xtrans 头文件：

```cpp
// targets/x86_64-linux/include/cudnn_api/cudnn.h
#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"

#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

```cpp
// targets/x86_64-linux/include/xpudnn/xpudnn.h
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

问题：只要上层没有定义 `USE_CUDA`，xtrans `cudnn_api/cudnn.h` 就会包含 `xpudnn_cuda_patch.h`，其中的宏会把 CUDA runtime 类型替换成 XPU 类型。

## 修改前最小复现

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  /home/shuzhenyi/code/m100/Paddle/ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda_before
```

结果：失败，符合预期。

关键错误：

```text
error: no matching function for call to 'cudaStreamCreate'
candidate function not viable: no known conversion from 'XPUStream *' (aka 'void **') to 'cudaStream_t *' (aka 'CUstream_st **') for 1st argument

error: no matching function for call to 'cudaStreamDestroy'
candidate function not viable: cannot convert argument of incomplete type 'XPUStream' (aka 'void *') to 'cudaStream_t' (aka 'CUstream_st *') for 1st argument
```

判断：复现证明当前失败不是 Paddle CMake 自身问题，而是 xtrans `cudnn_api/cudnn.h` 在未定义 `USE_CUDA` 时启用了 `xpudnn_cuda_patch.h`，导致 `cudaStream_t` 宏污染。

## 修改记录

### 已撤销方案：新增 `XTRANS_CUDNN_CUDA_COMPAT` 内部宏

曾短暂尝试新增 `XTRANS_CUDNN_CUDA_COMPAT` / `XTRANS_ENABLE_XPUDNN_CUDA_PATCH`，让 `xpudnn/xpudnn.h` 识别一个新的 xtrans CUDA-compatible 模式。

该方案已撤销，原因是它引入了新的 xtrans 头文件模式宏，改动面和语义都偏大，不符合当前“不要新增 macro、尽量最小化”的约束。

撤销后，`xpudnn/xpudnn.h` 恢复为原始逻辑：

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

### 修改 1：`cudnn_api/cudnn.h` 直接固定 public cuDNN 入口为 CUDA-style 展开

文件：

```text
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/cudnn.h
```

修改前：

```cpp
#pragma once

#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"

#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

修改后：

```cpp
#pragma once

#define USE_CUDA

#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"
```

目的：xtrans public `cudnn.h` 本身就是给 CUDA-compatible 上层工程包含的 cuDNN 入口，因此在这个入口内部直接保证现有 `USE_CUDA` 分支成立，避免继续包含 `xpudnn_cuda_patch.h`。这个方案没有新增任何新 macro，也没有修改 `xpudnn/xpudnn.h` 的既有分支语义。

## 快速验证记录

### 验证 1：不加 `-DUSE_CUDA` 编译最小验证程序

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  /home/shuzhenyi/code/m100/Paddle/ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda_after
```

结果：通过编译，无输出。

判断：同一个最小程序在修改前不加 `-DUSE_CUDA` 会因为 `cudaStream_t` 被替换成 `XPUStream` 而编译失败；修改后不加 `-DUSE_CUDA` 已可通过编译，说明头文件层面的 `xpudnn_cuda_patch.h` 默认污染已被消除。

### 验证 2：运行最小验证程序

命令：

```bash
LD_LIBRARY_PATH=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64:/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib:${LD_LIBRARY_PATH} \
  /tmp/test_cudnn_dynload_xtrans_no_use_cuda_after
```

结果：失败退出，exit code 2。

关键输出：

```text
cudnn version: 8907
dlsym(xpudnnGetActivationDescriptor): FOUND
dlsym(xpudnnGetStream): FOUND
dlsym(cudnnGetActivationDescriptor): MISSING
direct xtrans cudnn API path: PASS
xpudnn symbol path: PASS
cudnn symbol path for Paddle dlsym: FAIL
```

判断：本次 xtrans 头文件策略修改解决的是编译期 `USE_CUDA` / `xpudnn_cuda_patch.h` 污染问题；运行时 Paddle dynload 需要的 `cudnn*` ABI 符号仍然缺失，仍需 xtrans `libcudnn.so` 或兼容 shim 补 `cudnn* -> xpudnn*` wrapper / alias。

## Paddle targeted build 验证

待补充。

## 全量验证

待补充。
