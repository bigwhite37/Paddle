# M100 xtrans `USE_CUDA` 根因最小化修复记录

## 目标

在不修改 Paddle CMake、不引入新 macro 的前提下，验证并修复 xtrans cuDNN 头文件依赖上层全局 `-DUSE_CUDA` 才能避免 CUDA 类型污染的问题。

约束：

- 不改 Paddle CMake。
- 不引入新的兼容 macro。
- 不用 Paddle 侧 `#undef` / 局部 `#define` workaround。
- 抓根因：修 `xpudnn_cuda_patch.h` 中会污染 CUDA-style 编译环境的结构性问题。
- 修改迭代过程中优先使用快速、可靠的最小编译验证；最后才做全量编译验收。

## 根因判断

`USE_CUDA` 本身不是 Paddle 语义。它目前的作用是让 xtrans cuDNN 头文件避开 `xpudnn_cuda_patch.h`：

```cpp
#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

真正的问题是 `xpudnn_cuda_patch.h` 用预处理器宏改写 CUDA runtime 基础类型、函数和数据类型，例如：

```cpp
#define cudaStream_t XPUStream
#define cudaFree(A) ;
#define cudaMalloc(A,B) ;
#define cudaMemcpy(A,B,C,D) 0
#define __half        float16
#define __nv_bfloat16 bfloat16
```

以及一批带分号的 `CUDA_R_*` 宏。它们会让同一个编译单元中的 CUDA 类型含义前后不一致，或者让内存操作静默变成 no-op。

因此，最小化修复路径不是继续给 Paddle 或 xtrans public 入口补一个宏，而是收敛 `xpudnn_cuda_patch.h` 的危险宏污染。

## 推荐修复路径

1. 先恢复实验性改动：
   - 恢复 xtrans `cudnn_api/cudnn.h`，移除临时加入的 `#define USE_CUDA`。
   - 恢复 Paddle `cmake/configure.cmake` 中当前实验性删除的 `add_definitions(-DUSE_CUDA)`，因为本轮要求不改 CMake。
2. 在 `xpudnn_cuda_patch.h` 中按组移除危险宏：
   - 第一组：`#define cudaStream_t XPUStream`。
   - 第二组：`cudaFree` / `cudaMalloc` / `cudaMemcpy` no-op 宏。
   - 第三组：`__half` / `__nv_bfloat16` 类型替换宏。
   - 第四组：`CUDA_R_*` / `CUDA_C_*` 带分号数据类型宏。
3. 每组修改后使用最小 `.cu` 编译验证，不立即跑全量构建。
4. 只有最小验证和 targeted build 通过后，最后再做全量验收。

## 修改日志

### 0. 恢复实验性改动

已恢复 xtrans `cudnn_api/cudnn.h`，移除临时加入的 `#define USE_CUDA`，恢复为原始 include 结构：

```cpp
#pragma once

#include "xpudnn/xpudnn.h"
#include "cudnn_api/name_maps/name_map.h"

#if !defined(__KL3_XRE__) && !defined(USE_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```

已恢复 Paddle `cmake/configure.cmake` 中的当前实验性删除项：

```cmake
add_definitions(-DUSE_CUDA)
```

说明：本轮根因修复不改 Paddle CMake。恢复该行是为了避免把 CMake 删除混入 xtrans 根因修复验证。

### 1. 删除 `cudaStream_t` 污染宏

文件：

```text
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/xpudnn_cuda_patch.h
```

修改前：

```cpp
#if !defined(__KL3_XRE_)
#define cudaStream_t XPUStream
```

修改后：

```cpp
#if !defined(__KL3_XRE_)
```

理由：红灯基线的唯一编译错误来自 `cudaStream_t` 被宏替换成 `XPUStream`，所以第一步只移除这一行，不同时修改其他宏。

## 验证日志

### 验证 0：修改前最小复现基线

验证源码：

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <dlfcn.h>

#include <iostream>

#define CHECK_CUDA(expr)                                                \
  do {                                                                  \
    cudaError_t status = (expr);                                        \
    if (status != cudaSuccess) {                                        \
      std::cerr << #expr << " failed: " << cudaGetErrorString(status)  \
                << std::endl;                                          \
      return 1;                                                         \
    }                                                                   \
  } while (0)

#define CHECK_CUDNN(expr)                                               \
  do {                                                                  \
    cudnnStatus_t status = (expr);                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << #expr << " failed: " << cudnnGetErrorString(status) \
                << std::endl;                                          \
      return 1;                                                         \
    }                                                                   \
  } while (0)

static bool HasSymbol(void* handle, const char* name) {
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  std::cout << "dlsym(" << name << "): "
            << ((symbol != nullptr && error == nullptr) ? "FOUND" : "MISSING")
            << std::endl;
  return symbol != nullptr && error == nullptr;
}

int main() {
  std::cout << "cudnn version: " << cudnnGetVersion() << std::endl;

  cudaStream_t cuda_stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&cuda_stream));

  cudnnHandle_t handle = nullptr;
  CHECK_CUDNN(cudnnCreate(&handle));

  XPUStream xpu_stream = reinterpret_cast<XPUStream>(cuda_stream);
  CHECK_CUDNN(cudnnSetStream(handle, xpu_stream));

  XPUStream got_stream = nullptr;
  CHECK_CUDNN(cudnnGetStream(handle, &got_stream));
  if (got_stream != xpu_stream) {
    std::cerr << "cudnnGetStream returned a different stream" << std::endl;
    return 1;
  }

  cudnnActivationDescriptor_t activation = nullptr;
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation));
  CHECK_CUDNN(cudnnSetActivationDescriptor(
      activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  cudnnActivationMode_t mode;
  cudnnNanPropagation_t relu_nan_opt;
  double relu_ceiling = -1.0;
  CHECK_CUDNN(cudnnGetActivationDescriptor(
      activation, &mode, &relu_nan_opt, &relu_ceiling));
  if (mode != CUDNN_ACTIVATION_RELU || relu_nan_opt != CUDNN_NOT_PROPAGATE_NAN) {
    std::cerr << "cudnnGetActivationDescriptor returned unexpected attributes"
              << std::endl;
    return 1;
  }

  CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation));
  CHECK_CUDNN(cudnnDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(cuda_stream));

  void* cudnn = dlopen("libcudnn.so", RTLD_LAZY | RTLD_LOCAL);
  if (cudnn == nullptr) {
    std::cerr << "dlopen(libcudnn.so) failed: " << dlerror() << std::endl;
    return 1;
  }

  bool has_xpudnn = HasSymbol(cudnn, "xpudnnGetActivationDescriptor") &&
                    HasSymbol(cudnn, "xpudnnGetStream");
  bool has_cudnn = HasSymbol(cudnn, "cudnnGetActivationDescriptor") &&
                   HasSymbol(cudnn, "cudnnGetStream");

  dlclose(cudnn);

  std::cout << "direct xtrans cudnn API path: PASS" << std::endl;
  std::cout << "xpudnn symbol path: " << (has_xpudnn ? "PASS" : "FAIL")
            << std::endl;
  std::cout << "cudnn symbol path for Paddle dlsym: "
            << (has_cudnn ? "PASS" : "FAIL") << std::endl;

  return has_xpudnn && has_cudnn ? 0 : 2;
}
```

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  /home/shuzhenyi/code/m100/Paddle/ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda_red
```

结果：失败，符合预期。

关键错误：

```text
error: no matching function for call to 'cudaStreamCreate'
candidate function not viable: no known conversion from 'XPUStream *' (aka 'void **') to 'cudaStream_t *' (aka 'CUstream_st **') for 1st argument

error: no matching function for call to 'cudaStreamDestroy'
candidate function not viable: cannot convert argument of incomplete type 'XPUStream' (aka 'void *') to 'cudaStream_t' (aka 'CUstream_st *') for 1st argument
```

判断：当前最小失败由 `xpudnn_cuda_patch.h` 中 `#define cudaStream_t XPUStream` 直接触发。

### 验证 1：删除 `cudaStream_t` 宏后重新编译最小验证程序

命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  /home/shuzhenyi/code/m100/Paddle/ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda_after_stream_patch
```

结果：通过编译，无输出。

判断：只删除 `#define cudaStream_t XPUStream` 后，同一个不加 `-DUSE_CUDA` 的验证程序即可编译通过，证明当前 `USE_CUDA` 编译期问题的直接根因是该宏污染。

### 验证 2：运行最小验证程序，确认运行时边界

命令：

```bash
LD_LIBRARY_PATH=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64:/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib:${LD_LIBRARY_PATH} \
  /tmp/test_cudnn_dynload_xtrans_no_use_cuda_after_stream_patch
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

判断：删除 `cudaStream_t` 宏已解决本轮编译期 `USE_CUDA` 问题；运行时 Paddle dynload 仍缺 `cudnn*` ABI 符号，这是另一个 xtrans `libcudnn.so` alias/wrapper 问题，不属于本轮 `USE_CUDA` 头文件污染修复。

### 2. 删除 `cudaFree` / `cudaMalloc` / `cudaMemcpy` no-op 宏

新增专项验证文件：

```text
ku-docs/use_cuda_macro/test_cuda_memory_patch_xtrans.cu
```

内容：

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>

int main() {
  void* ptr = nullptr;
  cudaError_t malloc_status = cudaMalloc(&ptr, 16);
  cudaError_t memcpy_status = cudaMemcpy(ptr, ptr, 16, cudaMemcpyDeviceToDevice);
  cudaError_t free_status = cudaFree(ptr);
  return static_cast<int>(malloc_status) + static_cast<int>(memcpy_status) +
         static_cast<int>(free_status);
}
```

修改前编译失败，关键错误：

```text
error: expected expression
cudaError_t malloc_status = ;;

error: cannot initialize a variable of type 'cudaError_t' (aka 'cudaError') with an rvalue of type 'int'
cudaError_t memcpy_status = 0;

error: expected expression
cudaError_t free_status = ;;
```

判断：这组三个宏会把合法 CUDA runtime 表达式替换成无效表达式或错误类型，且在其他调用形态中可能造成静默 no-op，属于结构性污染。

删除以下宏：

```cpp
#define cudaFree(A) ;
#define cudaMalloc(A,B) ;
#define cudaMemcpy(A,B,C,D) 0
```

复测命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_memory_patch_xtrans.cu \
  -o /tmp/test_cuda_memory_patch_xtrans_after
```

结果：通过编译，无输出。

### 3. 删除 `__half` / `__nv_bfloat16` 类型替换宏

新增专项验证文件：

```text
ku-docs/use_cuda_macro/test_cuda_half_patch_xtrans.cu
```

内容：

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#ifdef __half
#error "__half is defined as a macro after including cudnn.h"
#endif

#ifdef __nv_bfloat16
#error "__nv_bfloat16 is defined as a macro after including cudnn.h"
#endif

int main() {
  __half h{};
  (void)h;
  return 0;
}
```

修改前编译失败，关键错误：

```text
"__half is defined as a macro after including cudnn.h"
"__nv_bfloat16 is defined as a macro after including cudnn.h"
```

判断：`cudnn.h` 被包含后不应把 CUDA runtime 的 `__half` / `__nv_bfloat16` 改写成 xpu util 类型；这会让同一编译单元中的 CUDA half/bfloat16 类型含义发生全局变化。

删除以下宏：

```cpp
#define __half        float16
#define __nv_bfloat16 bfloat16
```

复测命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_half_patch_xtrans.cu \
  -o /tmp/test_cuda_half_patch_xtrans_after
```

结果：通过编译，无输出。

### 4. 删除 `CUDA_R_*` / `CUDA_C_*` 数据类型常量宏

新增专项验证文件：

```text
ku-docs/use_cuda_macro/test_cuda_datatype_patch_xtrans.cu
```

内容：

```cpp
#include <cuda_runtime.h>
#include <library_types.h>
#include <cudnn.h>

#ifdef CUDA_R_32F
#error "CUDA_R_32F is defined as a macro after including cudnn.h"
#endif

#ifdef CUDA_C_32F
#error "CUDA_C_32F is defined as a macro after including cudnn.h"
#endif

int main() {
  cudaDataType_t dtype = CUDA_R_32F;
  return static_cast<int>(dtype);
}
```

修改前编译失败，关键错误：

```text
error: "CUDA_R_32F is defined as a macro after including cudnn.h"
error: "CUDA_C_32F is defined as a macro after including cudnn.h"
```

判断：`CUDA_R_*` / `CUDA_C_*` 是 `library_types.h` 中的 CUDA data type enum 常量，不应在 `cudnn.h` 之后变成预处理器宏。补丁头里的宏还带分号，会在函数实参、enum 声明等上下文中造成语法破坏或类型含义改变。

删除以下宏组：

```cpp
#define CUDA_R_16F xpudnn_CUDA_R_16F;
#define CUDA_C_16F xpudnn_CUDA_C_16F;
...
#define CUDA_R_8F_E4M3 xpudnn_CUDA_R_8F_E4M3;
#define CUDA_R_8F_E5M2 xpudnn_CUDA_R_8F_E5M2;
```

保留 `xpudnn_cudaDataType_t` enum 和 typedef，不在本轮引入新的数据类型映射策略，避免扩大修复范围。

复测命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_datatype_patch_xtrans.cu \
  -o /tmp/test_cuda_datatype_patch_xtrans_after
```

结果：通过编译，无输出。

## 快速验证汇总

四个不加 `-DUSE_CUDA` 的最小编译验证均已通过：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_no_use_cuda_final

/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_memory_patch_xtrans.cu \
  -o /tmp/test_cuda_memory_patch_xtrans_final

/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_half_patch_xtrans.cu \
  -o /tmp/test_cuda_half_patch_xtrans_final

/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart \
  ku-docs/use_cuda_macro/test_cuda_datatype_patch_xtrans.cu \
  -o /tmp/test_cuda_datatype_patch_xtrans_final
```

结果：四条命令均通过编译，无输出。

运行 `test_cudnn_dynload_xtrans` 的结果仍为已知运行时 ABI 边界问题：

```bash
LD_LIBRARY_PATH=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64:/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib:${LD_LIBRARY_PATH} \
  /tmp/test_cudnn_dynload_xtrans_no_use_cuda_final
```

输出：

```text
cudnn version: 8907
dlsym(xpudnnGetActivationDescriptor): FOUND
dlsym(xpudnnGetStream): FOUND
dlsym(cudnnGetActivationDescriptor): MISSING
direct xtrans cudnn API path: PASS
xpudnn symbol path: PASS
cudnn symbol path for Paddle dlsym: FAIL
```

判断：本轮 `USE_CUDA` 根因修复已经解决头文件编译期污染；`cudnn*` dynload alias/wrapper 缺失仍是 xtrans `libcudnn.so` 的运行时 ABI 问题，需要另行修复。

## Paddle targeted 构建验证

在最小 `.cu` 编译验证全部通过后，继续使用现有 `build/` 目录做 targeted Paddle 构建验证。当前 CMake cache 确认为：

```text
CMAKE_GENERATOR=Unix Makefiles
WITH_GPU=ON
WITH_CUDNN_FRONTEND=OFF
CMAKE_CUDA_COMPILER=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc
CMAKE_CXX_COMPILER=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/clang++
```

### `phi_core`

命令：

```bash
cmake --build build --target phi_core -j$(nproc)
```

结果：通过，输出末尾为：

```text
[100%] Linking CXX shared library libphi_core.so
[100%] Built target phi_core
```

错误关键字检查：

```bash
grep -n "error:\|Error \|FAILED\|Stop\." /tmp/claude-1001/-home-shuzhenyi-code-m100-Paddle/23d1f0ac-b844-442b-b04c-7a07ae6d1d0d/tasks/bl3dtb86c.output | tail -40
```

结果：无输出。

### `phi_gpu`

命令：

```bash
cmake --build build --target phi_gpu -j$(nproc)
```

结果：通过，输出末尾为：

```text
[100%] Linking CXX shared library libphi_gpu.so
[100%] Built target phi_gpu
```

错误关键字检查：

```bash
grep -n "error:\|Error \|FAILED\|Stop\." /tmp/claude-1001/-home-shuzhenyi-code-m100-Paddle/23d1f0ac-b844-442b-b04c-7a07ae6d1d0d/tasks/bg2fvhgmz.output | tail -40
```

结果：无输出。

## 全量 `build_m100.sh` 验收

命令：

```bash
./build_m100.sh
```

结果：脚本 exit code 1，但失败点不是本轮 xtrans 头文件修复导致的编译错误。

已通过的阶段：

```text
[100%] Built target copy_libpaddle
[100%] Built target paddle_python
[  OK] Build complete in 747s (12m 27s)
```

失败阶段：step 6 `Build & Install Wheel`。

关键输出：

```text
[INFO] Packaging wheel (setup.py bdist_wheel)...
[  OK] rpath set on shared libraries
tee: /tmp/wheel_build.log: Permission denied
fatal: no tag exactly matches '847020a4946727b34d7804845d3a012b79c2407e'
running bdist_wheel
...
adding 'paddlepaddle_gpu-3.4.0.dev20260519.dist-info/RECORD'
removing build/bdist.linux-x86_64/wheel
```

定位：`/tmp/wheel_build.log` 是旧的 root-owned 文件，当前用户 `dev2` 无法覆盖，导致脚本中 `setup.py bdist_wheel 2>&1 | tee /tmp/wheel_build.log` 的 pipeline 返回失败；wheel 实际已经生成：

```text
build/python/dist/paddlepaddle_gpu-3.4.0.dev20260519-cp310-cp310-linux_x86_64.whl
```

不修改 `build_m100.sh` 的前提下，继续执行脚本剩余的安装 wheel 和模拟器验证阶段：

```bash
PYTHON_BIN=/root/miniconda/envs/python310_torch25_cuda/bin/python3
SCRIPT_DIR=/home/shuzhenyi/code/m100/Paddle
PARENT_DIR=/home/shuzhenyi/code/m100
XCUDA_PATH=${PARENT_DIR}/xtrans_cuda_11.7_ubuntu2004_x86_64_mars
SIMULATOR_DIR=${PARENT_DIR}/xse_simulator
WHEEL=${SCRIPT_DIR}/build/python/dist/paddlepaddle_gpu-3.4.0.dev20260519-cp310-cp310-linux_x86_64.whl
SIMULATOR_SO=${SIMULATOR_DIR}/output/xse-ubuntu_2004_x86_64/so/libxpusim.so

export PATH=$(dirname "$PYTHON_BIN"):${XCUDA_PATH}/bin:${PATH}
export CUDA_PATH=${XCUDA_PATH}
export LD_LIBRARY_PATH=${XCUDA_PATH}/lib64:${XCUDA_PATH}/lib:${SIMULATOR_DIR}/output/xse-ubuntu_2004_x86_64/so:${LD_LIBRARY_PATH:-}
export LDFLAGS=-L${XCUDA_PATH}/lib64/
export XTRANS_DIR=${XCUDA_PATH}
export CXX=${XCUDA_PATH}/bin/clang++
export CUDNN_ROOT=${XCUDA_PATH}/targets/x86_64-linux
export CUPTI_ROOT=${XCUDA_PATH}/targets/x86_64-linux
export XMLIR_CUDNN_ENABLED=true
export CMAKE_CUDA_ARCHITECTURES=80
export XPU_SIMULATOR_MODE=1
export CUDA_AMODEL_DLL=${SIMULATOR_SO}
export CUDA_AMODEL_GPU=KL004
export XPUSIM_DEVICE_MODEL=XCN

"$PYTHON_BIN" -m pip install --force-reinstall "$WHEEL"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" demo_m100.py
```

结果：通过，输出：

```text
==================================================
M100 Simulator Verification
==================================================
  [device_setup] PASS
  [randn] PASS
  [matmul_gemm] PASS
  [elementwise] PASS
  [reduce] PASS
  [batch_matmul] PASS
  [clone_copy] PASS
==================================================
Results: 7/7 passed
ALL TESTS PASSED
```

结论：本轮 xtrans `USE_CUDA` 头文件污染修复通过了最小编译验证、Paddle targeted 构建验证，以及全量构建主体和模拟器运行验证。`./build_m100.sh` 本身仍有一个环境污染型问题：固定写 `/tmp/wheel_build.log`，遇到旧 root-owned 文件时会误失败。
