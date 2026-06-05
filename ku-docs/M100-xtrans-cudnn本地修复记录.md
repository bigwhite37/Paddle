# M100 xtrans cuDNN 本地修复记录

## 目标

从本记录创建开始，记录为验证“将 cuDNN 兼容问题修在 xtrans 侧，并撤销 Paddle `cudnn.h` workaround”所做的全部改动和验证结果。

## 当前计划

1. 修改本地 xtrans SDK 的 cuDNN name map。
2. 撤销 `paddle/phi/backends/dynload/cudnn.h` 中针对 xtrans 的本地 workaround。
3. 运行 targeted 构建验证，确认 Paddle 仍能编译通过。

## 初始观察

本地 xtrans SDK 路径：

```text
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars
```

关键头文件可写：

```text
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/name_maps/nm_xpudnn_ops_infer.h
```

已确认的问题：

1. xtrans name map 中存在函数式宏：

```cpp
#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
```

这个宏在普通调用场景可以工作，但会干扰 Paddle dynload 中的 `DECLARE_TYPE(__name, args...)` 展开。

2. xtrans name map 中缺少：

```cpp
#define cudnnGetActivationDescriptor xpudnnGetActivationDescriptor
```

但 xtrans 的 xpudnn 头中存在 `xpudnnGetActivationDescriptor` 声明。

## xtrans 修改必要性

这次验证关注两类问题：

1. 编译期 name map 问题。
2. 运行期 dynload 符号问题。

xtrans 当前提供的是“通过头文件宏把 `cudnn*` API 改写成 `xpudnn*` API”的兼容方式。这个机制对普通源码直接调用有效，但 Paddle 的 `paddle/phi/backends/dynload/cudnn.h` 不是普通直接调用，它会生成动态加载 wrapper，并在运行期通过 `dlsym` 查找符号。

因此 xtrans 侧至少需要修两个编译期问题：

### 1. `cudnnGetActivationDescriptor` 映射缺失

xtrans name map 中已有大量类似映射：

```cpp
#define cudnnCreateActivationDescriptor xpudnnCreateActivationDescriptor
#define cudnnSetActivationDescriptor xpudnnSetActivationDescriptor
#define cudnnDestroyActivationDescriptor xpudnnDestroyActivationDescriptor
```

但缺少：

```cpp
#define cudnnGetActivationDescriptor xpudnnGetActivationDescriptor
```

这会导致源码中使用 `cudnnGetActivationDescriptor` 时，无法被 xtrans 头文件改写到实际存在的 `xpudnnGetActivationDescriptor`。

### 2. `cudnnGetStream` 是函数式宏，会破坏 Paddle dynload 宏展开

当前 xtrans 写法是：

```cpp
#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
```

普通调用 `cudnnGetStream(handle, &stream)` 时，这种写法可以工作。但 Paddle dynload wrapper 会在类型推导里生成类似：

```cpp
decltype(cudnnGetStream(args...))
```

此时 `args...` 对预处理器不是两个明确的宏参数，函数式宏会报“参数数量不足”，导致编译失败。

所以，如果目标是让 Paddle `cudnn.h` 不再保留：

```cpp
#ifdef cudnnGetStream
#undef cudnnGetStream
#endif
```

那 xtrans 不能继续把 `cudnnGetStream` 定义成函数式宏；需要改成不破坏函数名/符号名使用的兼容方式，或者在 xtrans 侧提供真实 `cudnnGetStream` ABI wrapper。

### 3. 只改头文件还不够解决运行期 dynload

即使修了 name map，Paddle dynload 还有运行期问题：Paddle 运行时会通过 `dlsym(handle, "cudnn...")` 查找符号，而当前 xtrans `libcudnn.so` 只导出 `xpudnn*` 符号，不导出 `cudnn*` 符号。

所以完整的 xtrans 侧修复应包含：

- 头文件层：补齐缺失 name map，避免函数式宏破坏 Paddle dynload 编译。
- 动态库层：提供 `cudnn*` ABI alias/wrapper，或让兼容层支持 Paddle 这类 dynload 场景。

## 预处理验证

使用最小预处理实验确认：函数式 `cudnnGetStream(handle, stream)` 宏会在 Paddle dynload 的 `DECLARE_TYPE(cudnnGetStream, args...)` 场景触发宏实参数量错误。

结论：如果希望 Paddle `cudnn.h` 不保留 `#undef cudnnGetStream` workaround，xtrans 侧需要把 `cudnnGetStream` 改成不干扰函数名/符号名使用的形式。

## 后续改动记录

### 2026-05-20：修改本地 xtrans name map

文件：

```text
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/name_maps/nm_xpudnn_ops_infer.h
```

改动 1：将 `cudnnGetStream` 从函数式宏改为对象式映射。

```diff
-#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
+#define cudnnGetStream xpudnnGetStream
```

目的：避免 Paddle dynload 的 `DECLARE_TYPE(cudnnGetStream, args...)` 触发函数式宏实参数量错误。

风险：原函数式宏显式把 stream 参数转换为 `XPUStream*`，对象式映射不再做这个 cast。需要通过实际编译验证判断当前 include/type mapping 下是否仍兼容。

改动 2：补齐 `cudnnGetActivationDescriptor` 映射。

```diff
 #define cudnnCreateActivationDescriptor xpudnnCreateActivationDescriptor
 #define cudnnSetActivationDescriptor xpudnnSetActivationDescriptor
+#define cudnnGetActivationDescriptor xpudnnGetActivationDescriptor
 #define cudnnGetActivationDescriptorSwishBeta xpudnnGetActivationDescriptorSwishBeta
 #define cudnnDestroyActivationDescriptor xpudnnDestroyActivationDescriptor
```

目的：把缺失的 cuDNN API 映射补到 xtrans name map 中，而不是在 Paddle `cudnn.h` 中补 workaround。

### 2026-05-20：撤销 Paddle cudnn.h workaround

文件：

```text
paddle/phi/backends/dynload/cudnn.h
```

移除内容：

```cpp
#ifndef cudnnGetActivationDescriptor
#define cudnnGetActivationDescriptor xpudnnGetActivationDescriptor
#endif

#ifdef cudnnGetStream
#undef cudnnGetStream
#endif
```

目的：验证 xtrans 修复后，Paddle 侧是否可以恢复为不包含 xtrans/xpudnn 特殊处理的版本。

## 验证记录

### 2026-05-20：targeted 构建验证

命令 1：

```bash
cmake --build build --target phi_core -- -j$(nproc)
```

结果：通过。

关键输出：

```text
[ 13%] Linking CXX shared library libphi_core.so
[100%] Built target phi_core
```

说明：该目标重新编译了包含 `paddle/phi/backends/dynload/cudnn.h` 路径的 CUDA 编译单元，例如：

```text
Building CUDA object paddle/phi/CMakeFiles/phi_core.dir/kernels/funcs/math_function.cu.o
```

命令 2：

```bash
cmake --build build --target phi_gpu -- -j$(nproc)
```

结果：通过。

关键输出：

```text
[ 74%] Linking CXX shared library libphi_gpu.so
[100%] Built target phi_gpu
```

说明：当前 build 配置中 `WITH_CUDNN_FRONTEND=OFF`，因此这次 targeted build 没有覆盖 `PADDLE_WITH_CUDNN_FRONTEND` 分支下的 `cudnnGetStream` wrapper；但已覆盖当前默认构建会编译到的 `cudnn.h` dynload 路径，并确认撤销 Paddle workaround 后 `phi_core` / `phi_gpu` 可编译通过。

### 2026-05-20：未修改 xtrans 时的失败输出

为确认 xtrans 修改的必要性，先将本地 xtrans name map 恢复到原始状态：

```cpp
#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
```

且不包含：

```cpp
#define cudnnGetActivationDescriptor xpudnnGetActivationDescriptor
```

然后使用同一个验证程序重新编译。

编译命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -DUSE_CUDA \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans_unpatched
```

编译结果：失败。

关键输出：

```text
ku-docs/tests/test_cudnn_dynload_xtrans.cu:64:33: error: use of undeclared identifier 'cudnnGetActivationDescriptor'; did you mean 'xpudnnGetActivationDescriptor'?
   64 |   do { xpudnnStatus_t status = (cudnnGetActivationDescriptor( activation, &mode, &relu_nan_opt, &relu_ceiling)); if (status != XPUDNN_STATUS_SUCCESS) { std::cerr << "cudnnGetActivationDescriptor( activation, &mode, &relu_nan_opt, &relu_ceiling)" << " failed: " << xpudnnGetErrorString(status) << std::endl; return 1; } } while (0);
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                 xpudnnGetActivationDescriptor
xtrans_cuda_11.7_ubuntu2004_x86_64_mars/include/xpudnn/xpudnn_ops_infer.h:599:55: note: 'xpudnnGetActivationDescriptor' declared here
  599 | xpudnnStatus_t __attribute__((visibility("default"))) xpudnnGetActivationDescriptor(const xpudnnActivationDescriptor_t activationDesc,
      |                                                       ^
1 error generated when compiling for xcn.
```

这说明：不修改 xtrans 时，`cudnnGetActivationDescriptor` 这个 cuDNN API 没有被 name map 改写到 `xpudnnGetActivationDescriptor`，源码层面就无法编译。

另外，用最小 dynload 宏展开实验验证未修改的 `cudnnGetStream` 函数式宏：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/clang++ \
  -std=c++17 \
  -DUSE_CUDA \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -fsyntax-only /tmp/cudnn_stream_dynload_unpatched.*.cc
```

关键输出：

```text
/tmp/cudnn_stream_dynload_unpatched.cc:4:1: error: too few arguments provided to function-like macro invocation
DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(cudnnGetStream)
^
/tmp/cudnn_stream_dynload_unpatched.cc:3:137: note: expanded from macro 'DECLARE_DYNAMIC_LOAD_CUDNN_WRAP'
#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name) struct DynLoad__##__name { template <typename... Args> auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { return __name(args...); } };
                                                                                                                                        ^
/tmp/cudnn_stream_dynload_unpatched.cc:2:62: note: expanded from macro 'DECLARE_TYPE'
#define DECLARE_TYPE(__name, ...) decltype(__name(__VA_ARGS__))
                                                             ^
xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/name_maps/nm_xpudnn_ops_infer.h:12:9: note: macro 'cudnnGetStream' defined here
#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
        ^
```

这说明：未修改 xtrans 时，`cudnnGetStream` 的函数式宏会破坏 Paddle dynload 风格的 `DECLARE_TYPE(cudnnGetStream, args...)` 展开。

结论：xtrans 头文件修改是必要的；否则验证程序至少会在 `cudnnGetActivationDescriptor` 编译处失败，Paddle dynload 风格代码还会在 `cudnnGetStream` 宏展开处失败。

### 2026-05-20：运行时验证程序

新增验证程序：

```text
ku-docs/tests/test_cudnn_dynload_xtrans.cu
```

验证目标：

1. 在 xtrans/M100 环境中实际调用 `cudnnCreate`、`cudnnSetStream`、`cudnnGetStream`、`cudnnCreateActivationDescriptor`、`cudnnSetActivationDescriptor`、`cudnnGetActivationDescriptor` 等路径。
2. 检查 `libcudnn.so` 中是否存在 Paddle dynload 会通过 `dlsym` 查找的 `cudnn*` 动态符号。

编译命令：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 \
  -DUSE_CUDA \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64 \
  -lcudnn -lcudart -ldl \
  ku-docs/tests/test_cudnn_dynload_xtrans.cu \
  -o /tmp/test_cudnn_dynload_xtrans
```

说明：需要 `-DUSE_CUDA` 来匹配 Paddle 当前 CMake 编译环境；不加时 xtrans 的 `xpudnn_cuda_patch.h` 会把 `cudaStream_t` 替换为 `XPUStream`，导致 `cudaStreamCreate` / `cudaStreamDestroy` 编译失败。

运行命令：

```bash
LD_LIBRARY_PATH=/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib64:/home/shuzhenyi/code/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib:$LD_LIBRARY_PATH \
  /tmp/test_cudnn_dynload_xtrans
```

运行结果：

```text
cudnn version: 8907
dlsym(xpudnnGetActivationDescriptor): FOUND
dlsym(xpudnnGetStream): FOUND
dlsym(cudnnGetActivationDescriptor): MISSING
direct xtrans cudnn API path: PASS
xpudnn symbol path: PASS
cudnn symbol path for Paddle dlsym: FAIL
```

结论：

- 头文件宏映射后的直接 xtrans API 调用路径可以运行通过。
- xtrans `libcudnn.so` 当前只导出 `xpudnn*` 符号，没有导出 `cudnnGetActivationDescriptor` / `cudnnGetStream` 这类 `cudnn*` 符号。
- Paddle dynload wrapper 使用 `dlsym(handle, "cudnn...")` 这种符号名查找方式，因此仅修改 xtrans 头文件 name map 还不够；如果运行时真的走 Paddle cuDNN dynload wrapper，会查不到对应 `cudnn*` 符号。

补充符号表验证：

```text
xpudnnGetActivationDescriptor: FOUND
xpudnnGetStream: FOUND
cudnnGetActivationDescriptor: MISSING
cudnnGetStream: MISSING
```

因此，“完全撤销 Paddle `cudnn.h` workaround 并只改 xtrans 头文件”目前只能证明默认构建可编译，不能证明 Paddle dynload 运行时路径可用。要彻底修在 xtrans 侧，需要 xtrans 的 `libcudnn.so` 提供 `cudnn*` ABI alias，或者 Paddle dynload 需要按宏展开后的 `xpudnn*` 符号名做 `dlsym`。
