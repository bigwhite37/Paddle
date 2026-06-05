# cuBLAS 逐 api 验证

## 结论

当前 `tools/m100/verify_cublas_apis.cu` 覆盖 `paddle/phi/backends/dynload/cublas.h` 中 69 个已注册 cuBLAS API。使用预编译 xtrans SDK：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars
```

在 M100 simulator 下复现结果为：

```text
Results: 58/69 passed
```

剩余 11 个失败不是 Paddle 调用层问题，也不是缺少导出符号。按根因分为三类：

| API | 归因 | 最新状态 |
| --- | --- | --- |
| `cublasHgemm` | `libcublas.so` 兼容 wrapper 默认走 `xblas_batched_gemm` 分支，该分支返回 `CUBLAS_STATUS_INVALID_VALUE`；直接调用 `xcnblas_hgemm` 后端可正确计算 | `ENABLE_XBLAS=false ENABLE_XBLAS_F32=false` 下 fresh verification 已 PASS |
| `cublasHgemmStridedBatched` | 同上，默认走 `xblas_batched_gemm` 分支失败；关闭 `ENABLE_XBLAS` 后进入 `xcnblas_hgemm_strided_batched` 并通过 | `ENABLE_XBLAS=false ENABLE_XBLAS_F32=false` 下 fresh verification 已 PASS |
| `cublasSgemmEx` | 当前 verifier 的 half 输入/输出场景默认同样进入 `xblas_batched_gemm`，在 `xblas_gemm.cpp:372` 返回 `CUBLAS_STATUS_INVALID_VALUE`；关闭 `ENABLE_XBLAS` 后 wrapper 没有 `xcnblas` fallback，直接返回 `CUBLAS_STATUS_NOT_SUPPORTED` | 默认 FAIL；`ENABLE_XBLAS=false ENABLE_XBLAS_F32=false` 下仍 FAIL（错误由 7 变 15） |
| `cublasGemmEx` | 当前 F32 verifier 输入绕过 xblas，wrapper 在 `cublas.cpp` 中调用 `xcnblas_gemm_ex`，并把 cuBLAS 单 `C` 同时映射为 xcnblas 的 `c`/`d` | 源码级归因完成；后端返回 success 但不写输出，仍 FAIL |
| `cublasGemmBatchedEx` | wrapper 在 `cublas.cpp` 中直接调用 `xcnblas_gemm_batched_ex`，同样把 `C` array 同时映射为 xcnblas 的 `c`/`d` array | 源码级归因完成；后端返回 success 但不写输出，仍 FAIL |
| `cublasGemmStridedBatchedEx` | 当前 F32 verifier 输入绕过 xblas，wrapper 在 `cublas.cpp` 中调用 `xcnblas_gemm_strided_batched_ex`，并把 `C` 同时映射为 `c`/`d` | 源码级归因完成；后端返回 success 但不写输出，仍 FAIL |
| `cublasGemmEx_64` | `cublas.cpp` 源码直接 `return CUBLAS_STATUS_SUCCESS`；`libcublas.so` 中也是 `xor eax,eax; ret` 空桩 | 源码级归因完成；空桩不启动 kernel、不写输出，仍 FAIL |
| `cublasSgemmEx_64` | `cublas.cpp` 源码直接 `return CUBLAS_STATUS_SUCCESS`；`libcublas.so` 中也是 `xor eax,eax; ret` 空桩 | 源码级归因完成；空桩不启动 kernel、不写输出，仍 FAIL |
| `cublasGemmStridedBatchedEx_64` | `cublas.cpp` 源码直接 `return CUBLAS_STATUS_SUCCESS`；`libcublas.so` 中也是 `xor eax,eax; ret` 空桩 | 源码级归因完成；空桩不启动 kernel、不写输出，仍 FAIL |
| `cublasSetMathMode` | `cublas.cpp` 源码只调用 `xcn_xblas_set_math_mode(handle, mode)` 后返回 success；bridge 仅把 mode 写到伴生 xblas handle | 源码级归因完成；没有 cuBLAS 可读状态闭环，仍 FAIL |
| `cublasGetMathMode` | `cublas.cpp` 源码直接 `return CUBLAS_STATUS_SUCCESS`，不写 `mode` 输出指针；二进制也是 `xor eax,eax; ret` 空实现 | 源码级归因完成；返回 success 但不写输出，仍 FAIL |

因此，当前 Paddle 侧不应把这些失败简单归因为“验证器参数错误”或“symbol 不存在”。更深层问题在 xtrans 的 `libcublas.so` 兼容层与 `libxcnblas.so.0` Ex 后端实现。

## 复现环境

### 编译验证器

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 -arch=sm_80 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib \
  tools/m100/verify_cublas_apis.cu \
  -lcublas -lcudart \
  -o /tmp/verify_cublas_apis
```

### 运行验证器

```bash
XPU_SIMULATOR_MODE=1 \
CUDA_AMODEL_DLL=/home/shuzhenyi/code/m100/xse_simulator/output/xse-ubuntu_2004_x86_64/so/libxpusim.so \
CUDA_AMODEL_GPU=KL004 \
XPUSIM_DEVICE_MODEL=XCN \
LD_LIBRARY_PATH=/home/shuzhenyi/code/m100/xse_simulator/output/xse-ubuntu_2004_x86_64/so:/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib:/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/lib64:/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/lib:${LD_LIBRARY_PATH:-} \
/tmp/verify_cublas_apis
```

失败列表：

```text
[FAIL] cublasHgemm
[FAIL] cublasSgemmEx
[FAIL] cublasGemmEx
[FAIL] cublasHgemmStridedBatched
[FAIL] cublasSetMathMode
[FAIL] cublasGetMathMode
[FAIL] cublasGemmBatchedEx
[FAIL] cublasGemmStridedBatchedEx
[FAIL] cublasGemmStridedBatchedEx_64
[FAIL] cublasGemmEx_64
[FAIL] cublasSgemmEx_64
Results: 58/69 passed
```

### 实际加载库

`ldd /tmp/verify_cublas_apis` 在当前 `LD_LIBRARY_PATH` 下确认加载的是 xtrans 目标库：

```text
libcublas.so.11 => /home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libcublas.so.11
libcudart.so.12 => /home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libcudart.so.12
libxpucuda.so.1 => /home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libxpucuda.so.1
libxcnblas.so.0 => /home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libxcnblas.so.0
libxcnsolver.so.0 => /home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libxcnsolver.so.0
```

## 排查方法

这次按三层拆开验证：

1. **cuBLAS API 层**：运行 `tools/m100/verify_cublas_apis.cu`，确认哪些 API 在 CUDA/cuBLAS 兼容接口上失败。
2. **cuBLAS wrapper 层**：对 `libcublas.so.11` 反汇编，并用 `LD_PRELOAD` 拦截 `xcnblas_*` 后端调用，确认 wrapper 是否进入后端以及传参情况。
3. **xcnblas 后端层**：绕过 `libcublas.so`，直接调用 `xcnblas_*`，判断后端本身是否能计算并写输出。

## half GEMM 族：wrapper 默认分支问题

### 现象

默认环境下：

```text
[FAIL] cublasHgemm
[FAIL] cublasHgemmStridedBatched
[FAIL] cublasSgemmEx
```

其中 `cublasSgemmEx` 在 verifier 中使用的是 `CUDA_R_16F` 输入/输出、float 标量、`CUBLAS_COMPUTE_32F` 语义；它和 `cublasHgemm` 默认会落到同一个 `xblas_batched_gemm` 失败点。但 `cublasHgemmBatched` 能通过并计算正确。

### 后端直连验证

```c++
// /tmp/probe_xcnblas_backend.cu
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <xcnblas/xcnblas.h>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t status = (expr);                                                \
    if (status != cudaSuccess) {                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(status),       \
                  __FILE__, __LINE__);                                          \
      return false;                                                             \
    }                                                                           \
  } while (0)

static uint16_t half_bits(float value) {
  __half h = __float2half(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

static float half_float(uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(bits));
  return __half2float(h);
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t n) {
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) != cudaSuccess) {
      ptr = nullptr;
    }
  }
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  bool ok() const { return ptr != nullptr; }
  bool copy_from(const T* src, size_t n) {
    return cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool copy_to(T* dst, size_t n) const {
    return cudaMemcpy(dst, ptr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
};

static bool make_half_matrices(std::vector<uint16_t>* A,
                               std::vector<uint16_t>* B,
                               std::vector<uint16_t>* C) {
  constexpr int n = 16;
  A->assign(n * n, half_bits(0.0f));
  B->assign(n * n, half_bits(0.0f));
  C->assign(n * n, half_bits(-7.0f));
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      (*A)[row + col * n] = half_bits(row == col ? 1.0f : 0.0f);
      (*B)[row + col * n] = half_bits(static_cast<float>(row + col + 1));
    }
  }
  return true;
}

static bool check_half_result(const char* name, const std::vector<uint16_t>& got) {
  constexpr int n = 16;
  bool ok = true;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      const float expected = static_cast<float>(row + col + 1);
      const float actual = half_float(got[row + col * n]);
      if (std::fabs(actual - expected) > 0.01f) {
        if (ok) {
          std::printf("%s mismatch first row=%d col=%d got=%g expected=%g\n",
                      name, row, col, actual, expected);
        }
        ok = false;
      }
    }
  }
  std::printf("%s samples c0=%g c17=%g c255=%g\n", name,
              half_float(got[0]), half_float(got[17]), half_float(got[255]));
  return ok;
}

static bool probe_hgemm(xcnblas_handle h) {
  constexpr int n = 16;
  std::vector<uint16_t> A, B, C;
  make_half_matrices(&A, &B, &C);
  DeviceBuffer<uint16_t> dA(A.size()), dB(B.size()), dC(C.size());
  if (!dA.ok() || !dB.ok() || !dC.ok()) return false;
  dA.copy_from(A.data(), A.size());
  dB.copy_from(B.data(), B.size());
  dC.copy_from(C.data(), C.size());
  const uint16_t alpha = half_bits(1.0f);
  const uint16_t beta = half_bits(0.0f);
  xcnblas_status status = xcnblas_hgemm(h,
                                        xcnblas_operation_none,
                                        xcnblas_operation_none,
                                        n,
                                        n,
                                        n,
                                        reinterpret_cast<const xcnblas_half*>(&alpha),
                                        reinterpret_cast<const xcnblas_half*>(dA.ptr),
                                        n,
                                        reinterpret_cast<const xcnblas_half*>(dB.ptr),
                                        n,
                                        reinterpret_cast<const xcnblas_half*>(&beta),
                                        reinterpret_cast<xcnblas_half*>(dC.ptr),
                                        n);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<uint16_t> out(C.size());
  dC.copy_to(out.data(), out.size());
  bool ok = status == xcnblas_status_success && check_half_result("xcnblas_hgemm", out);
  std::printf("xcnblas_hgemm status=%d ok=%d\n", static_cast<int>(status), ok);
  return ok;
}

static bool probe_hgemm_batched(xcnblas_handle h) {
  constexpr int n = 16;
  std::vector<uint16_t> A, B, C;
  make_half_matrices(&A, &B, &C);
  DeviceBuffer<uint16_t> dA(A.size()), dB(B.size()), dC(C.size());
  if (!dA.ok() || !dB.ok() || !dC.ok()) return false;
  dA.copy_from(A.data(), A.size());
  dB.copy_from(B.data(), B.size());
  dC.copy_from(C.data(), C.size());
  const uint16_t alpha = half_bits(1.0f);
  const uint16_t beta = half_bits(0.0f);

  const xcnblas_half* hA[1] = {reinterpret_cast<const xcnblas_half*>(dA.ptr)};
  const xcnblas_half* hB[1] = {reinterpret_cast<const xcnblas_half*>(dB.ptr)};
  xcnblas_half* hC[1] = {reinterpret_cast<xcnblas_half*>(dC.ptr)};
  const xcnblas_half** dAarray = nullptr;
  const xcnblas_half** dBarray = nullptr;
  xcnblas_half** dCarray = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dAarray), sizeof(hA)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dBarray), sizeof(hB)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dCarray), sizeof(hC)));
  CHECK_CUDA(cudaMemcpy(dAarray, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBarray, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dCarray, hC, sizeof(hC), cudaMemcpyHostToDevice));

  xcnblas_status status = xcnblas_hgemm_batched(h,
                                                xcnblas_operation_none,
                                                xcnblas_operation_none,
                                                n,
                                                n,
                                                n,
                                                reinterpret_cast<const xcnblas_half*>(&alpha),
                                                dAarray,
                                                n,
                                                dBarray,
                                                n,
                                                reinterpret_cast<const xcnblas_half*>(&beta),
                                                dCarray,
                                                n,
                                                1);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaFree(dAarray);
  cudaFree(dBarray);
  cudaFree(dCarray);
  std::vector<uint16_t> out(C.size());
  dC.copy_to(out.data(), out.size());
  bool ok = status == xcnblas_status_success && check_half_result("xcnblas_hgemm_batched", out);
  std::printf("xcnblas_hgemm_batched status=%d ok=%d\n", static_cast<int>(status), ok);
  return ok;
}

static bool probe_hgemm_strided_batched(xcnblas_handle h) {
  constexpr int n = 16;
  std::vector<uint16_t> A, B, C;
  make_half_matrices(&A, &B, &C);
  DeviceBuffer<uint16_t> dA(A.size()), dB(B.size()), dC(C.size());
  if (!dA.ok() || !dB.ok() || !dC.ok()) return false;
  dA.copy_from(A.data(), A.size());
  dB.copy_from(B.data(), B.size());
  dC.copy_from(C.data(), C.size());
  const uint16_t alpha = half_bits(1.0f);
  const uint16_t beta = half_bits(0.0f);
  xcnblas_status status = xcnblas_hgemm_strided_batched(
      h,
      xcnblas_operation_none,
      xcnblas_operation_none,
      n,
      n,
      n,
      reinterpret_cast<const xcnblas_half*>(&alpha),
      reinterpret_cast<const xcnblas_half*>(dA.ptr),
      n,
      n * n,
      reinterpret_cast<const xcnblas_half*>(dB.ptr),
      n,
      n * n,
      reinterpret_cast<const xcnblas_half*>(&beta),
      reinterpret_cast<xcnblas_half*>(dC.ptr),
      n,
      n * n,
      1);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<uint16_t> out(C.size());
  dC.copy_to(out.data(), out.size());
  bool ok = status == xcnblas_status_success && check_half_result("xcnblas_hgemm_strided_batched", out);
  std::printf("xcnblas_hgemm_strided_batched status=%d ok=%d\n", static_cast<int>(status), ok);
  return ok;
}

static void make_float_matrices(std::vector<float>* A,
                                std::vector<float>* B,
                                std::vector<float>* C,
                                std::vector<float>* D) {
  constexpr int n = 16;
  A->assign(n * n, 0.0f);
  B->assign(n * n, 0.0f);
  C->assign(n * n, -3.0f);
  D->assign(n * n, -7.0f);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      (*A)[row + col * n] = row == col ? 1.0f : 0.0f;
      (*B)[row + col * n] = static_cast<float>(row + col + 1);
    }
  }
}

static bool check_float_result(const char* name, const std::vector<float>& got) {
  constexpr int n = 16;
  bool ok = true;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      const float expected = static_cast<float>(row + col + 1);
      const float actual = got[row + col * n];
      if (std::fabs(actual - expected) > 0.001f) {
        if (ok) {
          std::printf("%s mismatch first row=%d col=%d got=%g expected=%g\n",
                      name, row, col, actual, expected);
        }
        ok = false;
      }
    }
  }
  std::printf("%s samples c0=%g c17=%g c255=%g\n", name, got[0], got[17], got[255]);
  return ok;
}

static bool probe_gemm_ex_separate_cd(xcnblas_handle h) {
  constexpr int n = 16;
  std::vector<float> A, B, C, D;
  make_float_matrices(&A, &B, &C, &D);
  DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(C.size()), dD(D.size());
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dD.ok()) return false;
  dA.copy_from(A.data(), A.size());
  dB.copy_from(B.data(), B.size());
  dC.copy_from(C.data(), C.size());
  dD.copy_from(D.data(), D.size());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  xcnblas_status status = xcnblas_gemm_ex(h,
                                          xcnblas_operation_none,
                                          xcnblas_operation_none,
                                          n,
                                          n,
                                          n,
                                          &alpha,
                                          dA.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          dB.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          &beta,
                                          dC.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          dD.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          xcnblas_datatype_f32_r,
                                          xcnblas_gemm_algo_standard,
                                          0,
                                          0);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> out(D.size());
  dD.copy_to(out.data(), out.size());
  bool ok = status == xcnblas_status_success && check_float_result("xcnblas_gemm_ex_separate_cd", out);
  std::printf("xcnblas_gemm_ex_separate_cd status=%d ok=%d\n", static_cast<int>(status), ok);
  return ok;
}

static bool probe_gemm_ex_same_cd(xcnblas_handle h) {
  constexpr int n = 16;
  std::vector<float> A, B, C, D;
  make_float_matrices(&A, &B, &C, &D);
  DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(C.size());
  if (!dA.ok() || !dB.ok() || !dC.ok()) return false;
  dA.copy_from(A.data(), A.size());
  dB.copy_from(B.data(), B.size());
  dC.copy_from(C.data(), C.size());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  xcnblas_status status = xcnblas_gemm_ex(h,
                                          xcnblas_operation_none,
                                          xcnblas_operation_none,
                                          n,
                                          n,
                                          n,
                                          &alpha,
                                          dA.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          dB.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          &beta,
                                          dC.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          dC.ptr,
                                          xcnblas_datatype_f32_r,
                                          n,
                                          xcnblas_datatype_f32_r,
                                          xcnblas_gemm_algo_standard,
                                          0,
                                          0);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> out(C.size());
  dC.copy_to(out.data(), out.size());
  bool ok = status == xcnblas_status_success && check_float_result("xcnblas_gemm_ex_same_cd", out);
  std::printf("xcnblas_gemm_ex_same_cd status=%d ok=%d\n", static_cast<int>(status), ok);
  return ok;
}

int main() {
  CHECK_CUDA(cudaSetDevice(0));
  xcnblas_handle h = nullptr;
  xcnblas_status create_status = xcnblas_create_handle(&h);
  std::printf("xcnblas_create_handle status=%d handle=%p\n", static_cast<int>(create_status), h);
  if (create_status != xcnblas_status_success || h == nullptr) return 2;
  xcnblas_set_pointer_mode(h, xcnblas_pointer_mode_host);

  int passed = 0;
  int total = 0;
#define RUN(name)            \
  do {                       \
    ++total;                 \
    bool ok = name(h);       \
    passed += ok ? 1 : 0;    \
  } while (0)
  RUN(probe_hgemm);
  RUN(probe_hgemm_batched);
  RUN(probe_hgemm_strided_batched);
  RUN(probe_gemm_ex_separate_cd);
  RUN(probe_gemm_ex_same_cd);
#undef RUN

  xcnblas_status destroy_status = xcnblas_destroy_handle(h);
  std::printf("xcnblas_destroy_handle status=%d\n", static_cast<int>(destroy_status));
  std::printf("xcnblas backend probe Results: %d/%d passed\n", passed, total);
  return passed == total ? 0 : 1;
}
```

编写 `/tmp/probe_xcnblas_backend.cu`，直接调用：

- `xcnblas_hgemm`
- `xcnblas_hgemm_batched`
- `xcnblas_hgemm_strided_batched`
- `xcnblas_gemm_ex`

编译：

```bash
/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/bin/nvcc \
  -std=c++17 -arch=sm_80 \
  -I/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/include \
  -L/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib \
  /tmp/probe_xcnblas_backend.cu \
  -lxcnblas -lcudart \
  -o /tmp/probe_xcnblas_backend
```

运行结果：

```text
xcnblas_hgemm samples c0=1 c17=3 c255=31
xcnblas_hgemm status=0 ok=1
xcnblas_hgemm_batched samples c0=1 c17=3 c255=31
xcnblas_hgemm_batched status=0 ok=1
xcnblas_hgemm_strided_batched samples c0=1 c17=3 c255=31
xcnblas_hgemm_strided_batched status=0 ok=1
```

说明 half GEMM 后端不是整体不可用。

### wrapper 证据

`/home/shuzhenyi/code/m100/xtrans_src` 中能找到 `cublasHgemm` / `cublasHgemmStridedBatched` 的声明和文档映射，但没有找到 `cublasHgemm` wrapper 实现体、`ENABLE_XBLAS`、`ENABLE_XBLAS_F32` 或 `xblas_batched_gemm`。这些实现位于 `/home/shuzhenyi/code/m100/Xpumath`：

```text
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS/src/xblas_gemm.cpp
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS/src/xblas_gemm.h
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS/src/lazy_xblas.cpp
```

这与预编译 `libcublas.so.11` 的符号和内嵌文件名一致：

```text
0000000000088721 b ENABLE_XBLAS
0000000000088720 b ENABLE_XBLAS_F32
0000000000060890 00000000000010b1 t xblas_batched_gemm
0000000000000000 a xblas_gemm.cpp
```

`libcublas.so.11` 的 `cublasHgemm` wrapper 中存在两条路径：

```asm
cublasHgemm:
  cmp BYTE PTR [ENABLE_XBLAS], 0
  jne xblas_path
  call xcnblas_hgemm@plt

xblas_path:
  push 0x44  # CUBLAS_COMPUTE_32F
  call xblas_batched_gemm
```

`ENABLE_XBLAS` 初始化时通过 `env_option_enabled("ENABLE_XBLAS", true)` 读取环境变量，默认值为 true；`ENABLE_XBLAS_F32` 默认值为 false。因此默认环境下 `cublasHgemm` 会进入 `xblas_batched_gemm`，而不是 `xcnblas_hgemm`。

用 `LD_PRELOAD=/tmp/intercept_xcnblas.so` 拦截完整验证器时，默认失败场景没有拦截到 `xcnblas_hgemm` / `xcnblas_hgemm_strided_batched` 调用，说明 wrapper 没进入 xcnblas half 后端。

打开 `DEBUG_XBLAS=1 __DEBUG_XBLAS=1` 后，`xblas_batched_gemm` 打印出实际映射参数：

```text
use '/home/shuzhenyi/code/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/targets/x86_64-linux/lib/libxpu_blas.so'
#call xblas_gemm:
  trans_a:     CUBLAS_OP_N
  trans_b:     CUBLAS_OP_N
  a_type:      CUDA_R_16F 2
  stride_a:    0
  b_type:      CUDA_R_16F 2
  stride_b:    0
  c_type:      CUDA_R_16F 2
  stride_c:    0
  computeType: CUBLAS_COMPUTE_32F 68
xblasLtMatmulAlgoGetHeuristic failed to get algorithms. num results: 0 status: CUBLAS_STATUS_NOT_SUPPORTED
error at /mnt/ssd3/xtdk/workspace/d1d64436-0d06-4dc0-8905-3d2658bafa50/subpod/baidu/cluster-next/Xpumath/xpuBLAS/src/xblas_gemm.cpp 372, error: 7 CUBLAS_STATUS_INVALID_VALUE
```

`cublasHgemmStridedBatched` 的同一路径仅 stride 从 0 变成 256，类型组合仍是 `CUDA_R_16F` 输入/输出 + `CUBLAS_COMPUTE_32F`。反汇编 `xblas_batched_gemm` 可见该函数读取 `__DEBUG_XBLAS` 后打印这组参数；在 heuristic 结果不满足要求时打印 `xblasLtMatmulAlgoGetHeuristic failed...`，随后在同一错误处理路径打印 `xblas_gemm.cpp:372` 并返回 `CUBLAS_STATUS_INVALID_VALUE`。

`cublasSgemmEx` 在当前 verifier 的 half 输入/输出场景下也会打印完全相同的 xblas 参数并在同一行失败：

```text
#call xblas_gemm:
  trans_a:     CUBLAS_OP_N
  trans_b:     CUBLAS_OP_N
  a_type:      CUDA_R_16F 2
  stride_a:    0
  b_type:      CUDA_R_16F 2
  stride_b:    0
  c_type:      CUDA_R_16F 2
  stride_c:    0
  computeType: CUBLAS_COMPUTE_32F 68
xblasLtMatmulAlgoGetHeuristic failed to get algorithms. num results: 0 status: CUBLAS_STATUS_NOT_SUPPORTED
error at .../Xpumath/xpuBLAS/src/xblas_gemm.cpp 372, error: 7 CUBLAS_STATUS_INVALID_VALUE
cublasSgemmEx status=7 c0=-7
```

因此从默认失败点看，`cublasSgemmEx` 与 `cublasHgemm` 是同一类 wrapper/xblas 问题。

### Xpumath 源码路径

`cublasHgemm` 的默认 dispatch 在 `xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp`：

```cpp
cublasStatus_t cublasHgemm(...) {
    if (ENABLE_XBLAS) {
        return xblas_gemm(handle,
                          transa,
                          transb,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          CUDA_R_16F,
                          lda,
                          B,
                          CUDA_R_16F,
                          ldb,
                          beta,
                          C,
                          CUDA_R_16F,
                          ldc,
                          CUBLAS_COMPUTE_32F);
    }
    return xcnBLASStatusToCUStatus(xcnblas_hgemm(...));
}
```

`cublasHgemmStridedBatched` 同样默认进入 xblas：

```cpp
cublasStatus_t cublasHgemmStridedBatched(...) {
    if (ENABLE_XBLAS) {
        return xblas_batched_gemm(handle,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  CUDA_R_16F,
                                  lda,
                                  bsa,
                                  B,
                                  CUDA_R_16F,
                                  ldb,
                                  bsb,
                                  beta,
                                  C,
                                  CUDA_R_16F,
                                  ldc,
                                  bsc,
                                  batchCount,
                                  CUBLAS_COMPUTE_32F);
    }
    return xcnBLASStatusToCUStatus(xcnblas_hgemm_strided_batched(...));
}
```

`cublasSgemmEx` 的源码位置同样在 `xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp`，但它的 fallback 行为和 `cublasHgemm` 不同：

```cpp
cublasStatus_t cublasSgemmEx(...) {
    if(ENABLE_XBLAS
       && (ENABLE_XBLAS_F32 || (Atype != CUDA_R_32F && Btype != CUDA_R_32F && Ctype != CUDA_R_32F)))
    {
        if(is_xblas_gemm_supported(Atype, Btype, Ctype))
        {
            return xblas_gemm(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              Atype,
                              lda,
                              B,
                              Btype,
                              ldb,
                              beta,
                              C,
                              Ctype,
                              ldc,
                              CUBLAS_COMPUTE_32F);
        }
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}
```

`tools/m100/verify_cublas_apis.cu` 里的 `cublasSgemmEx` 用例传入 `Atype=Btype=Ctype=CUDA_R_16F`，所以默认 `ENABLE_XBLAS=true` 时满足上述条件并进入 `xblas_gemm`。但关闭 `ENABLE_XBLAS` 后，它不会像 `cublasHgemm` 那样调用 `xcnblas_hgemm`，而是直接落到函数末尾返回 `CUBLAS_STATUS_NOT_SUPPORTED`。

`ENABLE_XBLAS` 与 `ENABLE_XBLAS_F32` 在 `xpuBLAS/src/xblas_gemm.cpp` 初始化：

```cpp
bool ENABLE_XBLAS = env_option_enabled("ENABLE_XBLAS", true);
bool ENABLE_XBLAS_F32 = env_option_enabled("ENABLE_XBLAS_F32", false);
```

`xblas_gemm` 在 `xpuBLAS/src/xblas_gemm.h` 中只是把普通 GEMM 包装成 `xblas_batched_gemm(... stride=0, batch_count=1, ...)`。真正执行路径在 `xpuBLAS/src/xblas_gemm.cpp` 的 `xblas_batched_gemm`。

这里有一个关键实现问题：`xblasLtMatmulAlgoGetHeuristic` 调用被注释掉，代码直接把 heuristic 状态设为 `CUBLAS_STATUS_NOT_SUPPORTED`：

```cpp
constexpr int                  requestedAlgoCount = 1;
xblasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
int                            returnedResults = 0;
xblasLtMatmulAlgo_t*           p_algo          = nullptr;
void*                          work_space      = nullptr;
size_t                         work_space_size = 0;

// There are some bug in `xblasLtMatmulAlgoGetHeuristic`
auto get_heuristic_status = CUBLAS_STATUS_NOT_SUPPORTED;
// auto get_heuristic_status = xblas().xblasLtMatmulAlgoGetHeuristic(...);
if(returnedResults < 1 || get_heuristic_status != CUBLAS_STATUS_SUCCESS)
{
    if(__DEBUG_XBLAS)
        fprintf(stderr,
                "xblasLtMatmulAlgoGetHeuristic failed to get algorithms. num results: %d "
                "status: %s\n",
                returnedResults,
                getXblasStatusName(get_heuristic_status));
}
```

由于 `returnedResults` 保持 0、`p_algo` 保持 `nullptr`、workspace 也保持空，后续仍继续调用：

```cpp
status = xblas().xblasLtMatmul(g_lt_handle,
                               matmul_desc,
                               alpha,
                               A,
                               a_desc,
                               B,
                               b_desc,
                               beta,
                               C,
                               c_desc,
                               C,
                               c_desc,
                               p_algo,
                               work_space,
                               work_space_size,
                               stream);
break_if_failed_1(status);
```

`break_if_failed_1(status)` 正好位于 `xblas_gemm.cpp:372`，对应运行日志里的：

```text
error at .../Xpumath/xpuBLAS/src/xblas_gemm.cpp 372, error: 7 CUBLAS_STATUS_INVALID_VALUE
```

关闭 xblas 分支后：

```bash
ENABLE_XBLAS=0 ENABLE_XBLAS_F32=0 ... /tmp/probe_hgemm_pointer
```

拦截到：

```text
INTERCEPT xcnblas_hgemm ... m=16 n=16 k=16 ...
INTERCEPT xcnblas_hgemm return=0
INTERCEPT xcnblas_hgemm_strided_batched ... batch=1
INTERCEPT xcnblas_hgemm_strided_batched return=0
Hgemm pointer_mode=host status=0
HgemmStrided pointer_mode=host status=0
Hgemm pointer_mode=device status=0
HgemmStrided pointer_mode=device status=0
```

完整验证器在关闭 xblas 后：

```text
[PASS] cublasHgemm
[PASS] cublasHgemmStridedBatched
Results: 60/69 passed
```

同一组隔离环境的最小 probe 对比显示：

```text
# 默认 ENABLE_XBLAS=true
cublasHgemm status=7 c0=-7
cublasSgemmEx status=7 c0=-7

# ENABLE_XBLAS=false ENABLE_XBLAS_F32=false
cublasHgemm status=0 c0=1
cublasSgemmEx status=15 c0=-7
```

其中 7 是 `CUBLAS_STATUS_INVALID_VALUE`，来自 `xblas_batched_gemm`；15 是 `CUBLAS_STATUS_NOT_SUPPORTED`，来自 `cublasSgemmEx` wrapper 末尾的显式返回。

2026-05-28 重新用环境变量 `ENABLE_XBLAS=false ENABLE_XBLAS_F32=false` 做 fresh verification，完整 verifier 结果仍是 `60/69 passed`，相对默认环境 `58/69 passed` 多通过的正是这两个 half GEMM API：

```text
[PASS] cublasHgemm
[PASS] cublasHgemmStridedBatched
[FAIL] cublasSgemmEx
[FAIL] cublasGemmEx
[FAIL] cublasSetMathMode
[FAIL] cublasGetMathMode
[FAIL] cublasGemmBatchedEx
[FAIL] cublasGemmStridedBatchedEx
[FAIL] cublasGemmStridedBatchedEx_64
[FAIL] cublasGemmEx_64
[FAIL] cublasSgemmEx_64
Results: 60/69 passed
```

fresh verification 日志保存于：

```text
/tmp/verify_cublas_apis.enable_xblas_false.fresh.log
```

### 归因

`cublasHgemm`、`cublasHgemmStridedBatched` 与当前 verifier 覆盖的 `cublasSgemmEx` half 输入/输出场景，默认失败点都不是 Paddle 调用层问题，也不是 half GEMM kernel 整体不可用。更精确地说，问题发生在 Xpumath 的 cuBLAS 兼容 wrapper/xblas 路径：

1. `xpuBLAS/src/xblas_gemm.cpp` 中 `ENABLE_XBLAS` 默认初始化为 true。
2. `xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp` 中 `cublasHgemm` 和 `cublasHgemmStridedBatched` 在 `ENABLE_XBLAS=true` 时无条件进入 `xblas_gemm` / `xblas_batched_gemm`，并把 half GEMM 映射为 `CUDA_R_16F` 输入/输出、`CUBLAS_COMPUTE_32F` 计算。
3. 同文件中的 `cublasSgemmEx` 在 `ENABLE_XBLAS=true` 且 `Atype=Btype=Ctype=CUDA_R_16F` 时也进入 `xblas_gemm`，所以默认失败与 `cublasHgemm` 同源。
4. `xpuBLAS/src/xblas_gemm.cpp` 中 `xblasLtMatmulAlgoGetHeuristic` 调用被注释掉，`get_heuristic_status` 被强制设为 `CUBLAS_STATUS_NOT_SUPPORTED`，`returnedResults` 保持 0。
5. 代码在没有 heuristic algorithm 的情况下仍继续调用 `xblasLtMatmul`，传入 `p_algo=nullptr`、`work_space=nullptr`、`work_space_size=0`。
6. `xblasLtMatmul` 返回 `CUBLAS_STATUS_INVALID_VALUE`，并在 `xblas_gemm.cpp:372` 被 `break_if_failed_1(status)` 捕获。

但关闭 `ENABLE_XBLAS` 后三者表现分叉：`cublasHgemm` / `cublasHgemmStridedBatched` wrapper 会进入已验证可用的 `xcnblas_hgemm*` 后端并通过；`cublasSgemmEx` wrapper 没有对应的 `xcnblas` fallback，直接返回 `CUBLAS_STATUS_NOT_SUPPORTED`。因此 `cublasSgemmEx` 与 `cublasHgemm` 默认失败属于同一个 xblas 根因，但不能仅靠 `ENABLE_XBLAS=false` 修复。

## Ex GEMM 族：xcnblas Ex 后端返回 success 但不写输出

### xcnblas Ex 后端签名差异

`xcnblas_gemm_ex` 与 cuBLAS `cublasGemmEx` 不同。xcnblas 后端有独立的 C 输入与 D 输出：

```cpp
xcnblas_status xcnblas_gemm_ex(
    xcnblas_handle handle,
    xcnblas_operation transA,
    xcnblas_operation transB,
    xcnblas_int m,
    xcnblas_int n,
    xcnblas_int k,
    const void* alpha,
    const void* a,
    xcnblas_datatype a_type,
    xcnblas_int lda,
    const void* b,
    xcnblas_datatype b_type,
    xcnblas_int ldb,
    const void* beta,
    const void* c,
    xcnblas_datatype c_type,
    xcnblas_int ldc,
    void* d,
    xcnblas_datatype d_type,
    xcnblas_int ldd,
    xcnblas_datatype compute_type,
    xcnblas_gemm_algo algo,
    int32_t solution_index,
    uint32_t flags);
```

对应 batched/strided Ex 后端也有 C/D 分离参数：

```cpp
xcnblas_gemm_batched_ex(... const void* c, ... void* d, ...)
xcnblas_gemm_strided_batched_ex(... const void* c, ... void* d, ...)
```

因此验证时分别测试了 `C != D` 和 `C == D` 两种场景，避免误把 cuBLAS 单 C 指针语义映射成假失败。

### 普通 SGEMM 对照

在同一进程、同一 handle、同一矩阵布局、同一 simulator 环境下，普通后端 `xcnblas_sgemm` 可以正确计算：

```text
xcnblas_sgemm_n16 samples c0=1 c_diag=3 c_last=31 ok=1
xcnblas_sgemm_n16 status=0 ok=1
```

这排除了设备、stream、矩阵列主序、host scalar、基础 GEMM kernel 全部不可用等因素。

### `xcnblas_gemm_ex` 直连结果

`/tmp/probe_xcnblas_ex_variants.cu` 结果：

```text
xcnblas_gemm_ex_f32_n16_algo0_sol0_sepcd status=0 ok=0
xcnblas_gemm_ex_f32_n16_algo0_sol0_samecd status=0 ok=0
xcnblas_gemm_ex_f32_n16_algo1_sol1_sepcd status=0 ok=0
xcnblas_gemm_ex_f32_n32_algo0_sol0_sepcd status=0 ok=0
xcnblas_gemm_ex_f16_n16_compute150 status=0 ok=0
xcnblas_gemm_ex_f16_n16_compute151 status=0 ok=0
xcnblas ex variant Results: 1/7 passed
```

失败样例：

```text
xcnblas_gemm_ex_f32_n16_algo0_sol0_sepcd first mismatch row=0 col=0 got=-9 expected=1
xcnblas_gemm_ex_f32_n16_algo0_sol0_samecd first mismatch row=0 col=0 got=-7 expected=1
```

即函数返回 `xcnblas_status_success`，但输出矩阵仍保持初始化哨兵值。

### batched/strided Ex 后端直连结果

`/tmp/probe_xcnblas_ex_batched_variants.cu` 结果：

```text
xcnblas_gemm_batched_ex_sepcd first mismatch row=0 col=0 got=-7 expected=1
xcnblas_gemm_batched_ex_sepcd status=0 ok=0
xcnblas_gemm_batched_ex_samecd first mismatch row=0 col=0 got=-3 expected=1
xcnblas_gemm_batched_ex_samecd status=0 ok=0
xcnblas_gemm_strided_batched_ex_sepcd first mismatch row=0 col=0 got=-7 expected=1
xcnblas_gemm_strided_batched_ex_sepcd status=0 ok=0
xcnblas_gemm_strided_batched_ex_samecd first mismatch row=0 col=0 got=-3 expected=1
xcnblas_gemm_strided_batched_ex_samecd status=0 ok=0
xcnblas ex batched variant Results: 0/4 passed
```

### wrapper 调用证据

`LD_PRELOAD` 拦截完整验证器时，`cublasGemmEx` 确实进入 `xcnblas_gemm_ex`：

```text
INTERCEPT xcnblas_gemm_ex handle=... transA=111 transB=111 m=2 n=2 k=2 alpha=... A=... aType=151 lda=2 B=... bType=151 ldb=2 beta=... C=... cType=151 ldc=2 D=... dType=151 ldd=2 compute=151 algo=0 solution=0 flags=1
INTERCEPT xcnblas_gemm_ex return=0
[FAIL] cublasGemmEx
```

这里 `aType=bType=cType=dType=compute=151` 对应 F32。当前 verifier 对三个非 `_64` `cublasGemm*Ex` 都传 `CUDA_R_32F`，而 `ENABLE_XBLAS_F32` 默认是 false，因此 wrapper 会绕过 xblas 分支并进入 xcnblas Ex 后端。

### Xpumath wrapper 源码路径

`cublasGemmEx`、`cublasGemmBatchedEx`、`cublasGemmStridedBatchedEx` 的 wrapper 都在：

```text
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp
```

`cublasGemmEx` 的关键分支如下：

```cpp
cublasStatus_t cublasGemmEx(...,
                            void* C,
                            cudaDataType_t c_type,
                            int ldc,
                            cudaDataType_t compute_data_type,
                            cublasGemmAlgo_t algo) {
    if(ENABLE_XBLAS
       && (ENABLE_XBLAS_F32
           || (a_type != CUDA_R_32F && b_type != CUDA_R_32F && c_type != CUDA_R_32F)))
    {
        cublasComputeType_t compute_type{};
        cublasStatus_t status = datatype_to_computetype(compute_data_type, compute_type);
        if(status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
        return xblas_gemm(..., compute_type);
    }

    uint32_t solution_index = 0;
    xcnblas_gemm_flags flags = xcnblas_gemm_flags_none;
    xcnblas_status status = xcnblas_query_int8_layout_flag((xcnblas_handle)handle, &flags);
    if(status != xcnblas_status_success)
        return xcnBLASStatusToCUStatus(status);

    return xcnBLASStatusToCUStatus(xcnblas_gemm_ex((xcnblas_handle)handle,
                                                   cuOperationToHCCOperation(transa),
                                                   cuOperationToHCCOperation(transb),
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   CUDatatypeToXcnblasDatatype(a_type),
                                                   lda,
                                                   B,
                                                   CUDatatypeToXcnblasDatatype(b_type),
                                                   ldb,
                                                   beta,
                                                   C,
                                                   CUDatatypeToXcnblasDatatype(c_type),
                                                   ldc,
                                                   C,
                                                   CUDatatypeToXcnblasDatatype(c_type),
                                                   ldc,
                                                   CUDatatypeToXcnblasDatatype(compute_data_type),
                                                   CUGemmAlgoToXcnblasGemmAlgo(algo),
                                                   solution_index,
                                                   flags));
}
```

注意最后传参中 `C` 同时作为 xcnblas 的输入 `c` 和输出 `d`。这与 xcnblas 文档允许的 `C == D` 语义一致；直连 probe 也分别测过 `C != D` 与 `C == D`，两种情况下后端都没有写输出。

`cublasGemmBatchedEx` 没有 xblas 分支，直接调用 batched Ex 后端：

```cpp
return xcnBLASStatusToCUStatus(
    xcnblas_gemm_batched_ex((xcnblas_handle)handle,
                            cuOperationToHCCOperation(transa),
                            cuOperationToHCCOperation(transb),
                            m,
                            n,
                            k,
                            alpha,
                            (void*)A,
                            CUDatatypeToXcnblasDatatype(a_type),
                            lda,
                            (void*)B,
                            CUDatatypeToXcnblasDatatype(b_type),
                            ldb,
                            beta,
                            (void*)C,
                            CUDatatypeToXcnblasDatatype(c_type),
                            ldc,
                            (void*)C,
                            CUDatatypeToXcnblasDatatype(c_type),
                            ldc,
                            batch_count,
                            CUDatatypeToXcnblasDatatype(compute_type),
                            CUGemmAlgoToXcnblasGemmAlgo(algo),
                            solution_index,
                            flags));
```

`cublasGemmStridedBatchedEx` 与普通 `GemmEx` 类似：默认 F32 verifier 输入会绕过 xblas，进入 `xcnblas_gemm_strided_batched_ex`，并把 cuBLAS 的单 `C` 同时映射到 xcnblas 的 `c` 和 `d`：

```cpp
return xcnBLASStatusToCUStatus(
    xcnblas_gemm_strided_batched_ex((xcnblas_handle)handle,
                                    cuOperationToHCCOperation(transa),
                                    cuOperationToHCCOperation(transb),
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    CUDatatypeToXcnblasDatatype(a_type),
                                    lda,
                                    stride_A,
                                    B,
                                    CUDatatypeToXcnblasDatatype(b_type),
                                    ldb,
                                    stride_B,
                                    beta,
                                    C,
                                    CUDatatypeToXcnblasDatatype(c_type),
                                    ldc,
                                    stride_C,
                                    C,
                                    CUDatatypeToXcnblasDatatype(c_type),
                                    ldc,
                                    stride_C,
                                    batch_count,
                                    CUDatatypeToXcnblasDatatype(compute_type),
                                    CUGemmAlgoToXcnblasGemmAlgo(algo),
                                    solution_index,
                                    flags));
```

### 归因

`cublasGemmEx`、`cublasGemmBatchedEx`、`cublasGemmStridedBatchedEx` 在当前 F32 verifier 输入下不是 xblas heuristic 问题，也不是 Paddle verifier 参数错误。源码和拦截证据显示 wrapper 已经把调用转入 `libxcnblas.so.0` 的 Ex 后端；直连后端也返回 success 但输出矩阵保持哨兵值。因此核心问题在 `xcnblas_gemm_ex` / `xcnblas_gemm_batched_ex` / `xcnblas_gemm_strided_batched_ex` 后端实现：返回成功但没有按 `D = alpha * op(A) * op(B) + beta * C` 写输出。`cublasSgemmEx` 另有 wrapper 分支问题：当前 half 输入/输出场景默认走 `xblas_batched_gemm`，关闭 `ENABLE_XBLAS` 后直接返回 `CUBLAS_STATUS_NOT_SUPPORTED`。

## `_64` Ex 族：libcublas 空桩

`libcublas.so.11` 中相关 `_64` 符号已导出：

```text
000000000005e4f0 T cublasSgemmEx_64
000000000005e500 T cublasGemmEx_64
000000000005e830 T cublasGemmStridedBatchedEx_64
```

当前 Paddle 注册并验证的 Gemm `_64` Ex API 是：

```text
cublasGemmStridedBatchedEx_64
cublasGemmEx_64
cublasSgemmEx_64
```

`cublasGemmBatchedEx_64` 在 Xpumath 源码和二进制中同样是空桩，但当前没有在 `paddle/phi/backends/dynload/cublas.h` 注册，因此不属于这次 69 个 API verifier 的失败项。

### Xpumath `_64` 源码路径

这些 `_64` 实现同样位于：

```text
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp
```

源码中 `cublasSgemmEx_64` 与 `cublasGemmEx_64` 直接返回 success，没有调用 xblas 或 xcnblas：

```cpp
CUBLAS_EXPORT cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int64_t m,
                                              int64_t n,
                                              int64_t k,
                                              const float* alpha,
                                              const void* A,
                                              cudaDataType Atype,
                                              int64_t lda,
                                              const void* B,
                                              cudaDataType Btype,
                                              int64_t ldb,
                                              const float* beta,
                                              void* C,
                                              cudaDataType Ctype,
                                              int64_t ldc)
{
    return CUBLAS_STATUS_SUCCESS;
}

CUBLAS_EXPORT cublasStatus_t cublasGemmEx_64(cublasHandle_t handle,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb,
                                             int64_t m,
                                             int64_t n,
                                             int64_t k,
                                             const void* alpha,
                                             const void* A,
                                             cudaDataType Atype,
                                             int64_t lda,
                                             const void* B,
                                             cudaDataType Btype,
                                             int64_t ldb,
                                             const void* beta,
                                             void* C,
                                             cudaDataType Ctype,
                                             int64_t ldc,
                                             cublasComputeType_t computeType,
                                             cublasGemmAlgo_t algo)
{
    return CUBLAS_STATUS_SUCCESS;
}
```

同文件中 batched/strided `_64` Ex 也是同类空实现：

```cpp
CUBLAS_EXPORT cublasStatus_t cublasGemmBatchedEx_64(...) {
    return CUBLAS_STATUS_SUCCESS;
}

CUBLAS_EXPORT cublasStatus_t cublasGemmStridedBatchedEx_64(...) {
    return CUBLAS_STATUS_SUCCESS;
}
```

预编译 `libcublas.so.11` 与源码一致，反汇编显示这些符号是 `xor eax,eax; ret` 空桩：

```asm
000000000005e4f0 <cublasSgemmEx_64>:
   endbr64
   xor    eax,eax
   ret

000000000005e500 <cublasGemmEx_64>:
   endbr64
   xor    eax,eax
   ret

000000000005e820 <cublasGemmBatchedEx_64>:
   endbr64
   xor    eax,eax
   ret

000000000005e830 <cublasGemmStridedBatchedEx_64>:
   endbr64
   xor    eax,eax
   ret
```

`xor eax,eax; ret` 对外表现为 `CUBLAS_STATUS_SUCCESS`，但不会启动 kernel，也不会写输出。

### 归因

`cublasSgemmEx_64`、`cublasGemmEx_64`、`cublasGemmStridedBatchedEx_64` 属于 `libcublas.so` 已导出但未实现的兼容符号。它们不是 Paddle 问题，也不是 verifier 参数错误。`cublasGemmBatchedEx_64` 虽然同样未实现，但当前未注册进 Paddle 的动态加载列表，不在本轮失败 API 集合内。

## MathMode：Get 是空实现，Set 只写 xblas 内部状态

### 运行现象

`/tmp/probe_math_mode`：

```text
get_initial status=0 mode=-1
set_default status=0
get_after_default status=0 mode=-1
set_tensor status=0
get_after_tensor status=0 mode=-1
```

`Get` 返回 success，但 `mode` 保持哨兵值 `-1`。

### Xpumath 源码路径

`cublasGetMathMode` 与 `cublasSetMathMode` 的 wrapper 实现位于：

```text
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS-xcn-5.5.0/library/src/amd_detail/cublas.cpp
```

源码中 `cublasGetMathMode` 直接返回 success，没有写 `mode` 输出指针：

```cpp
CUBLAS_EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
    return CUBLAS_STATUS_SUCCESS;
}
```

源码中 `cublasSetMathMode` 只调用内部 bridge，然后返回 success：

```cpp
CUBLAS_EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    xcn_xblas_set_math_mode((void*)handle, (int)mode);
    return CUBLAS_STATUS_SUCCESS;
}
```

bridge 实现在：

```text
/home/shuzhenyi/code/m100/Xpumath/xpuBLAS/src/xcn_xblas_bridge.cpp
```

`cublasCreate` 成功创建 xcnblas handle 后，会调用 `xcn_xblas_handle_insert((void*)*handle)` 创建并记录一个伴生 xblas handle；`cublasDestroy` 调用 `xcn_xblas_handle_erase((void*)handle)` 销毁这个伴生 handle。`xcn_xblas_set_math_mode` 只查找这个 map 并调用 `xblasSetMathMode`：

```cpp
void xcn_xblas_set_math_mode(void* xcn_handle, int mode)
{
    std::shared_lock<std::shared_mutex> lock(_xcnhandle_map_mutex);
    auto it = _xcnhandle_to_xblashandle.find(xcn_handle);
    if(it == _xcnhandle_to_xblashandle.end()) return;
    auto lazy = xblas(false);
    if(lazy.xblasSetMathMode)
        lazy.xblasSetMathMode((xblasHandle_t)it->second, (xblasMath_t)mode);
}
```

`lazy_xblas.h` 只声明了 `xblasSetMathMode`，没有声明或加载 `xblasGetMathMode`。在 Xpumath 与 xtrans include 中也没有找到 `xcn_xblas_get_math_mode`、`xcnblas_get_math_mode` 或 `xcnblas_set_math_mode`。因此 `Set` 写入的是内部 xblas 伴生 handle 状态，而 cuBLAS wrapper 没有任何可用于 `Get` 读回的状态存储或后端 API。

### wrapper 机器码

`cublasGetMathMode`：

```asm
000000000005f1d0 <cublasGetMathMode>:
   endbr64
   xor    eax,eax
   ret
```

它没有写第二个参数 `mode`。

`cublasSetMathMode`：

```asm
000000000005f270 <cublasSetMathMode>:
   endbr64
   sub    rsp,0x8
   call   xcn_xblas_set_math_mode@plt
   xor    eax,eax
   add    rsp,0x8
   ret
```

符号归属：

```text
libcublas.so.11:
000000000005f1d0 T cublasGetMathMode
000000000005f270 T cublasSetMathMode
00000000000643d0 T xcn_xblas_set_math_mode
```

`libxcnblas.so.0` 中没有对应 `xcnblas_get_math_mode` / `xcnblas_set_math_mode` 后端状态 API。

### 归因

MathMode 的 cuBLAS 兼容实现不完整，且源码与二进制表现一致：

1. `cublasGetMathMode` 在 `cublas.cpp` 中是空实现，只返回 `CUBLAS_STATUS_SUCCESS`，不写 `mode`。
2. `cublasSetMathMode` 只通过 `xcn_xblas_set_math_mode` 把 mode 转发给伴生 xblas handle 的 `xblasSetMathMode`。
3. 该 bridge 没有对应 `get` 函数，也没有在 cuBLAS wrapper 层保存 math mode 状态。
4. `libxcnblas.so.0` / xcnblas headers 中没有 `xcnblas_get_math_mode` / `xcnblas_set_math_mode` 后端状态 API。

因此验证器要求 `Get` 能读回 `Set` 的值时必然失败；这不是 Paddle 调用层问题，也不是 verifier 参数问题，而是 `libcublas.so` 兼容层 MathMode 状态 API 未完整实现。

## 最小影响判断

1. 这 11 个失败都不是缺符号：相关符号在 `libcublas.so.11` 中可被动态加载。
2. 这 11 个失败也不是统一的一类问题：
   - half GEMM：wrapper 默认分支选择问题；xcnblas half 后端可用。
   - Ex GEMM 非 `_64`：xcnblas Ex 后端或 xblas Ex 路径不满足 cuBLAS 正确性语义。
   - Ex GEMM `_64`：`libcublas.so` 空桩。
   - MathMode：`libcublas.so` 兼容层状态 API 未完整实现。
3. Paddle 侧当前最多只能：
   - 暂不把这些 API 视为“正确可用”；
   - 或在构建/运行环境中显式控制 `ENABLE_XBLAS=0` 规避 half GEMM 两个 wrapper 分支问题；
   - 但 Ex 族与 MathMode 需要 xtrans SDK 修复，Paddle 侧无法通过正常 cuBLAS 调用修复“返回 success 但不写输出”的后端行为。

## 本次产生的临时证据文件

```text
/tmp/verify_cublas_apis.deep.log
/tmp/verify_cublas_apis.intercept.log
/tmp/verify_cublas_apis.disable_xblas.log
/tmp/probe_xcnblas_backend.cu
/tmp/probe_xcnblas_backend.log
/tmp/probe_xcnblas_ex_variants.cu
/tmp/probe_xcnblas_ex_variants.log
/tmp/probe_xcnblas_ex_batched_variants.cu
/tmp/probe_xcnblas_ex_batched_variants.log
/tmp/intercept_xcnblas.cpp
/tmp/intercept_xcnblas.so
/tmp/cublasHgemm.wrapper.asm
/tmp/cublasHgemmStrided.wrapper.asm
/tmp/cublasSgemmEx.wrapper.asm
/tmp/cublas_ex_batched_wrappers.asm
/tmp/cublas_ex64_wrappers.asm
/tmp/cublas_mathmode.wrapper.asm
```

这些文件用于归因取证，不是 Paddle 源码必需文件。
