#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef CUBLAS_USE_CU_HALF
#define CUBLAS_USE_CU_HALF
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t status = (expr);                                                \
    if (status != cudaSuccess) {                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(status),       \
                  __FILE__, __LINE__);                                          \
      return false;                                                             \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(expr)                                                      \
  do {                                                                          \
    cublasStatus_t status = (expr);                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      std::printf("cuBLAS error %d at %s:%d\n", static_cast<int>(status),       \
                  __FILE__, __LINE__);                                          \
      return false;                                                             \
    }                                                                           \
  } while (0)

struct TestCase {
  const char* name;
  bool (*fn)();
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;

  explicit DeviceBuffer(int n) {
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T));
    if (status != cudaSuccess) ptr = nullptr;
  }

  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }

  bool ok() const { return ptr != nullptr; }

  bool copy_from(const T* src, int n) {
    return cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
  }

  bool copy_to(T* dst, int n) const {
    return cudaMemcpy(dst, ptr, n * sizeof(T), cudaMemcpyDeviceToHost) ==
           cudaSuccess;
  }
};

template <typename T>
static T make_value(double r, double i = 0.0);

template <>
float make_value<float>(double r, double) {
  return static_cast<float>(r);
}

template <>
double make_value<double>(double r, double) {
  return r;
}

template <>
cuFloatComplex make_value<cuFloatComplex>(double r, double i) {
  return make_cuFloatComplex(static_cast<float>(r), static_cast<float>(i));
}

template <>
cuDoubleComplex make_value<cuDoubleComplex>(double r, double i) {
  return make_cuDoubleComplex(r, i);
}

template <>
__half make_value<__half>(double r, double) {
  return __float2half(static_cast<float>(r));
}

static bool nearly_equal(double a, double b, double eps) {
  return std::fabs(a - b) <= eps;
}

template <typename T>
static bool value_equal(T got, T want, double eps) {
  return nearly_equal(static_cast<double>(got), static_cast<double>(want), eps);
}

template <>
bool value_equal<cuFloatComplex>(cuFloatComplex got,
                                 cuFloatComplex want,
                                 double eps) {
  return nearly_equal(cuCrealf(got), cuCrealf(want), eps) &&
         nearly_equal(cuCimagf(got), cuCimagf(want), eps);
}

template <>
bool value_equal<cuDoubleComplex>(cuDoubleComplex got,
                                  cuDoubleComplex want,
                                  double eps) {
  return nearly_equal(cuCreal(got), cuCreal(want), eps) &&
         nearly_equal(cuCimag(got), cuCimag(want), eps);
}

template <>
bool value_equal<__half>(__half got, __half want, double eps) {
  return nearly_equal(__half2float(got), __half2float(want), eps);
}

template <typename T>
static double value_real(T v) {
  return static_cast<double>(v);
}

template <>
double value_real<cuFloatComplex>(cuFloatComplex v) {
  return cuCrealf(v);
}

template <>
double value_real<cuDoubleComplex>(cuDoubleComplex v) {
  return cuCreal(v);
}

template <>
double value_real<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
static double value_imag(T) {
  return 0.0;
}

template <>
double value_imag<cuFloatComplex>(cuFloatComplex v) {
  return cuCimagf(v);
}

template <>
double value_imag<cuDoubleComplex>(cuDoubleComplex v) {
  return cuCimag(v);
}

template <typename T>
static bool check_array(const T* got, const T* want, int n, double eps) {
  for (int i = 0; i < n; ++i) {
    if (!value_equal(got[i], want[i], eps)) {
      std::printf("mismatch at %d: got (%.12f, %.12f) want (%.12f, %.12f)\n",
                  i,
                  value_real(got[i]),
                  value_imag(got[i]),
                  value_real(want[i]),
                  value_imag(want[i]));
      return false;
    }
  }
  return true;
}

static cublasHandle_t create_handle() {
  cublasHandle_t handle = nullptr;
  cublasStatus_t status = cublasCreate_v2(&handle);
  return status == CUBLAS_STATUS_SUCCESS ? handle : nullptr;
}

template <typename T>
static bool make_pointer_array(T* ptr, T*** out) {
  T* host[1] = {ptr};
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(out), sizeof(host)));
  CHECK_CUDA(cudaMemcpy(*out, host, sizeof(host), cudaMemcpyHostToDevice));
  return true;
}

template <typename T>
static bool make_const_pointer_array(T* ptr, const T*** out) {
  const T* host[1] = {ptr};
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(out), sizeof(host)));
  CHECK_CUDA(cudaMemcpy(*out, host, sizeof(host), cudaMemcpyHostToDevice));
  return true;
}

static bool test_cublasCreate_v2() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate_v2(&handle));
  if (handle == nullptr) return false;
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return true;
}

static bool test_cublasDestroy_v2() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return true;
}

static bool test_cublasSetStream_v2() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream_v2(handle, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return true;
}

static bool test_cublasSetPointerMode_v2() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  CHECK_CUBLAS(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE));
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  CHECK_CUBLAS(cublasGetPointerMode_v2(handle, &mode));
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return mode == CUBLAS_POINTER_MODE_DEVICE;
}

static bool test_cublasGetPointerMode_v2() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  cublasPointerMode_t mode;
  CHECK_CUBLAS(cublasGetPointerMode_v2(handle, &mode));
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return mode == CUBLAS_POINTER_MODE_HOST || mode == CUBLAS_POINTER_MODE_DEVICE;
}

static bool test_cublasSetMathMode() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  cublasMath_t mode;
  CHECK_CUBLAS(cublasGetMathMode(handle, &mode));
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return mode == CUBLAS_DEFAULT_MATH;
}

static bool test_cublasGetMathMode() {
  cublasHandle_t handle = create_handle();
  if (!handle) return false;
  cublasMath_t mode;
  CHECK_CUBLAS(cublasGetMathMode(handle, &mode));
  CHECK_CUBLAS(cublasDestroy_v2(handle));
  return mode == CUBLAS_DEFAULT_MATH || mode == CUBLAS_TENSOR_OP_MATH ||
         mode == CUBLAS_TF32_TENSOR_OP_MATH;
}

template <typename T, typename Func>
static bool run_axpy(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  const int n = 2;
  T alpha = make_value<T>(2.0);
  T x[n] = {make_value<T>(1.0, 1.0), make_value<T>(2.0, -1.0)};
  T y[n] = {make_value<T>(10.0, 0.0), make_value<T>(20.0, 2.0)};
  T want[n] = {make_value<T>(12.0, 2.0), make_value<T>(24.0, 0.0)};
  DeviceBuffer<T> dx(n), dy(n);
  if (!dx.ok() || !dy.ok() || !dx.copy_from(x, n) || !dy.copy_from(y, n)) {
    return false;
  }
  CHECK_CUBLAS(func(h, n, &alpha, dx.ptr, 1, dy.ptr, 1));
  if (!dy.copy_to(y, n)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(y, want, n, 1e-5);
}

template <typename T, typename Func>
static bool run_scal(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  const int n = 4;
  T alpha = make_value<T>(0.5);
  T x[n] = {make_value<T>(2), make_value<T>(4), make_value<T>(6), make_value<T>(8)};
  T want[n] = {make_value<T>(1), make_value<T>(2), make_value<T>(3), make_value<T>(4)};
  DeviceBuffer<T> dx(n);
  if (!dx.ok() || !dx.copy_from(x, n)) return false;
  CHECK_CUBLAS(func(h, n, &alpha, dx.ptr, 1));
  if (!dx.copy_to(x, n)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(x, want, n, 1e-5);
}

template <typename T, typename Func>
static bool run_copy(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  const int n = 4;
  T x[n] = {make_value<T>(11), make_value<T>(22), make_value<T>(33), make_value<T>(44)};
  T y[n] = {};
  DeviceBuffer<T> dx(n), dy(n);
  if (!dx.ok() || !dy.ok() || !dx.copy_from(x, n)) return false;
  CHECK_CUBLAS(func(h, n, dx.ptr, 1, dy.ptr, 1));
  if (!dy.copy_to(y, n)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(y, x, n, 1e-5);
}

template <typename T, typename Func>
static bool run_gemv(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(1), make_value<T>(3), make_value<T>(2), make_value<T>(4)};
  T x[2] = {make_value<T>(5, 1), make_value<T>(6, -1)};
  T y[2] = {};
  T want[2] = {make_value<T>(17, -1), make_value<T>(39, -1)};
  T alpha = make_value<T>(1), beta = make_value<T>(0);
  DeviceBuffer<T> dA(4), dx(2), dy(2);
  if (!dA.ok() || !dx.ok() || !dy.ok() || !dA.copy_from(A, 4) ||
      !dx.copy_from(x, 2) || !dy.copy_from(y, 2)) {
    return false;
  }
  CHECK_CUBLAS(func(h, CUBLAS_OP_N, 2, 2, &alpha, dA.ptr, 2, dx.ptr, 1,
                    &beta, dy.ptr, 1));
  if (!dy.copy_to(y, 2)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(y, want, 2, 1e-4);
}

template <typename T, typename Func>
static bool run_gemm(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(1), make_value<T>(3), make_value<T>(2), make_value<T>(4)};
  T B[4] = {make_value<T>(5), make_value<T>(7), make_value<T>(6), make_value<T>(8)};
  T C[4] = {};
  T want[4] = {make_value<T>(19), make_value<T>(43), make_value<T>(22), make_value<T>(50)};
  T alpha = make_value<T>(1), beta = make_value<T>(0);
  DeviceBuffer<T> dA(4), dB(4), dC(4);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A, 4) ||
      !dB.copy_from(B, 4) || !dC.copy_from(C, 4)) {
    return false;
  }
  CHECK_CUBLAS(func(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, dA.ptr, 2,
                    dB.ptr, 2, &beta, dC.ptr, 2));
  if (!dC.copy_to(C, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C, want, 4, 1e-3);
}

static bool run_hgemm_strided(bool use_strided) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  const int dim = 16;
  const int elems = dim * dim;
  const int batches = use_strided ? 2 : 1;
  std::vector<__half> A(elems * batches);
  std::vector<__half> B(elems * batches);
  std::vector<__half> C(elems * batches);
  std::vector<__half> want(elems * batches);
  for (int b = 0; b < batches; ++b) {
    for (int col = 0; col < dim; ++col) {
      for (int row = 0; row < dim; ++row) {
        int idx = b * elems + col * dim + row;
        A[idx] = make_value<__half>(row == col ? 1 : 0);
        B[idx] = make_value<__half>((row + 1) + col);
        C[idx] = make_value<__half>(0);
        want[idx] = B[idx];
      }
    }
  }
  __half alpha = make_value<__half>(1), beta = make_value<__half>(0);
  DeviceBuffer<__half> dA(elems * batches), dB(elems * batches), dC(elems * batches);
  if (!dA.ok() || !dB.ok() || !dC.ok() ||
      !dA.copy_from(A.data(), elems * batches) ||
      !dB.copy_from(B.data(), elems * batches) ||
      !dC.copy_from(C.data(), elems * batches)) {
    return false;
  }
  if (use_strided) {
    CHECK_CUBLAS(cublasHgemmStridedBatched(h,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           dim,
                                           dim,
                                           dim,
                                           &alpha,
                                           dA.ptr,
                                           dim,
                                           elems,
                                           dB.ptr,
                                           dim,
                                           elems,
                                           &beta,
                                           dC.ptr,
                                           dim,
                                           elems,
                                           batches));
  } else {
    CHECK_CUBLAS(cublasHgemm(h,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             dim,
                             dim,
                             dim,
                             &alpha,
                             dA.ptr,
                             dim,
                             dB.ptr,
                             dim,
                             &beta,
                             dC.ptr,
                             dim));
  }
  if (!dC.copy_to(C.data(), elems * batches)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C.data(), want.data(), elems * batches, 1e-2);
}

template <typename T, typename Func>
static bool run_geam(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(1), make_value<T>(3), make_value<T>(2), make_value<T>(4)};
  T B[4] = {make_value<T>(5, 1), make_value<T>(7, 1), make_value<T>(6, 1), make_value<T>(8, 1)};
  T C[4] = {};
  T want[4] = {make_value<T>(6, 1), make_value<T>(10, 1), make_value<T>(8, 1), make_value<T>(12, 1)};
  T alpha = make_value<T>(1), beta = make_value<T>(1);
  DeviceBuffer<T> dA(4), dB(4), dC(4);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A, 4) ||
      !dB.copy_from(B, 4)) {
    return false;
  }
  CHECK_CUBLAS(func(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, &alpha, dA.ptr, 2,
                    &beta, dB.ptr, 2, dC.ptr, 2));
  if (!dC.copy_to(C, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C, want, 4, 1e-4);
}

template <typename T, typename Func>
static bool run_trsm(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(2), make_value<T>(1), make_value<T>(0), make_value<T>(3)};
  T B[4] = {make_value<T>(10), make_value<T>(38), make_value<T>(14), make_value<T>(46)};
  T want[4] = {make_value<T>(5), make_value<T>(11), make_value<T>(7), make_value<T>(13)};
  T alpha = make_value<T>(1);
  DeviceBuffer<T> dA(4), dB(4);
  if (!dA.ok() || !dB.ok() || !dA.copy_from(A, 4) || !dB.copy_from(B, 4)) {
    return false;
  }
  CHECK_CUBLAS(func(h,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT,
                    2,
                    2,
                    &alpha,
                    dA.ptr,
                    2,
                    dB.ptr,
                    2));
  if (!dB.copy_to(B, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(B, want, 4, 1e-4);
}

template <typename T, typename Func>
static bool run_gemm_batched(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(1), make_value<T>(3), make_value<T>(2), make_value<T>(4)};
  T B[4] = {make_value<T>(5), make_value<T>(7), make_value<T>(6), make_value<T>(8)};
  T C[4] = {};
  T want[4] = {make_value<T>(19), make_value<T>(43), make_value<T>(22), make_value<T>(50)};
  T alpha = make_value<T>(1), beta = make_value<T>(0);
  DeviceBuffer<T> dA(4), dB(4), dC(4);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A, 4) ||
      !dB.copy_from(B, 4) || !dC.copy_from(C, 4)) {
    return false;
  }
  const T** dAarray = nullptr;
  const T** dBarray = nullptr;
  T** dCarray = nullptr;
  if (!make_const_pointer_array(dA.ptr, &dAarray) ||
      !make_const_pointer_array(dB.ptr, &dBarray) ||
      !make_pointer_array(dC.ptr, &dCarray)) {
    return false;
  }
  CHECK_CUBLAS(func(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, dAarray, 2,
                    dBarray, 2, &beta, dCarray, 2, 1));
  cudaFree(dAarray);
  cudaFree(dBarray);
  cudaFree(dCarray);
  if (!dC.copy_to(C, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C, want, 4, 1e-3);
}

template <typename T, typename Func>
static bool run_gemm_strided_batched(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[8], B[8], C[8], want[8];
  double ah[4] = {1, 3, 2, 4};
  double bh[4] = {5, 7, 6, 8};
  double wh[4] = {19, 43, 22, 50};
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 4; ++i) {
      A[b * 4 + i] = make_value<T>(ah[i]);
      B[b * 4 + i] = make_value<T>(bh[i]);
      C[b * 4 + i] = make_value<T>(0);
      want[b * 4 + i] = make_value<T>(wh[i]);
    }
  }
  T alpha = make_value<T>(1), beta = make_value<T>(0);
  DeviceBuffer<T> dA(8), dB(8), dC(8);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A, 8) ||
      !dB.copy_from(B, 8) || !dC.copy_from(C, 8)) {
    return false;
  }
  CHECK_CUBLAS(func(h,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &alpha,
                    dA.ptr,
                    2,
                    4,
                    dB.ptr,
                    2,
                    4,
                    &beta,
                    dC.ptr,
                    2,
                    4,
                    2));
  if (!dC.copy_to(C, 8)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C, want, 8, 1e-3);
}

template <typename T, typename Func>
static bool run_trsm_batched(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(2), make_value<T>(1), make_value<T>(0), make_value<T>(3)};
  T B[4] = {make_value<T>(10), make_value<T>(38), make_value<T>(14), make_value<T>(46)};
  T want[4] = {make_value<T>(5), make_value<T>(11), make_value<T>(7), make_value<T>(13)};
  T alpha = make_value<T>(1);
  DeviceBuffer<T> dA(4), dB(4);
  if (!dA.ok() || !dB.ok() || !dA.copy_from(A, 4) || !dB.copy_from(B, 4)) {
    return false;
  }
  T** dAarray = nullptr;
  T** dBarray = nullptr;
  if (!make_pointer_array(dA.ptr, &dAarray) ||
      !make_pointer_array(dB.ptr, &dBarray)) {
    return false;
  }
  CHECK_CUBLAS(func(h,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT,
                    2,
                    2,
                    &alpha,
                    dAarray,
                    2,
                    dBarray,
                    2,
                    1));
  cudaFree(dAarray);
  cudaFree(dBarray);
  if (!dB.copy_to(B, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(B, want, 4, 1e-4);
}

template <typename T, typename Func>
static bool run_getrf_batched(Func func) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(2), make_value<T>(1), make_value<T>(1), make_value<T>(2)};
  T want_lu[4] = {make_value<T>(2), make_value<T>(0.5), make_value<T>(1), make_value<T>(1.5)};
  DeviceBuffer<T> dA(4);
  DeviceBuffer<int> dP(2), dInfo(1);
  if (!dA.ok() || !dP.ok() || !dInfo.ok() || !dA.copy_from(A, 4)) return false;
  T** dAarray = nullptr;
  if (!make_pointer_array(dA.ptr, &dAarray)) return false;
  CHECK_CUBLAS(func(h, 2, dAarray, 2, dP.ptr, dInfo.ptr, 1));
  int info = -1;
  if (!dInfo.copy_to(&info, 1) || !dA.copy_to(A, 4)) return false;
  cudaFree(dAarray);
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return info == 0 && check_array(A, want_lu, 4, 1e-4);
}

template <typename T, typename GetrfFunc, typename GetriFunc>
static bool run_getri_batched(GetrfFunc getrf, GetriFunc getri) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(2), make_value<T>(1), make_value<T>(1), make_value<T>(2)};
  T C[4] = {};
  T want[4] = {make_value<T>(2.0 / 3.0), make_value<T>(-1.0 / 3.0),
               make_value<T>(-1.0 / 3.0), make_value<T>(2.0 / 3.0)};
  DeviceBuffer<T> dA(4), dC(4);
  DeviceBuffer<int> dP(2), dInfo(1);
  if (!dA.ok() || !dC.ok() || !dP.ok() || !dInfo.ok() || !dA.copy_from(A, 4)) {
    return false;
  }
  T** dAarray = nullptr;
  T** dCarray = nullptr;
  if (!make_pointer_array(dA.ptr, &dAarray) || !make_pointer_array(dC.ptr, &dCarray)) {
    return false;
  }
  CHECK_CUBLAS(getrf(h, 2, dAarray, 2, dP.ptr, dInfo.ptr, 1));
  CHECK_CUBLAS(getri(h,
                     2,
                     dAarray,
                     2,
                     dP.ptr,
                     dCarray,
                     2,
                     dInfo.ptr,
                     1));
  int info = -1;
  if (!dInfo.copy_to(&info, 1) || !dC.copy_to(C, 4)) return false;
  cudaFree(dAarray);
  cudaFree(dCarray);
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return info == 0 && check_array(C, want, 4, 1e-4);
}

template <typename T, typename GetrfFunc, typename GetrsFunc>
static bool run_getrs_batched(GetrfFunc getrf, GetrsFunc getrs) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T A[4] = {make_value<T>(2), make_value<T>(1), make_value<T>(1), make_value<T>(2)};
  T B[2] = {make_value<T>(5), make_value<T>(5)};
  T want[2] = {make_value<T>(5.0 / 3.0), make_value<T>(5.0 / 3.0)};
  DeviceBuffer<T> dA(4), dB(2);
  DeviceBuffer<int> dP(2), dInfo(1);
  if (!dA.ok() || !dB.ok() || !dP.ok() || !dInfo.ok() || !dA.copy_from(A, 4) ||
      !dB.copy_from(B, 2)) {
    return false;
  }
  T** dAarray = nullptr;
  T** dBarray = nullptr;
  if (!make_pointer_array(dA.ptr, &dAarray) || !make_pointer_array(dB.ptr, &dBarray)) {
    return false;
  }
  CHECK_CUBLAS(getrf(h, 2, dAarray, 2, dP.ptr, dInfo.ptr, 1));
  int info = -1;
  CHECK_CUBLAS(getrs(h,
                     CUBLAS_OP_N,
                     2,
                     1,
                     dAarray,
                     2,
                     dP.ptr,
                     dBarray,
                     2,
                     &info,
                     1));
  if (!dB.copy_to(B, 2)) return false;
  cudaFree(dAarray);
  cudaFree(dBarray);
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return info == 0 && check_array(B, want, 2, 1e-4);
}

template <typename T, typename Func>
static bool run_dot(Func func, T want) {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  T x[2] = {make_value<T>(1, 1), make_value<T>(2, -1)};
  T y[2] = {make_value<T>(3, 2), make_value<T>(4, -1)};
  T got = make_value<T>(0);
  DeviceBuffer<T> dx(2), dy(2);
  if (!dx.ok() || !dy.ok() || !dx.copy_from(x, 2) || !dy.copy_from(y, 2)) {
    return false;
  }
  CHECK_CUBLAS(func(h, 2, dx.ptr, 1, dy.ptr, 1, &got));
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return value_equal(got, want, 1e-4);
}

static bool test_cublasDotEx() {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  float x[3] = {1, 2, 3};
  float y[3] = {4, 5, 6};
  float got = 0;
  DeviceBuffer<float> dx(3), dy(3);
  if (!dx.ok() || !dy.ok() || !dx.copy_from(x, 3) || !dy.copy_from(y, 3)) {
    return false;
  }
  CHECK_CUBLAS(cublasDotEx(h,
                           3,
                           dx.ptr,
                           CUDA_R_32F,
                           1,
                           dy.ptr,
                           CUDA_R_32F,
                           1,
                           &got,
                           CUDA_R_32F,
                           CUDA_R_32F));
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return nearly_equal(got, 32.0, 1e-5);
}

static bool test_cublasSgemmEx() {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  const int dim = 16;
  const int elems = dim * dim;
  std::vector<__half> A(elems);
  std::vector<__half> B(elems);
  std::vector<__half> C(elems);
  std::vector<__half> want(elems);
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row) {
      int idx = col * dim + row;
      A[idx] = make_value<__half>(row == col ? 1 : 0);
      B[idx] = make_value<__half>((row + 1) + col);
      C[idx] = make_value<__half>(0);
      want[idx] = B[idx];
    }
  }
  float alpha = 1, beta = 0;
  DeviceBuffer<__half> dA(elems), dB(elems), dC(elems);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A.data(), elems) ||
      !dB.copy_from(B.data(), elems) || !dC.copy_from(C.data(), elems)) {
    return false;
  }
  CHECK_CUBLAS(cublasSgemmEx(h,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             dim,
                             dim,
                             dim,
                             &alpha,
                             dA.ptr,
                             CUDA_R_16F,
                             dim,
                             dB.ptr,
                             CUDA_R_16F,
                             dim,
                             &beta,
                             dC.ptr,
                             CUDA_R_16F,
                             dim));
  if (!dC.copy_to(C.data(), elems)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C.data(), want.data(), elems, 1e-2);
}

static bool test_cublasGemmEx() {
  return run_gemm<float>([](cublasHandle_t h,
                            cublasOperation_t ta,
                            cublasOperation_t tb,
                            int m,
                            int n,
                            int k,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            const float* beta,
                            float* C,
                            int ldc) {
    return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, B,
                        CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  });
}

static bool test_cublasSgemmEx_64() {
  return run_gemm<float>([](cublasHandle_t h,
                            cublasOperation_t ta,
                            cublasOperation_t tb,
                            int m,
                            int n,
                            int k,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            const float* beta,
                            float* C,
                            int ldc) {
    return cublasSgemmEx_64(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, B,
                            CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc);
  });
}

static bool test_cublasGemmEx_64() {
  return run_gemm<float>([](cublasHandle_t h,
                            cublasOperation_t ta,
                            cublasOperation_t tb,
                            int m,
                            int n,
                            int k,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            const float* beta,
                            float* C,
                            int ldc) {
    return cublasGemmEx_64(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, B,
                           CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  });
}

static bool test_cublasGemmBatchedEx() {
  cublasHandle_t h = create_handle();
  if (!h) return false;
  float A[4] = {1, 3, 2, 4};
  float B[4] = {5, 7, 6, 8};
  float C[4] = {};
  float want[4] = {19, 43, 22, 50};
  float alpha = 1, beta = 0;
  DeviceBuffer<float> dA(4), dB(4), dC(4);
  if (!dA.ok() || !dB.ok() || !dC.ok() || !dA.copy_from(A, 4) ||
      !dB.copy_from(B, 4) || !dC.copy_from(C, 4)) {
    return false;
  }
  const void* Ahost[1] = {dA.ptr};
  const void* Bhost[1] = {dB.ptr};
  void* Chost[1] = {dC.ptr};
  const void** dAarray = nullptr;
  const void** dBarray = nullptr;
  void** dCarray = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dAarray), sizeof(Ahost)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dBarray), sizeof(Bhost)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dCarray), sizeof(Chost)));
  CHECK_CUDA(cudaMemcpy(dAarray, Ahost, sizeof(Ahost), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBarray, Bhost, sizeof(Bhost), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dCarray, Chost, sizeof(Chost), cudaMemcpyHostToDevice));
  CHECK_CUBLAS(cublasGemmBatchedEx(h,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   2,
                                   2,
                                   2,
                                   &alpha,
                                   dAarray,
                                   CUDA_R_32F,
                                   2,
                                   dBarray,
                                   CUDA_R_32F,
                                   2,
                                   &beta,
                                   dCarray,
                                   CUDA_R_32F,
                                   2,
                                   1,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT));
  cudaFree(dAarray);
  cudaFree(dBarray);
  cudaFree(dCarray);
  if (!dC.copy_to(C, 4)) return false;
  CHECK_CUBLAS(cublasDestroy_v2(h));
  return check_array(C, want, 4, 1e-3);
}

static bool test_cublasGemmStridedBatchedEx() {
  return run_gemm_strided_batched<float>([](cublasHandle_t h,
                                            cublasOperation_t ta,
                                            cublasOperation_t tb,
                                            int m,
                                            int n,
                                            int k,
                                            const float* alpha,
                                            const float* A,
                                            int lda,
                                            long long int strideA,
                                            const float* B,
                                            int ldb,
                                            long long int strideB,
                                            const float* beta,
                                            float* C,
                                            int ldc,
                                            long long int strideC,
                                            int batchCount) {
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F,
                                      lda, strideA, B, CUDA_R_32F, ldb, strideB,
                                      beta, C, CUDA_R_32F, ldc, strideC,
                                      batchCount, CUDA_R_32F,
                                      CUBLAS_GEMM_DEFAULT);
  });
}

static bool test_cublasGemmStridedBatchedEx_64() {
  return run_gemm_strided_batched<float>([](cublasHandle_t h,
                                            cublasOperation_t ta,
                                            cublasOperation_t tb,
                                            int m,
                                            int n,
                                            int k,
                                            const float* alpha,
                                            const float* A,
                                            int lda,
                                            long long int strideA,
                                            const float* B,
                                            int ldb,
                                            long long int strideB,
                                            const float* beta,
                                            float* C,
                                            int ldc,
                                            long long int strideC,
                                            int batchCount) {
    return cublasGemmStridedBatchedEx_64(h, ta, tb, m, n, k, alpha, A,
                                         CUDA_R_32F, lda, strideA, B,
                                         CUDA_R_32F, ldb, strideB, beta, C,
                                         CUDA_R_32F, ldc, strideC, batchCount,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
  });
}

static bool test_cublasSaxpy() { return run_axpy<float>(cublasSaxpy); }
static bool test_cublasDaxpy() { return run_axpy<double>(cublasDaxpy); }
static bool test_cublasCaxpy() { return run_axpy<cuFloatComplex>(cublasCaxpy); }
static bool test_cublasZaxpy() { return run_axpy<cuDoubleComplex>(cublasZaxpy); }
static bool test_cublasSscal() { return run_scal<float>(cublasSscal); }
static bool test_cublasDscal() { return run_scal<double>(cublasDscal); }
static bool test_cublasScopy() { return run_copy<float>(cublasScopy); }
static bool test_cublasDcopy() { return run_copy<double>(cublasDcopy); }
static bool test_cublasSgemv_v2() { return run_gemv<float>(cublasSgemv_v2); }
static bool test_cublasDgemv_v2() { return run_gemv<double>(cublasDgemv_v2); }
static bool test_cublasCgemv_v2() { return run_gemv<cuFloatComplex>(cublasCgemv_v2); }
static bool test_cublasZgemv_v2() { return run_gemv<cuDoubleComplex>(cublasZgemv_v2); }
static bool test_cublasSgemm_v2() { return run_gemm<float>(cublasSgemm_v2); }
static bool test_cublasDgemm_v2() { return run_gemm<double>(cublasDgemm_v2); }
static bool test_cublasCgemm_v2() { return run_gemm<cuFloatComplex>(cublasCgemm_v2); }
static bool test_cublasZgemm_v2() { return run_gemm<cuDoubleComplex>(cublasZgemm_v2); }
static bool test_cublasHgemm() { return run_hgemm_strided(false); }
static bool test_cublasSgeam() { return run_geam<float>(cublasSgeam); }
static bool test_cublasDgeam() { return run_geam<double>(cublasDgeam); }
static bool test_cublasCgeam() { return run_geam<cuFloatComplex>(cublasCgeam); }
static bool test_cublasZgeam() { return run_geam<cuDoubleComplex>(cublasZgeam); }
static bool test_cublasStrsm_v2() { return run_trsm<float>(cublasStrsm_v2); }
static bool test_cublasDtrsm_v2() { return run_trsm<double>(cublasDtrsm_v2); }
static bool test_cublasCtrsm_v2() { return run_trsm<cuFloatComplex>(cublasCtrsm_v2); }
static bool test_cublasZtrsm_v2() { return run_trsm<cuDoubleComplex>(cublasZtrsm_v2); }
static bool test_cublasSgemmBatched() { return run_gemm_batched<float>(cublasSgemmBatched); }
static bool test_cublasDgemmBatched() { return run_gemm_batched<double>(cublasDgemmBatched); }
static bool test_cublasCgemmBatched() { return run_gemm_batched<cuFloatComplex>(cublasCgemmBatched); }
static bool test_cublasZgemmBatched() { return run_gemm_batched<cuDoubleComplex>(cublasZgemmBatched); }
static bool test_cublasSgemmStridedBatched() { return run_gemm_strided_batched<float>(cublasSgemmStridedBatched); }
static bool test_cublasDgemmStridedBatched() { return run_gemm_strided_batched<double>(cublasDgemmStridedBatched); }
static bool test_cublasCgemmStridedBatched() { return run_gemm_strided_batched<cuFloatComplex>(cublasCgemmStridedBatched); }
static bool test_cublasZgemmStridedBatched() { return run_gemm_strided_batched<cuDoubleComplex>(cublasZgemmStridedBatched); }
static bool test_cublasHgemmStridedBatched() { return run_hgemm_strided(true); }
static bool test_cublasStrsmBatched() { return run_trsm_batched<float>(cublasStrsmBatched); }
static bool test_cublasDtrsmBatched() { return run_trsm_batched<double>(cublasDtrsmBatched); }
static bool test_cublasCtrsmBatched() { return run_trsm_batched<cuFloatComplex>(cublasCtrsmBatched); }
static bool test_cublasZtrsmBatched() { return run_trsm_batched<cuDoubleComplex>(cublasZtrsmBatched); }
static bool test_cublasSgetrfBatched() { return run_getrf_batched<float>(cublasSgetrfBatched); }
static bool test_cublasDgetrfBatched() { return run_getrf_batched<double>(cublasDgetrfBatched); }
static bool test_cublasCgetrfBatched() { return run_getrf_batched<cuFloatComplex>(cublasCgetrfBatched); }
static bool test_cublasZgetrfBatched() { return run_getrf_batched<cuDoubleComplex>(cublasZgetrfBatched); }
static bool test_cublasSgetriBatched() { return run_getri_batched<float>(cublasSgetrfBatched, cublasSgetriBatched); }
static bool test_cublasDgetriBatched() { return run_getri_batched<double>(cublasDgetrfBatched, cublasDgetriBatched); }
static bool test_cublasCgetriBatched() { return run_getri_batched<cuFloatComplex>(cublasCgetrfBatched, cublasCgetriBatched); }
static bool test_cublasZgetriBatched() { return run_getri_batched<cuDoubleComplex>(cublasZgetrfBatched, cublasZgetriBatched); }
static bool test_cublasSgetrsBatched() { return run_getrs_batched<float>(cublasSgetrfBatched, cublasSgetrsBatched); }
static bool test_cublasDgetrsBatched() { return run_getrs_batched<double>(cublasDgetrfBatched, cublasDgetrsBatched); }
static bool test_cublasSdot_v2() { return run_dot<float>(cublasSdot_v2, make_value<float>(11)); }
static bool test_cublasDdot_v2() { return run_dot<double>(cublasDdot_v2, make_value<double>(11)); }
static bool test_cublasCdotu_v2() { return run_dot<cuFloatComplex>(cublasCdotu_v2, make_value<cuFloatComplex>(8, -1)); }
static bool test_cublasZdotu_v2() { return run_dot<cuDoubleComplex>(cublasZdotu_v2, make_value<cuDoubleComplex>(8, -1)); }
static bool test_cublasCdotc_v2() { return run_dot<cuFloatComplex>(cublasCdotc_v2, make_value<cuFloatComplex>(14, 1)); }
static bool test_cublasZdotc_v2() { return run_dot<cuDoubleComplex>(cublasZdotc_v2, make_value<cuDoubleComplex>(14, 1)); }

int main() {
  TestCase tests[] = {
      {"cublasSaxpy", test_cublasSaxpy},
      {"cublasDaxpy", test_cublasDaxpy},
      {"cublasCaxpy", test_cublasCaxpy},
      {"cublasZaxpy", test_cublasZaxpy},
      {"cublasSscal", test_cublasSscal},
      {"cublasDscal", test_cublasDscal},
      {"cublasScopy", test_cublasScopy},
      {"cublasDcopy", test_cublasDcopy},
      {"cublasSgemv_v2", test_cublasSgemv_v2},
      {"cublasDgemv_v2", test_cublasDgemv_v2},
      {"cublasCgemv_v2", test_cublasCgemv_v2},
      {"cublasZgemv_v2", test_cublasZgemv_v2},
      {"cublasSgemm_v2", test_cublasSgemm_v2},
      {"cublasDgemm_v2", test_cublasDgemm_v2},
      {"cublasCgemm_v2", test_cublasCgemm_v2},
      {"cublasZgemm_v2", test_cublasZgemm_v2},
      {"cublasHgemm", test_cublasHgemm},
      {"cublasSgemmEx", test_cublasSgemmEx},
      {"cublasSgeam", test_cublasSgeam},
      {"cublasDgeam", test_cublasDgeam},
      {"cublasStrsm_v2", test_cublasStrsm_v2},
      {"cublasDtrsm_v2", test_cublasDtrsm_v2},
      {"cublasCtrsm_v2", test_cublasCtrsm_v2},
      {"cublasZtrsm_v2", test_cublasZtrsm_v2},
      {"cublasCreate_v2", test_cublasCreate_v2},
      {"cublasDestroy_v2", test_cublasDestroy_v2},
      {"cublasSetStream_v2", test_cublasSetStream_v2},
      {"cublasSetPointerMode_v2", test_cublasSetPointerMode_v2},
      {"cublasGetPointerMode_v2", test_cublasGetPointerMode_v2},
      {"cublasSgemmBatched", test_cublasSgemmBatched},
      {"cublasDgemmBatched", test_cublasDgemmBatched},
      {"cublasCgemmBatched", test_cublasCgemmBatched},
      {"cublasZgemmBatched", test_cublasZgemmBatched},
      {"cublasStrsmBatched", test_cublasStrsmBatched},
      {"cublasDtrsmBatched", test_cublasDtrsmBatched},
      {"cublasCtrsmBatched", test_cublasCtrsmBatched},
      {"cublasZtrsmBatched", test_cublasZtrsmBatched},
      {"cublasSgetrfBatched", test_cublasSgetrfBatched},
      {"cublasSgetriBatched", test_cublasSgetriBatched},
      {"cublasDgetrfBatched", test_cublasDgetrfBatched},
      {"cublasDgetriBatched", test_cublasDgetriBatched},
      {"cublasCgetrfBatched", test_cublasCgetrfBatched},
      {"cublasCgetriBatched", test_cublasCgetriBatched},
      {"cublasZgetrfBatched", test_cublasZgetrfBatched},
      {"cublasZgetriBatched", test_cublasZgetriBatched},
      {"cublasSgetrsBatched", test_cublasSgetrsBatched},
      {"cublasDgetrsBatched", test_cublasDgetrsBatched},
      {"cublasSdot_v2", test_cublasSdot_v2},
      {"cublasDdot_v2", test_cublasDdot_v2},
      {"cublasCdotc_v2", test_cublasCdotc_v2},
      {"cublasZdotc_v2", test_cublasZdotc_v2},
      {"cublasCdotu_v2", test_cublasCdotu_v2},
      {"cublasZdotu_v2", test_cublasZdotu_v2},
      {"cublasDotEx", test_cublasDotEx},
      {"cublasGemmEx", test_cublasGemmEx},
      {"cublasSgemmStridedBatched", test_cublasSgemmStridedBatched},
      {"cublasDgemmStridedBatched", test_cublasDgemmStridedBatched},
      {"cublasCgemmStridedBatched", test_cublasCgemmStridedBatched},
      {"cublasZgemmStridedBatched", test_cublasZgemmStridedBatched},
      {"cublasHgemmStridedBatched", test_cublasHgemmStridedBatched},
      {"cublasSetMathMode", test_cublasSetMathMode},
      {"cublasGetMathMode", test_cublasGetMathMode},
      {"cublasCgeam", test_cublasCgeam},
      {"cublasZgeam", test_cublasZgeam},
      {"cublasGemmBatchedEx", test_cublasGemmBatchedEx},
      {"cublasGemmStridedBatchedEx", test_cublasGemmStridedBatchedEx},
      {"cublasGemmStridedBatchedEx_64", test_cublasGemmStridedBatchedEx_64},
      {"cublasGemmEx_64", test_cublasGemmEx_64},
      {"cublasSgemmEx_64", test_cublasSgemmEx_64},
  };
  int passed = 0;
  int total = sizeof(tests) / sizeof(tests[0]);
  for (const auto& test : tests) {
    bool ok = test.fn();
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", test.name);
    passed += ok ? 1 : 0;
  }
  std::printf("Results: %d/%d passed\n", passed, total);
  return passed == total ? 0 : 1;
}




