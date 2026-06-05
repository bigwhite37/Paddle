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
