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
