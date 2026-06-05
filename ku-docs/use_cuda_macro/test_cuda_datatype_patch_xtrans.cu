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
