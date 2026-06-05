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
