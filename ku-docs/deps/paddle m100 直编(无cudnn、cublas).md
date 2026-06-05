# 准备环境

## 1.2. 下载xcuda包
```bash
# wget https://klx-sdk-release-public.su.bcebos.com/mars_release/XTRANSCUDA/dev/latest/xtrans_cuda_11.7_ubuntu2004_x86_64_mars.tar.gz
# tar -xvf xtrans_cuda_11.7_ubuntu2004_x86_64_mars.tar.gz
wget https://klx-sdk-release-public.su.bcebos.com/mars_release/XTRANSCUDA/dev/latest/xtrans_cuda_12.8_ubuntu2004_x86_64_mars.tar.gz
tar -xzvf xtrans_cuda_12.8_ubuntu2004_x86_64_mars.tar.gz
```


## 1.3. 下载xnccl产出包(只验证了cuda 12)
```bash
wget -O output.tar.gz --no-check-certificate --header "IREPO-TOKEN:b7bcb630-febc-4c17-8fad-ccbce41b6931" "https://irepo.baidu-int.com/rest/prod/v3/baidu/xpu/bkcl-ci/nodes/96914472/files"
tar -xzvf output.tar.gz
```
# 编译

## 2.2 配置环境变量
```bash
# export XCUDA_PATH=/framework/m100/xtrans_cuda_11.7_ubuntu2004_x86_64_mars
export XCUDA_PATH=/framework/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars
export PATH=${XCUDA_PATH}/bin:/root/miniconda/envs/python310_torch25_cuda/bin:${PATH}
export CUDA_PATH=${XCUDA_PATH}
export LD_LIBRARY_PATH=${XCUDA_PATH}/lib64:${XCUDA_PATH}/lib:${LD_LIBRARY_PATH}
export LDFLAGS=-L$XCUDA_PATH/lib64/
export XTRANS_DIR=${XCUDA_PATH}
export CXX=$XCUDA_PATH/bin/clang++
export CUDNN_ROOT=$XCUDA_PATH/targets/x86_64-linux
# export CUDNN_ROOT=/framework/m100/xcudnn/output
export CUPTI_ROOT=$XCUDA_PATH/targets/x86_64-linux
export NCCL_ROOT=/framework/m100/nccl/output/xccl_Linux_x86_64_nccl_cuda12
# export CPATH=${NCCL_ROOT}/include:${CPATH}
# export XMLIR_CUDNN_ENABLED=true
export CMAKE_CUDA_ARCHITECTURES=80

```


## 2.3. 修改额外代码
```cpp
#if !defined(__KL3_XRE__) && !defined(USE_CUDA) && !defined(PADDLE_WITH_CUDA)
#include "cudnn_api/xpudnn_cuda_patch.h"
#endif
```


[评审：xpu-paddlepaddle-1135 Add support of PADDLE_WITH_CUDA](https://console.cloud.baidu-int.com/devops/icode/repos/baidu/xpu/cuDNN2/reviews/120862481/?t=mention&mt=doc&dt=sdk)



### 2.3.2. cuda 12.8 修改Paddle
#### **2.3.2.1 链接****libcudnn.so.9**
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=004937e0ec494237ae56cad2acb9a620&docGuid=TgOjbAniE11KQ-)
**cuda版本大于12.6时找libcudnn.so.9，但xtrans里没有，需要链接**

```bash
cd $XCUDA_PATH/targets/x86_64-linux/lib
ln -s libcudnn.so.8.9 libcudnn.so.9
```


#### 2.3.2.2 xtrans增加重载(两个文件都需要)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=4daf910cde92407eaa26ca585d2592f1&docGuid=TgOjbAniE11KQ-)
```cpp
__device__ static inline unsigned int __ffs(unsigned long long int input) {
  return __ffsll(input);
}

__device__ static inline unsigned int __ffs(unsigned long int input) {
  return __ffsll(static_cast<unsigned long long int>(input));
}
```


![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=5e23c1279f944d95b97cc1c46865361f&docGuid=TgOjbAniE11KQ-)
```cpp

__device__ static inline unsigned int __ffs(unsigned long long int input) {
  return __ffsll(input);
}

__device__ static inline unsigned int __ffs(unsigned long int input) {
  return __ffsll(static_cast<unsigned long long int>(input));
}
```


#### 2.3.2.3 Paddle修改
```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 6c01573ae2..53af480407 100755
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -534,6 +534,7 @@ if(WITH_GPU)
   if(NOT WIN32)
     include(cupti)
   endif()
+  set_property(GLOBAL APPEND PROPERTY OS_DEPENDENCY_MODULES ${CUDA_LIBRARIES})
 endif()

 if(WITH_ROCM)
diff --git a/cmake/cuda.cmake b/cmake/cuda.cmake
index 1b94d00812..c6dcd1a025 100644
--- a/cmake/cuda.cmake
+++ b/cmake/cuda.cmake
@@ -228,12 +228,12 @@ function(select_nvcc_arch_flags out_variable out_arch_bin)
     if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
       # User explicitly specified PTX for the concrete BIN
       string(APPEND nvcc_flags
-             " -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1}")
+             " --cuda-gpu-arch=sm_${CMAKE_MATCH_1}")
       string(APPEND nvcc_archs_readable " sm_${CMAKE_MATCH_1}")
       string(APPEND nvcc_archs_bin_list " ${CMAKE_MATCH_1}")
     else()
       # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
-      string(APPEND nvcc_flags " -gencode arch=compute_${arch},code=sm_${arch}")
+      string(APPEND nvcc_flags " --cuda-gpu-arch=sm_${arch}")
       string(APPEND nvcc_archs_readable " sm_${arch}")
       string(APPEND nvcc_archs_bin_list " ${arch}")
     endif()
@@ -242,7 +242,7 @@ function(select_nvcc_arch_flags out_variable out_arch_bin)
   # Tell NVCC to add PTX intermediate code for the specified architectures
   foreach(arch ${cuda_arch_ptx})
     string(APPEND nvcc_flags
-           " -gencode arch=compute_${arch},code=compute_${arch}")
+           " --cuda-gpu-arch=compute_${arch}")
     string(APPEND nvcc_archs_readable " compute_${arch}")
   endforeach()

diff --git a/cmake/external/dgc.cmake b/cmake/external/dgc.cmake
index 579b7f2da8..af9bf1a647 100644
--- a/cmake/external/dgc.cmake
+++ b/cmake/external/dgc.cmake
@@ -76,7 +76,8 @@ ExternalProject_Add(
   URL_MD5 ${DGC_URL_MD5}
   PREFIX "${DGC_PREFIX_DIR}"
   CONFIGURE_COMMAND ""
-  BUILD_COMMAND make -j${NPROC}
+  # BUILD_COMMAND make -j${NPROC}
+  BUILD_COMMAND make -j${NPROC} NCCL_INCLUDE=${NCCL_INCLUDE_DIR}
   DOWNLOAD_DIR ${DGC_DOWNLOAD_DIR}
   SOURCE_DIR ${DGC_SOURCES_DIR}
   INSTALL_COMMAND
diff --git a/cmake/external/warpctc.cmake b/cmake/external/warpctc.cmake
index 17ef70b4a0..ad6e24aaa8 100644
--- a/cmake/external/warpctc.cmake
+++ b/cmake/external/warpctc.cmake
@@ -124,6 +124,9 @@ ExternalProject_Add(
   #BUILD_ALWAYS    1
   CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
+             -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
+             -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
+             -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
              -DCMAKE_C_FLAGS=${WARPCTC_C_FLAGS}
              -DCMAKE_C_FLAGS_DEBUG=${WARPCTC_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_RELEASE=${WARPCTC_C_FLAGS_RELEASE}
diff --git a/cmake/external/warprnnt.cmake b/cmake/external/warprnnt.cmake
index ce4b43343a..46a5793a89 100644
--- a/cmake/external/warprnnt.cmake
+++ b/cmake/external/warprnnt.cmake
@@ -125,6 +125,9 @@ ExternalProject_Add(
   #BUILD_ALWAYS    1
   CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
+             -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
+             -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
+             -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
              -DCMAKE_C_FLAGS=${WARPRNNT_C_FLAGS}
              -DCMAKE_C_FLAGS_DEBUG=${WARPRNNT_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_RELEASE=${WARPRNNT_C_FLAGS_RELEASE}
diff --git a/cmake/nccl.cmake b/cmake/nccl.cmake
index eaa7bd23fd..2ed893ddda 100644
--- a/cmake/nccl.cmake
+++ b/cmake/nccl.cmake
@@ -16,6 +16,8 @@ if(WITH_NCCL)
     PATHS ${NCCL_ROOT} ${NCCL_ROOT}/include ${NCCL_ROOT}/local/include
           $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include $ENV{NCCL_ROOT}/local/include
     NO_DEFAULT_PATH)
+  include_directories(BEFORE ${NCCL_INCLUDE_DIR})
+  add_definitions("-DBKCL_CUDA_FP16_COMPAT_H_")

   file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)

diff --git a/paddle/phi/core/enforce.h b/paddle/phi/core/enforce.h
index 4c4097e1d7..5b839c9ed0 100644
--- a/paddle/phi/core/enforce.h
+++ b/paddle/phi/core/enforce.h
@@ -113,7 +113,7 @@ void ThrowWarnInternal(const std::string& message);
              __LINE__,                                             \
              #_IS_NOT_ERROR,                                       \
              ##__VA_ARGS__);                                       \
-      asm("trap;");                                                \
+      __builtin_trap();                                                \
     }                                                              \
   } while (0)
 #elif defined(__HIPCC__)
diff --git a/paddle/phi/core/platform/device/gpu/gpu_dnn.h b/paddle/phi/core/platform/device/gpu/gpu_dnn.h
index 3418089f8d..7c68e4a59d 100644
--- a/paddle/phi/core/platform/device/gpu/gpu_dnn.h
+++ b/paddle/phi/core/platform/device/gpu/gpu_dnn.h
@@ -22,6 +22,7 @@ namespace paddle {
 namespace platform {

 using DataLayout = phi::DataLayout;
+#ifdef WITH_CUDNN_FRONTEND
 using PoolingMode = phi::backends::gpu::PoolingMode;
 template <typename T>
 using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
@@ -40,6 +41,7 @@ using ScopedRNNTensorDescriptor = phi::backends::gpu::ScopedRNNTensorDescriptor;
 using ScopedSpatialTransformerDescriptor =
     phi::backends::gpu::ScopedSpatialTransformerDescriptor;
 #endif
+#endif  // WITH_CUDNN_FRONTEND

 }  // namespace platform
 }  // namespace paddle
diff --git a/paddle/phi/kernels/funcs/blas/blas_impl.cu.h b/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
index e9a033a24c..0c02292370 100644
--- a/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
+++ b/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
@@ -196,7 +196,8 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void GETRF_BATCH(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
+    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented( "GETRI_BATCH is not supported by xtrans."));
   }

   template <typename... ARGS>
@@ -218,8 +219,8 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void TRSM_BATCH(ARGS... args) {
-  //  PADDLE_THROW(phi::errors::Unimplemented("SmatinvBatched is not supported by xtrans."));
-  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("SmatinvBatched is not supported by xtrans."));
+    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
   }

   template <typename... ARGS>
@@ -306,7 +307,7 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void GETRI_BATCH(ARGS... args) {
-    PADDLE_THROW(phi::errors::Unimplemented("GETRI_BATCH is not supported by xtrans. upgrade"));
+    PADDLE_THROW(phi::errors::Unimplemented("GETRI_BATCH is not supported by xtrans."));
     // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetriBatched(args...));
   }

@@ -323,8 +324,8 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void TRSM_BATCH(ARGS... args) {
-    // PADDLE_THROW(phi::errors::Unimplemented("DmatinvBatched is not supported by xtrans."));
-         PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("DmatinvBatched is not supported by xtrans."));
+         // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
   }

   template <typename... ARGS>
diff --git a/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu b/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu
index 75f0ab872c..b4c03973ab 100644
--- a/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu
+++ b/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu
@@ -52,7 +52,8 @@ template <int Width = 32>
 __device__ __nv_bfloat16 WarpReduceMax(__nv_bfloat16 x) {
   constexpr unsigned mask = (uint64_t(1) << Width) - 1;
   for (int offset = Width / 2; offset > 0; offset /= 2) {
-    __nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
+    __nv_bfloat16 t = static_cast<__nv_bfloat16>(__shfl_down_sync(mask, static_cast<float>(x), offset));
+    // __nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
     x = BF16_MAX(x, t);
   }
   return x;
diff --git a/paddle/phi/kernels/legacy/gpu/batched_gemm.cu b/paddle/phi/kernels/legacy/gpu/batched_gemm.cu
index e8ab664e13..45c2dd139e 100644
--- a/paddle/phi/kernels/legacy/gpu/batched_gemm.cu
+++ b/paddle/phi/kernels/legacy/gpu/batched_gemm.cu
@@ -123,7 +123,7 @@ void CublasGemm(cublasHandle_t cublas_handle,
                                            CUDA_R_32F,
                                            CUBLAS_GEMM_DEFAULT));
   } else if constexpr (std::is_same<T, float>::value) {
-    CUBLAS_CALL(phi::dynload::cublasSgemm(cublas_handle,
+    CUBLAS_CALL(phi::dynload::cublasSgemm_v2(cublas_handle,
                                           transpose_b,
                                           transpose_a,
                                           m,
diff --git a/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu b/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu
index 40bb78f973..fdcccab0e6 100644
--- a/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu
+++ b/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu
@@ -433,7 +433,7 @@ __device__ void ComputeRowScale(const v64_t<T> x[8],
     // reduce [32] => [1]
     T warp_max = local_max;
     for (uint32_t offset = 16; offset > 0; offset /= 2) {
-      T other = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
+      T other = static_cast<T>(__shfl_down_sync(0xFFFFFFFF, static_cast<float>(warp_max), offset));
       warp_max = device_max(warp_max, other);
     }
     if (threadIdx.x == 0) {
diff --git a/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu b/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu
index 31dbeafc1e..8f3765cd84 100644
--- a/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu
+++ b/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu
@@ -114,7 +114,9 @@ __device__ void ComputeScaleAndWrite(__nv_bfloat16 *data,

   // Parallel reduction within each group
   for (int stride = group_size / 2; stride > 0; stride >>= 1) {
-    __nv_bfloat16 other = __shfl_down_sync(mask, global_max, stride);
+    // __nv_bfloat16 other = __shfl_down_sync(mask, global_max, stride);
+    __nv_bfloat16 other = static_cast<__nv_bfloat16>(__shfl_down_sync(mask, static_cast<float>(global_max), stride));
+
     global_max = BF16_MAX(other, global_max);
   }

diff --git a/patches/cccl/util_device.cuh.patch b/patches/cccl/util_device.cuh.patch
index bdf7165328..fa33359e19 100644
--- a/patches/cccl/util_device.cuh.patch
+++ b/patches/cccl/util_device.cuh.patch
@@ -29,3 +29,43 @@ index c7e15cafe..756336914 100644
  cudaError_t MaxSmOccupancy(
      int&                max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
      KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
+diff --git a/cub/cub/util_ptx.cuh b/cub/cub/util_ptx.cuh
+index ff6fdb07f5..daaed83e5a 100644
+--- a/cub/cub/util_ptx.cuh
++++ b/cub/cub/util_ptx.cuh
+@@ -394,7 +394,7 @@ __device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
+  * \brief Terminates the calling thread
+  */
+ __device__ __forceinline__ void ThreadExit() {
+-    asm volatile("exit;");
++    asm volatile("s_endpgm");
+ }
+
+
+@@ -402,7 +402,7 @@ __device__ __forceinline__ void ThreadExit() {
+  * \brief  Abort execution and generate an interrupt to the host CPU
+  */
+ __device__ __forceinline__ void ThreadTrap() {
+-    asm volatile("trap;");
++    __builtin_trap();
+ }
+
+
+diff --git a/libcudacxx/include/cuda/pipeline b/libcudacxx/include/cuda/pipeline
+index ec5cbd1c93..5b1a103aba 100644
+--- a/libcudacxx/include/cuda/pipeline
++++ b/libcudacxx/include/cuda/pipeline
+@@ -108,7 +108,7 @@ _LIBCUDACXX_BEGIN_NAMESPACE_CUDA
+                 NV_IS_DEVICE,
+                 (
+                     uint32_t __lane_id;
+-                    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(__lane_id));
++                    __lane_id = ::__lane_id();
+                     return __lane_id;
+                 ),
+                 (
+                     return 0;
+                )
+            )
+        }
+    };
\ No newline at end of file
diff --git a/patches/warpctc/CMakeLists.txt.cuda.patch b/patches/warpctc/CMakeLists.txt.cuda.patch
index 9cf204e95a..5fcac855c4 100644
--- a/patches/warpctc/CMakeLists.txt.cuda.patch
+++ b/patches/warpctc/CMakeLists.txt.cuda.patch
@@ -55,3 +55,14 @@
  ENDIF()

  IF (APPLE)
+@@ -160,7 +123,9 @@ IF (WITH_GPU OR WITH_ROCM)
+         CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/.ctc_entrypoint.cu src/reduce.cu)
+     else()
+         IF (WITH_GPU)
+-            CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
++            # CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
++            enable_language(CUDA)
++            add_library(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
+         ELSE()
+             HIP_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
+             TARGET_LINK_LIBRARIES(warpctc PUBLIC ${ROCM_HIPRTC_LIB})
diff --git a/patches/warprnnt/CMakeLists.txt.cuda.patch b/patches/warprnnt/CMakeLists.txt.cuda.patch
index 16967534de..23d5315da6 100644
--- a/patches/warprnnt/CMakeLists.txt.cuda.patch
+++ b/patches/warprnnt/CMakeLists.txt.cuda.patch
@@ -54,3 +54,14 @@
  ENDIF()

  IF (APPLE)
+@@ -145,7 +107,9 @@ IF (WITH_GPU OR WITH_ROCM)
+     endif()
+
+     IF (WITH_GPU)
+-        CUDA_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
++        # CUDA_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
++        enable_language(CUDA)
++        add_library(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
+     ELSE()
+         HIP_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
+         TARGET_LINK_LIBRARIES(warprnnt PUBLIC ${ROCM_HIPRTC_LIB})
```


## 2.4 编译
`cd Paddle && mkdir build`

```bash
source /framework/m100/env_xcuda.sh
# export {http,https}_proxy=http://10.63.229.53:8891
export {http,https}_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
export no_proxy="baidu.com,bcebos.com,baidu-int.com,.baidu.com,.bcebos.com,.baidu-int.com,localhost,127.0.0.1,10.*,172.16.*"

# xcuda 11.7
# /opt/cmake-3.22.2/bin/cmake .. \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DWITH_GPU=ON \
#   -DWITH_PYTHON=ON \
#   -DWITH_TESTING=OFF \
#   -DWITH_NCCL=OFF \
#   -DNCCL_ROOT=OFF \
#   -DWITH_DISTRIBUTE=OFF \
#   -DWITH_CINN=OFF \
#   -DWITH_CUDNN_FRONTEND=OFF \
#   -DCUDNN_ROOT=${CUDNN_ROOT} \
#   -DWITH_MKL=ON \
#   -DCMAKE_CUDA_ARCHITECTURES=80 \
#   -DCUDA_ARCH_NAME=Ampere \
#   -DPYTHON_EXECUTABLE=/root/miniconda/envs/python310_torch25_cuda/bin/python3.10

# xcuda 12.8 + nccl + distribute
/opt/cmake-3.22.2/bin/cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GPU=ON \
  -DWITH_PYTHON=ON \
  -DWITH_TESTING=OFF \
  -DWITH_NCCL=ON \
  -DNCCL_ROOT=${NCCL_ROOT} \
  -DWITH_DISTRIBUTE=ON \
  -DWITH_CINN=OFF \
  -DWITH_CUDNN_FRONTEND=OFF \
  -DCUDNN_ROOT=${CUDNN_ROOT} \
  -DWITH_MKL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_ARCH_NAME=Ampere \
  -DPYTHON_EXECUTABLE=/root/miniconda/envs/python310_torch25_cuda/bin/python3.10

make -j$(nproc)

```
`bash build.sh`



**由于当前编译架构是sm_80, 所以distribute不会编译deep_ep**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=c63e2bb85f0249e2bba916394e81af40&docGuid=TgOjbAniE11KQ-)
# paddle xtrans 升级 cuda_12.8 踩坑
## cuda版本大于12.6时找libcudnn.so.9，但xtrans里没有，需要链接
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b5d6794c50ee403fb0105732d7cac69c&docGuid=TgOjbAniE11KQ-)
```bash
cd $XCUDA_PATH/targets/x86_64-linux/lib
ln -s libcudnn.so.8.9 libcudnn.so.9
```


## 不认识`trap和exit `内联汇编指令
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=a5b587fc90614577b5c7bc3781dba285&docGuid=TgOjbAniE11KQ-)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=45627e59aed54825bc2daa9c366b03ee&docGuid=TgOjbAniE11KQ-)
**需要xtrans里面转**，目前先进行以下修改绕过去了

```cpp
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)               \
  do {                                                             \
    if (!(_IS_NOT_ERROR)) {                                        \
      printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n", \
             __FILE__,                                             \
             __LINE__,                                             \
             #_IS_NOT_ERROR,                                       \
             ##__VA_ARGS__);                                       \
      __builtin_trap();                                                \
    }                                                              \
  } while (0)
```


```cpp
__device__ __forceinline__ void ThreadTrap() {
    asm volatile("trap;");
}

__device__ __forceinline__ void ThreadExit() {
    // asm volatile("exit;");
    asm volatile("s_endpgm");
}

```


## `__shfl_down`**歧义**
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3ee39f84821244c1b6311374c18c8986&docGuid=TgOjbAniE11KQ-)
这个是 `__shfl_down`** 对 **`__xpu_bfloat16`** 没有唯一匹配重载**，所以编译器不知道该选哪个版本。

报错里显示：

```cpp
__xpu_bfloat16 t = __shfl_down(x, offset);
```
而源码里原本是：

```cpp
__nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
```
说明 xtrans/M100 的头文件把 CUDA 的 `__nv_bfloat16` / shuffle API 映射成了自己的 `__xpu_bfloat16` / `__shfl_down`。但 `__xpu_bfloat16` 可能同时能隐式转成 `float`、`half`、整数底层类型等，导致 `__shfl_down(...)` 有多个候选函数都可用，于是 ambiguous。

于是这个函数里改成：

```cpp
__nv_bfloat16 t = static_cast<__nv_bfloat16>(__shfl_down_sync(mask, static_cast<float>(x), offset));
```
Paddle 里类似封装也这么做：`paddle/phi/backends/gpu/cuda/cuda_device_function.h:65` 对 `phi::dtype::bfloat16` 的 shuffle 就是先 `static_cast<float>(val)` 再 `__shfl_down_sync`。

所以这个问题本质是：**xtrans 对 bfloat16 的 shuffle 重载/类型转换适配不完善，直接传 **`__xpu_bfloat16`** 会歧义；显式转成 **`float`** 可以消除歧义。**



还有其他3个类似的，先改掉绕过去，后边xtrans修复后再放开：

```cpp
template <int Width = 32>
__device__ __nv_bfloat16 WarpReduceMax(__nv_bfloat16 x) {
  constexpr unsigned mask = (uint64_t(1) << Width) - 1;
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    __nv_bfloat16 t = static_cast<__nv_bfloat16>(__shfl_down_sync(mask, static_cast<float>(x), offset));
    // __nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
    x = BF16_MAX(x, t);
  }
  return x;
}
```
```cpp
template <typename T, bool use_pow2_scale, bool using_ue8m0_scale>
__device__ void ComputeRowScale(const v64_t<T> x[8],
                                float block_scale[128],
                                T *shm,
                                const float epsilon) {
  for (uint32_t i = 0; i < 8; i++) {
    // reduce [32, (4)] => [32]
    T local_max;
    for (uint32_t j = 0; j < 4; j++) {
      T other = device_abs(x[i].val[j]);
      local_max = j == 0 ? other : device_max(local_max, other);
    }

    // reduce [32] => [1]
    T warp_max = local_max;
    for (uint32_t offset = 16; offset > 0; offset /= 2) {
      T other = static_cast<T>(__shfl_down_sync(0xFFFFFFFF, static_cast<float>(warp_max), offset));
      warp_max = device_max(warp_max, other);
    }
    if (threadIdx.x == 0) {
      shm[i * 16 + threadIdx.y] = warp_max;
    }
  }
```
```cpp
template <int VecSize, bool Power2Scaling>
__device__ void ComputeScaleAndWrite(__nv_bfloat16 *data,
                                     float *scale,
                                     float *scale_out,
                                     int64_t local_scale_id,
                                     int64_t dest_scale_row,
                                     int64_t dest_scale_col,
                                     int64_t scale_row_num,
                                     int64_t scale_col_num) {
  // -------------------------------------------------------------------------
  // Step 1: Compute local maximum within each thread's vector
  // -------------------------------------------------------------------------
  __nv_bfloat16 local_max = __float2bfloat16(-INFINITY);
  for (int i = 0; i < VecSize; ++i) {
    __nv_bfloat16 val = BF16_ABS(data[i]);
    local_max = BF16_MAX(val, local_max);
  }

  // -------------------------------------------------------------------------
  // Step 2: Reduce maximum across warp using shuffle operations
  // -------------------------------------------------------------------------
  static_assert(VecSize >= 4,
                "VecSize must be at least 4 to avoid cross-warp reduction");
  static_assert(TileSize >= VecSize && TileSize % VecSize == 0,
                "TileSize must be >= VecSize and a multiple of VecSize");

  __nv_bfloat16 global_max = local_max;
  constexpr int group_size = TileSize / VecSize;  // Elements per thread group
  const int lane_id = threadIdx.x % WarpSize;
  const int group_id = lane_id / group_size;
  const int group_lane = lane_id % group_size;
  const unsigned mask =
      (1u << ((group_id + 1) * group_size)) - (1u << (group_id * group_size));

  // Parallel reduction within each group
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    // __nv_bfloat16 other = __shfl_down_sync(mask, global_max, stride);
    __nv_bfloat16 other = static_cast<__nv_bfloat16>(__shfl_down_sync(mask, static_cast<float>(global_max), stride));

    global_max = BF16_MAX(other, global_max);
  }
```


## **Paddle dynload 封装名和 cuBLAS 真实符号名不一致**。
`paddle/phi/backends/dynload/cublas.h` 里注册的是：

```
C/C++
phi::dynload::cublasSgemm_v2
```
不是：

```
C/C++
phi::dynload::cublasSgemm
```
你这个文件里调用了：

```
C/C++
phi::dynload::cublasSgemm(...)
```
所以编译器说：

```
text
no member named 'cublasSgemm' in namespace 'phi::dynload'did you mean simply 'cublasSgemm'?
```
意思是：

* 全局命名空间里有 `::cublasSgemm`，来自 xtrans 的 `cublas_v2.h`
* 但 `phi::dynload` 命名空间里没有 `cublasSgemm`
* `phi::dynload` 里只有 `cublasSgemm_v2`

修法一般是把 `batched_gemm.cu:126` 改成：

```cpp
CUBLAS_CALL(phi::dynload::cublasSgemm_v2(cublas_handle,
                                         transpose_b,
                                         transpose_a,
                                         m,
                                         n,
                                         k,
                                         &alpha,
                                         b,
                                         ldb,
                                         a,
                                         lda,
                                         &beta,
                                         c,
                                         c_cols));
```


## xcuda 12.8中的xtrans nvcc 会自动添加原生 GPU 架构 (sm_90) 和 xcn 架构，无视**指定的 **`-gencode`** **参数。
### 5.1 测试
```cpp
#define STR(x) #x
#define XSTR(x) STR(x)
#if defined(__CUDA_ARCH__)
#pragma message("__CUDA_ARCH__ = " XSTR(__CUDA_ARCH__))
#if __CUDA_ARCH__ >= 900
#pragma message(">= sm_90 detected!")
#endif
#endif
__global__ void test_kernel() {}
int main() { return 0; }
```
`/framework/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/bin/nvcc -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -c test_arch.cu -o test_arch.o`

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8a4ab01033e44d1da23909a792951e38&docGuid=TgOjbAniE11KQ-)
### 5.2 解决方案
用 `--cuda-gpu-arch` 代替 `-gencode` 就不会多出 `sm_90`。

`/framework/m100/xtrans_cuda_12.8_ubuntu2004_x86_64_mars/bin/nvcc --cuda-gpu-arch=sm_80 -c test_arch.cu -o test_arch.o`

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0986989b7a9b49e2adbcc0c2254b1407&docGuid=TgOjbAniE11KQ-)
修改`Paddle/cmake/cuda.cmake`

```diff
diff --git a/cmake/cuda.cmake b/cmake/cuda.cmake
index 1b94d00812..c6dcd1a025 100644
--- a/cmake/cuda.cmake
+++ b/cmake/cuda.cmake
@@ -228,12 +228,12 @@ function(select_nvcc_arch_flags out_variable out_arch_bin)
     if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
       # User explicitly specified PTX for the concrete BIN
       string(APPEND nvcc_flags
-             " -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1}")
+             " --cuda-gpu-arch=sm_${CMAKE_MATCH_1}")
       string(APPEND nvcc_archs_readable " sm_${CMAKE_MATCH_1}")
       string(APPEND nvcc_archs_bin_list " ${CMAKE_MATCH_1}")
     else()
       # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
-      string(APPEND nvcc_flags " -gencode arch=compute_${arch},code=sm_${arch}")
+      string(APPEND nvcc_flags " --cuda-gpu-arch=sm_${arch}")
       string(APPEND nvcc_archs_readable " sm_${arch}")
       string(APPEND nvcc_archs_bin_list " ${arch}")
     endif()
@@ -242,7 +242,7 @@ function(select_nvcc_arch_flags out_variable out_arch_bin)
   # Tell NVCC to add PTX intermediate code for the specified architectures
   foreach(arch ${cuda_arch_ptx})
     string(APPEND nvcc_flags
-           " -gencode arch=compute_${arch},code=compute_${arch}")
+           " --cuda-gpu-arch=compute_${arch}")
     string(APPEND nvcc_archs_readable " compute_${arch}")
   endforeach()
```


## 链接时符号缺失
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=08dd94818aff462e9beffd232bb67942&docGuid=TgOjbAniE11KQ-)
## 根本原因
你的项目（PaddlePaddle）在 Linux 上使用 **shared CUDA runtime**（`CUDA_USE_STATIC_CUDA_RUNTIME OFF` + `--cudart shared`），但 **没有任何地方把 **`-lcudart`** 加到最终的链接命令中**。

所有报错的 CUDA 符号（`cudaStreamSynchronize`、`cudaMalloc`、`cudaMemcpy` 等）都是直接调用 CUDA Runtime API，**没有**通过 dynload 包装层（dynload 只包装了 driver API 和第三方库如 cublas/cudnn，不包装 runtime）。

## 具体问题链路
1. `phi_core`/`phi_gpu` 通过 `nv_library` 编译（`paddle/phi/CMakeLists.txt:174-181`），但 `nv_library` 函数（`cmake/generic.cmake:650-690`）**没有显式链接 **`cudart`——它依赖 CMake 的隐式传播机制。
2. `standalone_executor` 通过 `cc_library` 编译（`framework/new_executor/CMakeLists.txt:61`），`cc_library` 完全不感知 CUDA。
3. 最终链接目标（如 `libpaddle.so`、`paddle_inference_shared.so`、`standalone_executor_test`）都只链接了 `os_dependency_modules`，而这个变量在 Linux 上**是空的**（只在 Windows 上有值）。
4. CUDA runtime 依赖在 `STATIC -> STATIC -> SHARED` 的长链路中丢失了。

## 修复方案
最简单的修复是在最终链接目标处加上 `-lcudart`。有三种选择：

**方案 A（推荐，全局修复）** — 在顶层 `CMakeLists.txt` 中把 CUDA 库加入全局依赖：

在 `Paddle/CMakeLists.txt` 的 `if(WITH_GPU)` 块（约 522 行）后面加：

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 6c01573ae2..53af480407 100755
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -534,6 +534,7 @@ if(WITH_GPU)
   if(NOT WIN32)
     include(cupti)
   endif()
+  set_property(GLOBAL APPEND PROPERTY OS_DEPENDENCY_MODULES ${CUDA_LIBRARIES})
 endif()

 if(WITH_ROCM)
```


## `__ffs` ambiguous和`__lane_id` 名字冲突
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ffc228d299044afcbc335a749b93edf5&docGuid=TgOjbAniE11KQ-)
这是两个 **xtrans/M100 兼容 CUDA 12.8 libcudacxx 头文件不完整** 的问题，仍然集中在 `<cuda/barrier>` / `<cuda/pipeline>` 这条路径。

### `__ffs` ambiguous
这里调用的是：

```
__ffs(__source_address | __destination_address | __size)
```
其中 `__source_address` / `__destination_address` 是 `uintptr_t`，在 64 位平台上通常是 `unsigned long` 或 `unsigned long long`。

但 xtrans 只提供了：

```cpp
__ffs(unsigned int)__ffs(int)
```
没有精确匹配 64 位无符号类型的 `__ffs(...)`，于是 `uintptr_t` 转 `unsigned int` 和转 `int` 都可行，编译器不知道选哪个，就报歧义。

可以补一个 64 位 overload，例如在 `xtdk_device_functions.h` 里加：

```cpp
__device__ static inline unsigned int __ffs(unsigned long long int input) {  return __ffsll(input);}
```
如果 `uintptr_t` 在这个环境是 `unsigned long`，还要补：

```cpp
__device__ static inline unsigned int __ffs(unsigned long int input) {  return __ffsll(static_cast<unsigned long long int>(input));}
```
### 7.2. `__lane_id` 名字冲突
报错展开后变成了：

```cpp
uint32_t __lane_id;__lane_id = __lane_id();
```
这里 libcudacxx 的 `cuda/pipeline` 里函数名叫 `__lane_id()`，局部变量也叫 `__lane_id`。NVIDIA 编译器能处理原始宏分支/内联 asm，会把它当成 PTX 指令（`mov.u32 rX, %laneid;`）但 xtrans 不行，宏展开后把它变成了`__lane_id = __lane_id();` 且这样会递归调用自己，与原语义不符

```cpp
uint32_t __lane_id;   // 变量__lane_id();          // 现在解析成“调用变量”，所以报不是函数
```
这个应该把 `cuda/pipeline` 里__lane_id()函数替换为__builtin 的内联汇编

```cpp
                    uint32_t __lane_id;
                    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(__lane_id));
                    return __lane_id;
```
改成：

```cpp
return __builtin_xcn_mbcnt_hi(-1, __builtin_xcn_mbcnt_lo(-1, 0));
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=32728c0a5050447fb9cc5296e3b94468&docGuid=TgOjbAniE11KQ-)


# Paddle 修改：
~~cuda 11.7 修改Paddle~~

```diff
diff --git a/cmake/external/warpctc.cmake b/cmake/external/warpctc.cmake
index 17ef70b4a0..ad6e24aaa8 100644
--- a/cmake/external/warpctc.cmake
+++ b/cmake/external/warpctc.cmake
@@ -124,6 +124,9 @@ ExternalProject_Add(
   #BUILD_ALWAYS    1
   CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
+             -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
+             -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
+             -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
              -DCMAKE_C_FLAGS=${WARPCTC_C_FLAGS}
              -DCMAKE_C_FLAGS_DEBUG=${WARPCTC_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_RELEASE=${WARPCTC_C_FLAGS_RELEASE}
diff --git a/cmake/external/warprnnt.cmake b/cmake/external/warprnnt.cmake
index ce4b43343a..46a5793a89 100644
--- a/cmake/external/warprnnt.cmake
+++ b/cmake/external/warprnnt.cmake
@@ -125,6 +125,9 @@ ExternalProject_Add(
   #BUILD_ALWAYS    1
   CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
+             -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
+             -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
+             -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
              -DCMAKE_C_FLAGS=${WARPRNNT_C_FLAGS}
              -DCMAKE_C_FLAGS_DEBUG=${WARPRNNT_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_RELEASE=${WARPRNNT_C_FLAGS_RELEASE}
diff --git a/paddle/phi/kernels/funcs/blas/blas_impl.cu.h b/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
index e9a033a24c..0c02292370 100644
--- a/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
+++ b/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
@@ -196,7 +196,8 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void GETRF_BATCH(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
+    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented( "GETRI_BATCH is not supported by xtrans."));
   }

   template <typename... ARGS>
@@ -218,8 +219,8 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void TRSM_BATCH(ARGS... args) {
-  //  PADDLE_THROW(phi::errors::Unimplemented("SmatinvBatched is not supported by xtrans."));
-  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("SmatinvBatched is not supported by xtrans."));
+    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
   }

   template <typename... ARGS>
@@ -306,7 +307,7 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void GETRI_BATCH(ARGS... args) {
-    PADDLE_THROW(phi::errors::Unimplemented("GETRI_BATCH is not supported by xtrans. upgrade"));
+    PADDLE_THROW(phi::errors::Unimplemented("GETRI_BATCH is not supported by xtrans."));
     // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetriBatched(args...));
   }

@@ -323,8 +324,8 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void TRSM_BATCH(ARGS... args) {
-    // PADDLE_THROW(phi::errors::Unimplemented("DmatinvBatched is not supported by xtrans."));
-         PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("DmatinvBatched is not supported by xtrans."));
+         // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
   }

   template <typename... ARGS>
diff --git a/patches/warpctc/CMakeLists.txt.cuda.patch b/patches/warpctc/CMakeLists.txt.cuda.patch
index 9cf204e95a..5fcac855c4 100644
--- a/patches/warpctc/CMakeLists.txt.cuda.patch
+++ b/patches/warpctc/CMakeLists.txt.cuda.patch
@@ -55,3 +55,14 @@
  ENDIF()

  IF (APPLE)
+@@ -160,7 +123,9 @@ IF (WITH_GPU OR WITH_ROCM)
+         CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/.ctc_entrypoint.cu src/reduce.cu)
+     else()
+         IF (WITH_GPU)
+-            CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
++            # CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
++            enable_language(CUDA)
++            add_library(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
+         ELSE()
+             HIP_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
+             TARGET_LINK_LIBRARIES(warpctc PUBLIC ${ROCM_HIPRTC_LIB})
diff --git a/patches/warprnnt/CMakeLists.txt.cuda.patch b/patches/warprnnt/CMakeLists.txt.cuda.patch
index 16967534de..23d5315da6 100644
--- a/patches/warprnnt/CMakeLists.txt.cuda.patch
+++ b/patches/warprnnt/CMakeLists.txt.cuda.patch
@@ -54,3 +54,14 @@
  ENDIF()

  IF (APPLE)
+@@ -145,7 +107,9 @@ IF (WITH_GPU OR WITH_ROCM)
+     endif()
+
+     IF (WITH_GPU)
+-        CUDA_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
++        # CUDA_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
++        enable_language(CUDA)
++        add_library(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
+     ELSE()
+         HIP_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
+         TARGET_LINK_LIBRARIES(warprnnt PUBLIC ${ROCM_HIPRTC_LIB})
```


~~cuda 11.8 修改Paddle~~

```diff

```


# 编译xnccl踩坑
# 找不到nccl.h 头文件
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8d6092b128da4ab8a21d12469ed5ea72&docGuid=YM1fzjiA2pWAbz)
原因dgc 是用 原始 make 编译的（不是 CMake），它有自己的 Makefile，完全看不到 Paddle 的 include_directories，通过指定`NCCL_INCLUDE`解决

```bash
  BUILD_COMMAND make -j${NPROC} NCCL_INCLUDE=${NCCL_ROOT}/include
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=35f4401e4e1f43cab44801a63a1e1831&docGuid=YM1fzjiA2pWAbz)


# 找错了nccl.h的路径
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1c0adcd12e294c668ecbc5d05c8329a8&docGuid=YM1fzjiA2pWAbz)
**xtrans_cuda 自带了一个 **`nccl.h`，而且它的 include 路径被**优先加入**了：

```
configure.cmake:137  →  include_directories(${CUDA_TOOLKIT_INCLUDE})
                        = /framework/m100/xtrans_cuda_.../targets/x86_64-linux/include/
                        （这个目录里有 nccl.h！）

CMakeLists.txt:510    →  include(nccl) → include_directories(${NCCL_INCLUDE_DIR})
                        = /framework/m100/nccl/output/.../include/
```
CMake 的 `include_directories` 是**追加**模式，编译器按顺序搜索，xtrans_cuda 的路径在前面，所以它自带的 `nccl.h` 先被找到。

修复方法：把 NCCL 的路径插到最前面（用 `BEFORE`）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=43547669c5dd475e98f3437dcc1e3a45&docGuid=YM1fzjiA2pWAbz)
```
if(WITH_NCCL)
  set(NCCL_ROOT
      "/usr"
      CACHE PATH "NCCL ROOT")
  find_path(
    NCCL_INCLUDE_DIR nccl.h
    PATHS ${NCCL_ROOT} ${NCCL_ROOT}/include ${NCCL_ROOT}/local/include
          $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include $ENV{NCCL_ROOT}/local/include
    NO_DEFAULT_PATH)
  include_directories(BEFORE ${NCCL_INCLUDE_DIR})
```




# phi路径大量fp16的函数重定义
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=07c0a0ecd3e5485b8088120639be0f41&docGuid=YM1fzjiA2pWAbz)
~~这是~~~~**两套 fp16 实现冲突**~~~~：~~

|~~~~|~~xtrans_cuda 编译器自带 (~~`xtdk_fp16_gcc.h`~~)~~|~~NCCL 自带 (~~`cuda_fp16.h`~~)~~|
|-|-|-|
|~~**包含方式**~~|~~clang 编译器自动内置引入（无需 ~~`#include`~~）~~|~~~~`nccl.h:11`~~ 主动 ~~`#include "cuda_fp16.h"`~~~~|
|~~**头文件保护宏**~~|~~~~`#pragma only`~~~~|~~~~`BKCL_CUDA_FP16_COMPAT_H_`~~~~|
|~~**冲突原因**~~|~~两者的保护宏不同，互相不认识，导致都生效~~|~~同上~~|

~~**核心问题**~~~~：NCCL 的 ~~`cuda_fp16.h`~~ 是一个 ~~~~**compat 层**~~~~（给没有 CUDA toolkit 的环境用的），它自己重新定义了 ~~`__half`~~、~~`__float2half`~~ 等。但 xtrans_cuda 已经通过编译器内置路径提供了这些定义，两者撞车了。~~

~~~~

~~修复思路：NCCL 自带的 cuda_fp16.h 是个 compat 层（给没有 CUDA toolkit 的环境用的）。你的环境已经有 xtrans_cuda 提供的完整 fp16 实现，所以需要在 #include <nccl.h> 之前定义 NCCL 的头文件保护宏，跳过它自带的~~

```
#define BKCL_CUDA_FP16_COMPAT_H_
#include <nccl.h>
```
**和5一样，直接使用一个最终解决方案**

# `CudnnDataType`，`PoolingMode`等类型缺失
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3de40ec2f6f64a5f9b9c3e5509ffa1e8&docGuid=YM1fzjiA2pWAbz)
当前为了简化，不开WITH_CUDNN_FRONTEND的编译选项，但是Paddle代码本身有缺陷——`gpu_dnn.h` 在 **没有** `WITH_CUDNN_FRONTEND` 时也引用了这些类型，但定义被 `#ifdef` 守住了。这是 Paddle 代码的一个条件编译遗漏。

**总结修改**：在 `/framework/m100/Paddle/paddle/phi/core/platform/device/gpu/gpu_dnn.h` 中，将所有 cuDNN 相关的 `using` 声明用 `#ifdef WITH_CUDNN_FRONTEND ... #endif` 包起来。这样关闭 `WITH_CUDNN_FRONTEND` 时这些类型不会被引用，可以继续编译其他部分。

```
using DataLayout = phi::DataLayout;
#ifdef WITH_CUDNN_FRONTEND
using PoolingMode = phi::backends::gpu::PoolingMode;
......
using ScopedSpatialTransformerDescriptor =
    phi::backends::gpu::ScopedSpatialTransformerDescriptor;
#endif
#endif  // WITH_CUDNN_FRONTEND
```


# 5.fluid路径大量fp16的函数重定义
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=909057adcb1d438ca2c3e375caee55d8&docGuid=YM1fzjiA2pWAbz)
有**两条独立的 include 链路**都会引入 fp16 定义：

```
text
链路1（之前修的）: enforce.h:58 → dynload/nccl.h → nccl.h → NCCL的cuda_fp16.h  ✅ 已修复链路2（现在的）: enforce.h:22 → curand.h → xpu/xpu_fp16.h → xtdk_fp16_gcc.h  ← 先定义了 __half 等                                                    ↓                          （某处又引入了）NCCL的cuda_fp16.h              ← 再定义，冲突！
```
在单个文件里加 `#define` 治标不治本——所有经过 `enforce.h:22` 的 `.cc` 文件都会触发。最彻底的方案是**通过编译器全局定义这个宏**：

已在 `nccl.cmake:20` 添加全局编译器定义：

```
add_definitions("-DBKCL_CUDA_FP16_COMPAT_H_")
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=a8f861041963474288e7b9a1c9915cc2&docGuid=YM1fzjiA2pWAbz)


# Paddle 修改：
在上一个paddle xtrans 升级 cuda_12.8 踩坑 的Paddle修改后新增的

```diff
diff --git a/cmake/external/dgc.cmake b/cmake/external/dgc.cmake
index 579b7f2da8..af9bf1a647 100644
--- a/cmake/external/dgc.cmake
+++ b/cmake/external/dgc.cmake
@@ -76,7 +76,8 @@ ExternalProject_Add(
   URL_MD5 ${DGC_URL_MD5}
   PREFIX "${DGC_PREFIX_DIR}"
   CONFIGURE_COMMAND ""
-  BUILD_COMMAND make -j${NPROC}
+  # BUILD_COMMAND make -j${NPROC}
+  BUILD_COMMAND make -j${NPROC} NCCL_INCLUDE=${NCCL_INCLUDE_DIR}
   DOWNLOAD_DIR ${DGC_DOWNLOAD_DIR}
   SOURCE_DIR ${DGC_SOURCES_DIR}
   INSTALL_COMMAND
diff --git a/cmake/nccl.cmake b/cmake/nccl.cmake
index eaa7bd23fd..2ed893ddda 100644
--- a/cmake/nccl.cmake
+++ b/cmake/nccl.cmake
@@ -16,6 +16,8 @@ if(WITH_NCCL)
     PATHS ${NCCL_ROOT} ${NCCL_ROOT}/include ${NCCL_ROOT}/local/include
           $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include $ENV{NCCL_ROOT}/local/include
     NO_DEFAULT_PATH)
+  include_directories(BEFORE ${NCCL_INCLUDE_DIR})
+  add_definitions("-DBKCL_CUDA_FP16_COMPAT_H_")

   file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)

diff --git a/paddle/phi/core/platform/device/gpu/gpu_dnn.h b/paddle/phi/core/platform/device/gpu/gpu_dnn.h
index 3418089f8d..7c68e4a59d 100644
--- a/paddle/phi/core/platform/device/gpu/gpu_dnn.h
+++ b/paddle/phi/core/platform/device/gpu/gpu_dnn.h
@@ -22,6 +22,7 @@ namespace paddle {
 namespace platform {

 using DataLayout = phi::DataLayout;
+#ifdef WITH_CUDNN_FRONTEND
 using PoolingMode = phi::backends::gpu::PoolingMode;
 template <typename T>
 using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
@@ -40,6 +41,7 @@ using ScopedRNNTensorDescriptor = phi::backends::gpu::ScopedRNNTensorDescriptor;
 using ScopedSpatialTransformerDescriptor =
     phi::backends::gpu::ScopedSpatialTransformerDescriptor;
 #endif
+#endif  // WITH_CUDNN_FRONTEND

 }  // namespace platform
 }  // namespace paddle
```