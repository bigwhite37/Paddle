M100-Paddle适配调研

# 前期参考文档
[XCUDA-Toolkit MARS V1.1.1.1 发版测试报告](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/25PzPNBfap/N93m6P7_EWczGV?t=mention&mt=doc&dt=doc)

[Paddle编译](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/25PzPNBfap/5mLXkRr79QEYo2?t=mention&mt=doc&dt=doc)

[环境变量设置](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/25PzPNBfap/Z79MFpp_cCTfSa?t=mention&mt=doc&dt=doc)

[Paddle-CUDA化方案](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/4N358KYYvi/QRZy8HoUOtRzd8?t=mention&mt=doc&dt=doc)

[Mars V1.1软件栈产出发布](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/25PzPNBfap/Fz1r1p-gHiBtt5?t=mention&mt=doc&dt=doc)

[CUDA生态库](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/25PzPNBfap/Di_NeTT10zVP1x?t=mention&mt=doc&dt=doc)

[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/xpu/xpu-p800_install_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/xpu/xpu-p800_install_cn.html)

[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

# 机器 & 环境
## 2.1 踩坑记录
     （1）昆仑芯负责人给出的建议机器是优先P800，其次选用纯CPU机器，但要保证使用他们的Docker。

```shell
iregistry.baidu-int.com/isa/xtdk_ubuntu_2004_x86_64:v0.95
```
    （2）使用PaddleCloud加载上镜像会出现报错，与PaddleCloud值班人员沟通后，对方说只能加载专用镜像，开会对齐后建议使用物理机。

```shell
# 使用WebRelay链接，当前机器有问题，需要先ssh到其他厂内Linux机器再ssh跳转过去 https://webrelay.baidu-int.com/relay
ssh yq02-inf-sci-k8s-a100-aa2ni5-0065.yq02
ssh zwlt-node0868.bcc-zwlt.baidu.com
```
    （3）登录机器后发现docker load会报错，尝试docker login登录 / 换各种代理均不解决问题，最终求助昆仑芯QA帮忙上传bos下载使用。

```shell
wget https://klx-sdk-release-public.su.bcebos.com/DS_PD/docker/xtdk_ubuntu_2004_x86_64_v0.95.tar.gz
docker load -i xtdk_ubuntu_2004_x86_64_v0.95.tar.gz
```
    （4）下载后使用docker load数小时也没有结束，尝试使用gunzip先解压后加载，但是gunzip也无法正常结束，尝试出下方命令可以完成解压，但是解压后docker load仍然失败，怀疑文件损坏。

```shell
pv xtdk_ubuntu_2004_x86_64_v0.95.tar.gz | gunzip -c > xtdk_ubuntu_2004_x86_64_v0.95.tar
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=d596d6041f364092a2d6220897d4e8b7&docGuid=o8_CyK4_2B8NcQ)
## 2.2 实际机器使用步骤
（1）询问昆仑芯QA之前使用的机器，可在门神上申请使用，通过搜索ID进行审批

```shell
http://giano.baidu.com/dgweb/index.php?r=Privilege/personalInfo/DoorGodApply&nodeId=212748796&parentId=

zzjg-isa-ai-p800-klxnode04.zzjg
```
（2）docker可以使用如下方式使用

```shell
docker pull iregistry.baidu-int.com/isa/xtdk_ubuntu_2004_x86_64:v0.95

docker run -it --name paddle-xy -v $(pwd):/work  -v /home:/home -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi   -w=/work --shm-size=128G --network=host --privileged    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  iregistry.baidu-int.com/isa/xtdk_ubuntu_2004_x86_64:v0.95 /bin/bash
```
```shell
docker exec -it paddle-xy /bin/bash
```
(3)  xcuda软件包部署

```shell
wget https://klx-sdk-release-public.su.bcebos.com/mars_release/XTRANSCUDA/dev/latest/xtrans_cuda_11.7_ubuntu2004_x86_64_mars.tar.gz
tar -xvf xtrans_cuda_11.7_ubuntu2004_x86_64_mars.tar.gz

#设置XCUDA环境变量
export XCUDA_PATH=$PWD/xtrans_cuda_11.7_ubuntu2004_x86_64_mars
export PATH=${XCUDA_PATH}/bin:${PATH}
export CUDA_PATH=${XCUDA_PATH}
export LD_LIBRARY_PATH=${XCUDA_PATH}/lib64:${XCUDA_PATH}/lib:${LD_LIBRARY_PATH}
export LDFLAGS=-L$XCUDA_PATH/lib64/
export XTRANS_DIR=${XCUDA_PATH}
export CXX=$XCUDA_PATH/bin/clang++
export CUDNN_ROOT=$XCUDA_PATH/targets/x86_64-linux
export CUPTI_ROOT=$XCUDA_PATH/targets/x86_64-linux
export XMLIR_CUDNN_ENABLED=true
```
# Paddle编译步骤
（从易到难去实验，因为分布式当前未支持先去掉）

```shell
#开发机环境需要代理
export {http,https}_proxy=http://agent.baidu.com:8891
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout v3.1.0  # 切换到稳定分支（如 3.1.0）

git submodule sync
git submodule update --init --recursive

unset {http,https}_proxy
```
```shell
WITH_CUDNN_FRONTEND=OFF CUDAARCHS=all WITH_GPU=1 WITH_NCCL=0 python setup.py develop
WITH_CUDNN_FRONTEND=OFF CUDAARCHS=all WITH_GPU=1 WITH_NCCL=0 python setup.py develop > build.log 2>&1
```
```shell
# 如遇到报错，经常需要先行删除build再重新编译
rm -rf build/

# 如果遇到多核编译出现一堆报错的情况，可暂时禁用多核编译
export MAX_JOBS=1

# 查找当前目录下是否包含“”中的字符
find .|xargs grep -ri "warpSize"
```


# 4 问题汇总
## 4.1 CMake
（1）问题现象：自动检测GPU架构失败，导致空列表操作。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=d90f0f3b267a42138472c726da201b20&docGuid=o8_CyK4_2B8NcQ)


解决方案：手动补充架构参数，避免空列表。

```shell
vim cmake/cuda.cmake

# 80行新增下方代码
list(APPEND nvcc_out "8.0")
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b7144388648f4668b825b498f9ce3e77&docGuid=o8_CyK4_2B8NcQ)


（2）问题现象：没找到cudnn可供编译。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=18a7bcfd682d45d39362c8716f74e6ab&docGuid=o8_CyK4_2B8NcQ)


解决方案：在cmake/cudnn.cmake中寻找cudnn.h时没找到，把正确的路径加入到环境变量中，同时新增一个查找位置。

```shell
# 加入以下环境变量
export CUDNN_ROOT=$XCUDA_PATH/targets/x86_64-linux
export CUPTI_ROOT=$XCUDA_PATH/targets/x86_64-linux

# 进入该文件增加一个查找路径，cudnn.h在${CUDNN_ROOT}/include/cudnn_api路径下
vim cmake/cudnn.cmake

# 21行新增下方代码
${CUDNN_ROOT}/include/cudnn_api
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=411f195177234e80be8dcbc6ec1dab6c&docGuid=o8_CyK4_2B8NcQ)


（3）问题现象：缺少patchelf包。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b351dfbfce7b45bab9a16633c6b6d38a&docGuid=o8_CyK4_2B8NcQ)
解决方案：安装即可。

```shell
apt-get install -y patchelf
```


## 4.2 三方库编译
(1) 问题现象：make生成watpctc与warprnnt中的.so文件出错

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ee0e95a4db074710bfa03d461cc81847&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* CMake 在编译 CUDA 源文件（reduce.cu）时，生成了一个名为 NVCC-depend 的中间目标。这个步骤本应只是做依赖扫描/设备链接，不应该去链接一个可执行文件。
* 但是当前 CUDA 编译链路走的是 /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/bin/clang++，它按“链接可执行文件”的方式去处理了单个对象文件 reduce.cu-1488.o，于是链接器找不到 main，就报错了。
* 这一类问题通常发生在项目使用了 CMake 的旧版 FindCUDA/cuda_add_library 逻辑（假定 NVCC），但你用的是 Clang 的 CUDA 前端或自定义工具链，导致 CMake 生成的“NVCC 依赖/设备链接”步骤与实际编译器不匹配。

解决方案：把第三方 warpctc 的 CMakeLists.txt 里 cuda_add_library/cuda_compile 等 NVCC 特定宏替换为 CMake 原生 CUDA 语言支持（CMake >= 3.18）：直接用 enable_language(CUDA)、add_library()。

将CUDA_ADD_LIBRARY替换为add_library，前方需要enable CUDA。

```shell
# line 126
vim third_party/warpctc/CMakeLists.txt

            enable_language(CUDA)
            add_library(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)

ii
# line 109
vim third_party/warprnnt/CMakeLists.txt

        enable_language(CUDA)
        add_library(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8fabb3184eb3470d920960a0cb837efa&docGuid=o8_CyK4_2B8NcQ)


（2）问题现象：Clang编译遇到“依赖基类的成员需要显式限定”

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6df12c3d05a8455aa734795fcbec5457&docGuid=o8_CyK4_2B8NcQ)
问题根因： Clang 系的 CUDA 转译器（xtrans_cuda，目标架构 xcn，houyi-device-only）去编译 flash-attn 的 Cutlass/CUTE 代码时撞上了“依赖基类的成员需要显式限定”的两阶段查找问题。

解决方案：子类模板里用时没有加 this->，加入即可。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=19900a34dd0344c8ab9f3cf6dc56e5d8&docGuid=o8_CyK4_2B8NcQ)


（3）问题现象，cutlas的早期已知bug，参考[https://github.com/NVIDIA/cutlass/issues/1603](https://github.com/NVIDIA/cutlass/issues/1603)

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=cd5e4b19332f4deb80c7c53e07457101&docGuid=o8_CyK4_2B8NcQ)
解决方案：先尝试了升级flash_attention下cutlas的版本可以解决，但是担心引入未知问题，直接修改set_slice3x3

```shell
# 方式一
cd third_party/flashattn/csrc/flash_attn_with_bias_and_mask/cutlass/
git checkout v4.0.0

#方式二 直接用sed替换，注意路径
CUTLASS_MATRIX_H="/home/xuanyuan/Paddle/third_party/flashattn/csrc/flash_attn_with_bias_and_mask/cutlass/include/cutlass/matrix.h"
sed -i 's/set_slice3x3/set_slice_3x3/g' "$CUTLASS_MATRIX_H"
```


(4)  FA问题较多，暂时跳过

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8c60c03eae3d4f5299834add991889b1&docGuid=o8_CyK4_2B8NcQ)
```shell
#先尝试单独跳，问题很多
# vim /home/xuanyuan/Paddle/third_party/flashattn/csrc/CMakeLists.txt

# 296   328   550   582


#整个FA不编译
#584
vim cmake/third_party.cmake
AND 0
```


## 4.3 paddle包编译
(1) 缺少xcn下的文件

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=940da2cbc80b4363bc32b849106b6d01&docGuid=o8_CyK4_2B8NcQ)
问题根因：没有把xtrans下的路径include进phi库中

解决方法：

```shell
# line 188
vim paddle/phi/CMakeLists.txt

target_compile_options(phi_core PRIVATE -iquote/home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/lib/clang/19/include)
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=4c2eeba71822417eb3ff48ea5a379fd4&docGuid=o8_CyK4_2B8NcQ)


(2) XCUDA问题

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=fc6ad06c90a14fd987dac7c144df6841&docGuid=o8_CyK4_2B8NcQ)
解决方案：

```shell
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/xpudnn_cuda_patch.h
```
删除以下三行

```c++
#define cudaFree(A) ;
#define cudaMalloc (A,B);
#define cudaMemcpy (A,B,C,D) 0
```


(3) C++ 预处理器宏冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6999b7ee4d9d44f29cf28a6573f4cc0a&docGuid=o8_CyK4_2B8NcQ)
根因分析：

* **PaddlePaddle 的动态加载机制（dynload）：**PaddlePaddle 为了解耦，在运行时动态加载 cuDNN 库（通过 `dlopen` / `dlsym`）。为了避免手写几百个函数的加载代码，它定义了一个宏 `CUDNN_DNN_ROUTINE_EACH_FRONTEND`，里面罗列了所有需要的 cuDNN 函数名（其中就包括 `cudnnGetStream`）。然后通过包裹宏 `DECLARE_DYNAMIC_LOAD_CUDNN_WRAP` 批量生成函数指针。展开后的代码大致长这样：`DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(cudnnGetStream)`
* **NVIDIA cuDNN 头文件的变动：**在你当前使用的 cuDNN 版本（很可能是 cuDNN v9 或者特定的高版本 v8）中，NVIDIA 的官方头文件 `cudnn.h` 将 `cudnnGetStream`**从一个普通的 C 函数改成了一个带参数的宏（Function-like Macro）**，形式类似于 `#define cudnnGetStream(handle, stream) ...`，它强制要求传入 2 个参数。
* **两头相撞（报错发生）：**当 C++ 编译器预处理到 PaddlePaddle 的 `DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(cudnnGetStream)` 时，它看到了 `cudnnGetStream` 这个词。因为 NVIDIA 头文件已经把它定义成了一个需要 2 个参数的宏，但这里它要么没带括号，要么在展开时被识别为只传入了 1 个参数，编译器就会立刻抛出 `requires 2 arguments, but only 1 given` 的错误。

解决方案1：

通过 `#undef` 临时解除宏（快速 Hack）,但是会导致诸多问题，放弃使用

```shell
# 178
vim /home/xuanyuan/Paddle/paddle/phi/backends/dynload/cudnn.h

#ifdef cudnnGetStream
#undef cudnnGetStream
#endif
```
解决方案：

我理解这里需要把cudnnGetStream宏的参数去掉;

```shell
//#define cudnnGetStream(handle, stream) xpudnnGetStream(handle, (XPUStream*)(stream))
#define cudnnGetStream xpudnnGetStream
```
并在xpudnnGetStream实现的地方做重载，当前是先在使用过程中把输入进行强转。

```shell
vim /home/xuanyuan/Paddle/build/third_party/cudnn-frontend/src/extern_cudnn_frontend/include/cudnn_f
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=9d51cac1b8a7443aa7b01d79987ab979&docGuid=o8_CyK4_2B8NcQ)


(4) 异构计算代码转换工具（Translation Tool）带来的类型不匹配

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=c7fd71b1b1e14e929ee65debe4c5f0e4&docGuid=o8_CyK4_2B8NcQ)
错误根因：

* **宏替换/类型映射不彻底：**`xtrans` 工具在转换 `cuda_runtime.h` 头文件时，把 `cudaMallocAsync` 函数的第四个参数类型从原生的 `cudaStream_t` 替换成了你们目标硬件的流类型 `XPUStream`（它的底层其实是个 `void*`）。
* **底层调用的函数未适配：** 在 `cudaMallocAsync` 函数内部，它调用了更底层的 `::cudaMallocFromPoolAsync`。问题在于，这个底层函数**依然要求传入原始的 **`cudaStream_t`**（底层是 **`CUstream_st*`**）**。
* **C++ 的严格类型检查：**C++ 编译器发现你要把一个 `void*` (XPUStream) 强塞给一个 `CUstream_st*` (cudaStream_t)，这在 C++ 中属于非法隐式转换，因此直接报错。

尝试修改1:

强制转换reinterpret_cast<cudaStream_t>之后仍然报错

return ::cudaMallocFromPoolAsync(ptr, size, memPool, reinterpret_cast<cudaStream_t>(stream));

错误根因-修改1:

虽然加上了 `reinterpret_cast<cudaStream_t>`，但编译器**依然**认为这个表达式的结果是 `XPUStream {aka void*}`。

这暴露出一个关键信息：在 `xtrans` 工具的头文件环境里，`cudaStream_t`** 这个词已经被 **`#define`** 或者 **`typedef`** 强行重新定义成了 **`XPUStream`**（也就是 **`void*`**）**。

所以你写的 `reinterpret_cast<cudaStream_t>(stream)` 在编译器眼里实际上变成了 `reinterpret_cast<void*>(stream)`。转了等于没转，底层真正的 NVIDIA 函数 `::cudaMallocFromPoolAsync` 依然收不到它想要的 `CUstream_st*` 类型。

修改方案：

既然 `cudaStream_t` 这个名字已经被污染了，我们需要**绕过这个别名，直接强转为最底层的原生指针类型 **`CUstream_st*`。

```shell
# line648
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cuda_runtime.h

return ::cudaMallocFromPoolAsync(ptr, size, memPool, reinterpret_cast<CUstream_st*>(stream));
```


（5）数据类型相关

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=068a9e47340f49f692664f12771555ce&docGuid=o8_CyK4_2B8NcQ)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=57f16ab276a04c8bbdb5c578c1c44e15&docGuid=o8_CyK4_2B8NcQ)
错误根因：

* **先入为主的 XPU 定义：** 在 `/home/.../xpu/refactor/util/float16.h` 的第 327 行，昆仑芯（XPU）底层的库已经定义了一个专门的数据结构：`struct float16 { ... };`。
* **暴力的宏替换补丁：** 在 `xtrans` 工具生成的补丁头文件 `xpudnn_cuda_patch.h` 第 93 行，工具为了强行让 CUDA 代码里的 `__half` 类型兼容 XPU，写了一句非常暴力的宏定义：`#define __half float16`。
* **两头相撞：** 宏 `#define` 是无视 C++ 语法规则的纯文本替换。在 `xpudnn_cuda_patch.h` 之后的代码中（可能是引入的某个 CUDA 头文件或者补丁文件自身），肯定存在类似 `struct __half { ... };` 这样的声明。预处理器看到 `__half`，立刻将它无脑替换成了 `float16`，于是代码就变成了 `struct float16 { ... };`。这就导致编译器在同一个编译单元里看到了**两次**`struct float16` 的定义，因此直接抛出 `redefinition`（重复定义）的严重错误。

尝试修改1:

将 `#define` 替换为 `typedef`

#define __half        float16

按照这种方式修改会引起Thrust库的新报错，继续注释掉Thrust库中struct __half; 会引起连环爆雷。

修改方案：

彻底删除引发雪崩的别名

```shell
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/xpudnn_cuda_patch.h

 // #define __half        float16
```


(6) 缺少接口，确认一下

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6c6e6c1eb78841dfa2bc997a2c55bf05&docGuid=o8_CyK4_2B8NcQ)
问题根因：

PaddlePaddle 为了兼容不同的环境，使用了一套动态加载机制（DynLoad），在代码里罗列了数百个需要从 `libcublas.so` 和 `libcudnn.so` 中加载的函数名。

但是，你们的 `xtrans` 转换工具目前**并没有实现或者暴露所有的 NVIDIA 接口**（比如旧版的 `_v2` 后缀函数，或者某些不常用的 `Batched` 矩阵求逆函数）。Paddle 找不到这些声明，于是报错。

解决方案：

找到如下未实现接口，直接删除

```shell
cublasSaxpy_v2
cublasDaxpy_v2
cublasCaxpy_v2
cublasZaxpy_v2
cublasSscal_v2
cublasDscal_v2
cublasDcopy_v2
cublasScopy_v2
cublasSmatinvBatched
cublasDmatinvBatched
cublasCmatinvBatched
cublasZmatinvBatched
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=45949b4b6646490d8f7eba308276b1fc&docGuid=o8_CyK4_2B8NcQ)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1e5cf80df893489db9e2090072533006&docGuid=o8_CyK4_2B8NcQ)
```shell
cudnnGetActivationDescriptor
```


（7）paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ddb12bd044b64a00bc417b4bb3ed8e7a&docGuid=o8_CyK4_2B8NcQ)
问题根因：在 C++ 中，为了类型安全，Paddle 原生源码倾向于把句柄声明为严格的结构体指针（例如 `struct cublasContext*`）。

然而，`xtrans` 作为转译层，为了省事和通用，它在底层头文件里把所有的句柄（Stream, Handle）全部简单粗暴地定义成了**万能指针 **`void*`。

当 Paddle 的前置声明遇到 `xtrans` 的底层定义时，C++ 编译器发现同一个名字一个是“结构体指针”，一个是“空指针”，立刻报错冲突。

解决方案：

在异构编译时，让 Paddle 妥协，接受 `void*`。

尝试在xtrans里强转。

```shell
using cudaStream_t = void*;

// 将类似 using cublasHandle_t = struct cublasContext *; 改为：
using cublasHandle_t = void*;

// 同理修改以下几个：
using cusolverDnHandle_t = void*;
using cusparseHandle_t = void*;
// 如果有 cudaStream_t 或者 XPUStream 的 using 声明，也改为 void*
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=83dc43c89c304b0fbb5eaa63ec3dd1c6&docGuid=o8_CyK4_2B8NcQ)


（8）paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=e3523835ed8846f4b97035b0c840b6bf&docGuid=o8_CyK4_2B8NcQ)
当前问题与上述问题类似，但是按照上述问题改为void*后会报出新的错误；

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1eea6df61f3f4ec09166d01ec6e0c435&docGuid=o8_CyK4_2B8NcQ)
问题根因：上次把 `forwards.h` 里的描述符改成 `void*`，目的是绕过 Paddle 的结构体限制。

**但低估了 **`xtrans`** 宏的“霸道”程度！**`xtrans` 在它自己的头文件里，不仅强行用宏重命名了这些描述符（比如把 `cudnnTensorDescriptor_t` 变成了 `xpudnnTensorDescriptor_t`），而且**它自己已经给这些类型做好了完整的 **`typedef`（比如定义为了 `struct xpudnnTensorStruct*`）。

由于 `void*` 和 `struct xpudnnTensorStruct*` 是两种不同的类型，所以编译器立刻报错，不允许我们用 `void*` 覆盖它。

解决方式：

既然 `xtrans` 的头文件已经把这些 cuDNN 类型定义得非常完善，并且通过宏全局注入了，那我们就**不允许 Paddle 在 **`forwards.h`** 里再去插手定义它们，全部删除。**



```shell
vim /home/xuanyuan/Paddle/paddle/phi/backends/gpu/forwards.h
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=33c36563bc66444e8cff62b7df73bdda&docGuid=o8_CyK4_2B8NcQ)


（9）paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=bbb1f9d3e6c249d48f4375a7098a3d97&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* **类型不匹配 (Type Mismatch)：**Paddle 原生代码里的函数返回类型是原生的 `cudaDataType_t` (枚举类型)。而 `xtrans` 把 `CUDA_R_32F` 替换成了 `xpudnn_CUDA_R_32F`，这个值属于另一个完全不同的枚举类型 `xpudnn_cudaDataType_t`。在 C++ 严格的类型检查下，两个不同的枚举类型是**绝对不允许隐式互相转换**的，编译器直接拦截报错。
* **宏定义带分号（非常严重的坏习惯）：**仔细看报错日志里宏定义的内容：`#define CUDA_R_32F xpudnn_CUDA_R_32F;`（最后有一个分号）。宏是纯文本替换。如果 Paddle 代码写了 `func(CUDA_R_32F)`，替换后就会变成 `func(xpudnn_CUDA_R_32F;)`，这会导致语法彻底崩溃。现在的报错还没体现出分号的破坏力，只是因为代码刚好是 `return CUDA_R_32F;`（变成了 `return xpudnn_CUDA_R_32F;;`，两个分号被当成了空语句侥幸逃过一劫）。

修改方式：**去掉分号，并加上强制类型转换**。

```shell
#define CUDA_R_16F ((cudaDataType_t)xpudnn_CUDA_R_16F)
#define CUDA_C_16F ((cudaDataType_t)xpudnn_CUDA_C_16F)
#define CUDA_R_16BF ((cudaDataType_t)xpudnn_CUDA_R_16BF)
#define CUDA_C_16BF ((cudaDataType_t)xpudnn_CUDA_C_16BF)
#define CUDA_R_32F ((cudaDataType_t)xpudnn_CUDA_R_32F)
#define CUDA_C_32F ((cudaDataType_t)xpudnn_CUDA_C_32F)
#define CUDA_R_64F ((cudaDataType_t)xpudnn_CUDA_R_64F)
#define CUDA_C_64F ((cudaDataType_t)xpudnn_CUDA_C_64F)
#define CUDA_R_4I ((cudaDataType_t)xpudnn_CUDA_R_4I)
#define CUDA_C_4I ((cudaDataType_t)xpudnn_CUDA_C_4I)
#define CUDA_R_4U ((cudaDataType_t)xpudnn_CUDA_R_4U)
#define CUDA_C_4U ((cudaDataType_t)xpudnn_CUDA_C_4U)
#define CUDA_R_8I ((cudaDataType_t)xpudnn_CUDA_R_8I)
#define CUDA_C_8I ((cudaDataType_t)xpudnn_CUDA_C_8I)
#define CUDA_R_8U ((cudaDataType_t)xpudnn_CUDA_R_8U)
#define CUDA_C_8U ((cudaDataType_t)xpudnn_CUDA_C_8U)
#define CUDA_R_16I ((cudaDataType_t)xpudnn_CUDA_R_16I)
#define CUDA_C_16I ((cudaDataType_t)xpudnn_CUDA_C_16I)
#define CUDA_R_16U ((cudaDataType_t)xpudnn_CUDA_R_16U)
#define CUDA_C_16U ((cudaDataType_t)xpudnn_CUDA_C_16U)
#define CUDA_R_32I ((cudaDataType_t)xpudnn_CUDA_R_32I)
#define CUDA_C_32I ((cudaDataType_t)xpudnn_CUDA_C_32I)
#define CUDA_R_32U ((cudaDataType_t)xpudnn_CUDA_R_32U)
#define CUDA_C_32U ((cudaDataType_t)xpudnn_CUDA_C_32U)
#define CUDA_R_64I ((cudaDataType_t)xpudnn_CUDA_R_64I)
#define CUDA_C_64I ((cudaDataType_t)xpudnn_CUDA_C_64I)
#define CUDA_R_64U ((cudaDataType_t)xpudnn_CUDA_R_64U)
#define CUDA_C_64U ((cudaDataType_t)xpudnn_CUDA_C_64U)
#define CUDA_R_8F_E4M3 ((cudaDataType_t)xpudnn_CUDA_R_8F_E4M3)
#define CUDA_R_8F_E5M2 ((cudaDataType_t)xpudnn_CUDA_R_8F_E5M2)
```


（10）paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=724a9e58dc5240ca8d1da8e5c8848265&docGuid=o8_CyK4_2B8NcQ)
根因分析：**Paddle 的前置声明机制与 **`xtrans`** 宏替换机制水火不容。**

1. `cudaStream_t`** 冲突（时而宏，时而 typedef）：**在之前的编译中，有的文件先包含了 `xtrans` 的补丁文件，`cudaStream_t` 变成了一个宏。所以我们改成 `void*` 成功骗过了编译器。但在当前的 `device_tracer.cc` 中，它的头文件包含顺序不同，**补丁文件还没被加载**，编译器看到的是 NVIDIA 最原始的 `typedef struct CUstream_st *cudaStream_t;`。此时我们硬塞的 `using cudaStream_t = void*;` 就会和原始定义打架。
2. `cudnnActivationStruct`** 找不到类型：**Paddle 原本为了加快编译速度，在 `forwards.h` 里写了大量的“前置声明”（比如 `struct cudnnActivationStruct;`）而不是直接 `#include <cudnn.h>`。为了解决上一步的死锁，我们把这些前置声明删了。现在 `gpu_decls.h` 在映射类型时，发现既没有 `#include <cudnn.h>`，又没有前置声明，两眼一抹黑，自然报错。

解决方案：**放弃前置声明，直接加载原生头文件**

在处理像 `xtrans`（或 AMD 的 HIP）这种重度依赖宏替换的底层工具链时，**C++ 的前置声明是彻底失效的**。因为你无法前置声明一个“实际上是宏”的类型。

业界解决这个问题的标准做法非常简单粗暴：**在 **`forwards.h`** 中，不再自己声明类型，而是直接把官方头文件 **`#include`** 进来。**

```shell
vim /home/xuanyuan/Paddle/paddle/phi/backends/gpu/forwards.h
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=cb9baa5842254db09efbd21eeacaf8b4&docGuid=o8_CyK4_2B8NcQ)


(11)  paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ad11c4960a094abaaacd8c5fc0692df8&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* **Paddle 的跨平台抽象：** Paddle 为了兼容 CUDA、HIP (AMD) 甚至 XPU，在框架底层把统一的流类型 `phi::gpuStream_t` 定义成了万能指针 `void*`。所以这里的 `raw_stream()` 返回的是一个 `void*`。
* **底层 API 的强类型要求：** 经过我们刚才对 `forwards.h` 的重构，编译器现在非常清楚，NVIDIA（或 xtrans）底层的 `cudaStreamDestroy` 函数需要的是一个严格的 `cudaStream_t`（本质上是 `CUstream_st*` 结构体指针）。
* **C++ 的强类型壁垒：** 在 C 语言中，`void*` 可以隐式转换为任何指针；**但在 C++ 中，**`void*`** 不能隐式转换为具体的结构体指针 **`CUstream_st*`，必须进行强制转换。编译器在此处严格把关，抛出了报错。

解决方法：在调用底层原生 API 的地方，把 Paddle 的 `void*` 显式强转为它需要的具体类型。

```shell
reinterpret_cast<CUstream_st*>(             //在stream外加入强转)
```
vim /home/xuanyuan/Paddle/paddle/phi/core/cuda_stream.h

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=d516885df60c4a8db29bba4d002656ff&docGuid=o8_CyK4_2B8NcQ)
vim /home/xuanyuan/Paddle/paddle/phi/api/profiler/event.h

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=289396e09c7b4876b0450446403615a9&docGuid=o8_CyK4_2B8NcQ)


(12) paddle与xtrans的冲突

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=690d5ecbed4144b5a17222e3f70a417a&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* **上半场（正常）：** 编译器最先读取了 `tensor.h`。此时 `xtrans` 的补丁头文件还没被加载进去，编译器看到的是纯正的 NVIDIA 定义，它记住了：“`Tensor` 类里有一个 `stream()` 方法，它的返回值是 `CUstream_st*`”。
* **中场休息（被投毒）：** 随着 `tensor.cc` 继续往下 include，它间接加载了 `xtrans` 的补丁文件 `xpudnn_cuda_patch.h`。这个补丁文件里有一句极其霸道的宏定义：`#define cudaStream_t XPUStream`（而 `XPUStream` 底层是 `void*`）。
* **下半场（崩溃）：** 当编译器解析到 `tensor.cc` 第 393 行的函数实现 `gpuStream_t Tensor::stream() const { ... }` 时，受宏的影响，这里的返回值被替换成了 `void*`。
* **编译器大怒：** “你在头文件里跟我说返回值是 `CUstream_st*`，怎么到了源文件实现里，返回值变成了 `void*`？这根本不是同一个函数！” 于是直接报错。

解决方案：

这种在编译中途强行改变基本数据类型含义的宏定义，在大型 C++ 工程里是绝对的“毒瘤”。我们必须**拔掉这个宏**，让 `cudaStream_t` 从头到尾保持一致的 `CUstream_st*` 身份。

```shell
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/xpudnn_cuda_patch.h

// #define cudaStream_t XPUStream
```


(13) Clang较为严格

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6175c7c65a0346b7b104a286b243b52f&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* **GCC vs Clang 的分歧：** PaddlePaddle 的原生编译系统默认主要针对 Linux 下的 GCC 编译器，因此在 CMake 里配置了很多 GCC 专属的警告开关（比如 `-Wno-error=literal-suffix`）。
* `xtrans`** 底层其实是 Clang：**从你的编译命令 `/home/xuanyuan/.../bin/clang++ -x cuda` 可以清晰地看到，你们的异构转译编译器 `xtrans` 的底层核心是 **Clang**（LLVM）。
* **Clang 的“洁癖”：**Clang 编译器不认识 `literal-suffix` 这个 GCC 的专属参数。并且，因为编译命令里开启了 `-Werror`（把所有警告视为错误），Clang 发现了一个自己不认识的参数，就直接触发了 `-Wunknown-warning-option`，硬生生把编译给中断了。

解决方案：

解决这个问题的最佳实践，是**给 Clang 穿上一件“防弹衣”**：告诉 Clang，以后遇到不认识的警告参数，直接无视掉就好，不要报错。

```shell
vim cmake/flags.cmake
```


![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0e90401e5b9a4fed87ace4ed1103e7b4&docGuid=o8_CyK4_2B8NcQ)


（14）数据类型相关

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=195dc4bf2dea49b1a1abdd14af69c81e&docGuid=o8_CyK4_2B8NcQ)
根因分析：

在 Paddle 原生的 `bfloat16.h` 源码中，为了适配 NVIDIA，原本写的是 CUDA 的原生类型 `nv_bfloat16`。

但 `xtrans` 工具在转译你的工程时，可能跑了一个脚本，**强行把所有的 **`nv_bfloat16`** 粗暴地替换成了 **`bfloat16`。

这直接引发了“连环车祸”：

1. **Error 1 & 2（构造函数与赋值符冲突）：** 原本代码是 `bfloat16(const nv_bfloat16& val)`，被脚本盲目替换后变成了 `bfloat16(const bfloat16& val)`。这就和 C++ 编译器自动生成的**默认拷贝构造函数**完全一样了！C++ 严禁同时写两个一模一样的拷贝构造函数，所以报错 `constructor cannot be redeclared`。
2. **Error 3 & 4（类型转换失败）：**底层 `xtrans` 提供的数学函数（比如 `__float2bfloat16`）需要的是昆仑芯的真实原生类型 `__xpu_bfloat16`，但脚本把代码里的类型全改成了 Paddle 自己封装的 `bfloat16`，导致两边类型对不上。

解决方案：

在 `paddle/phi/common/bfloat16.h` 这个文件里，把被脚本改坏的这几行代码，手动修正为 XPU 的原生类型 `__xpu_bfloat16`。

确认一下，不用xcuda的bfloat16

```shell
1. 修复隐式转换报错 (大概在 95 行)
bfloat16 tmp = __float2bfloat16(val);
修改为： （因为转换构造函数是 explicit 的，所以要用显式构造或强转）
bfloat16 tmp(__float2bfloat16(val));

2. 修复构造函数重定义 (大概在 104 行)
__attribute__((host)) __attribute__((device)) inline explicit bfloat16(const bfloat16& val) {
修改为： （把参数改回底层的真实硬件类型 __xpu_bfloat16）
__attribute__((host)) __attribute__((device)) inline explicit bfloat16(const __xpu_bfloat16& val) {

3. 修复赋值运算符重定义 (大概在 115 行)
__attribute__((host)) __attribute__((device)) inline bfloat16& operator=(const bfloat16& val) {
修改为：
__attribute__((host)) __attribute__((device)) inline bfloat16& operator=(const __xpu_bfloat16& val) {

4. 修复底层 float 转换报错 (大概在 190 行)
return __bfloat162float(*reinterpret_cast<const bfloat16*>(&x));
修改为：
return __bfloat162float(*reinterpret_cast<const __xpu_bfloat16*>(&x));
```


（15）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=48e26f59ccf34cf3b1805cf56847d045&docGuid=o8_CyK4_2B8NcQ)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=85eedf38030845e4b7807c6015f43ffc&docGuid=o8_CyK4_2B8NcQ)
根因分析：

我们在之前为了解决返回值类型不匹配，在 `xpudnn_cuda_patch.h` 里写了 `#define CUDA_R_16F ((cudaDataType_t)xpudnn_CUDA_R_16F)`。

但是！`xtrans` 的 `library_types.h` 里原本就有原生的枚举定义：

```shell
typedefenumcudaDataType_t {    CUDA_R_16F = 2,     // ...} cudaDataType;
```
预处理器在这里把 `CUDA_R_16F` 强行替换成了 `((cudaDataType_t)...)`，导致整个 Enum 的语法直接崩溃（变成了 `((cudaDataType_t)...) = 2`）。这使得整个 `cudaDataType` 都未能成功声明，所以后续所有用到它的地方全都报 `has not been declared`。

**结论：既然 **`library_types.h`** 已经提供了完美的原生枚举，我们根本不需要写这些宏，直接删掉最安全！**

解决方案：

```shell
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/cudnn_api/xpudnn_cuda_patch.h

// 把这些统统删掉，把控制权交还给 library_types.h 里的原生枚举
// #define CUDA_R_16F ((cudaDataType_t)xpudnn_CUDA_R_16F)
// #define CUDA_C_16F ((cudaDataType_t)xpudnn_CUDA_C_16F)
// ... 中间所有的类型宏 ...
// #define CUDA_R_8F_E5M2 ((cudaDataType_t)xpudnn_CUDA_R_8F_E5M2)
```


（16）xtrans问题

问题根因：

`libraryPropertyType_t`** 被定义了两次（xtrans 的锅）：**

看报错日志，`library_types.h` 和 `xpudnn_ops_infer.h` 这两个 `xtrans` 提供的头文件里，居然定义了两个一模一样的 `enum libraryPropertyType_t`。由于 C++ 不允许在同一个文件里重复定义相同的枚举，直接导致了 `multiple definition` 错误。

解决方案：

```shell
// line108
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/xpudnn/xpudnn_ops_infer.h

// typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType;
```


（17） 缺少接口，确认一下

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=63069f055a074f27ba29430cd4ebf3a8&docGuid=o8_CyK4_2B8NcQ)
根因分析：PaddlePaddle 的 `cufft.cc` 试图从底层的 FFT 库中动态加载一系列名字里带 `Xt` 的高级扩展函数（例如 `cufftXtSetGPUs`, `cufftXtMalloc` 等）。

但是，你们的 `xtrans` 异构转换工具在其底层的 FFT 库实现中，**并没有提供这些高级的 **`Xt`** 扩展接口**。当 Paddle 找不到这些函数的声明时，自然就抛出了 `has not been declared` 错误。

解决方案：

```shell
// 在 CUFFT_FFT_ROUTINE_EACH 宏定义内部，把以下带有 Xt 的函数全部删掉：

// 删除这行：__macro(cufftXtSetGPUs);                \
// 删除这行：__macro(cufftXtMalloc);                 \
// 删除这行：__macro(cufftXtMemcpy);                 \
// 删除这行：__macro(cufftXtFree);                   \
// 删除这行：__macro(cufftXtSetWorkArea);            \
// 删除这行：__macro(cufftXtExecDescriptorC2C);      \
// 删除这行：__macro(cufftXtExecDescriptorR2C);      \
// 删除这行：__macro(cufftXtExecDescriptorC2R);      \
// 删除这行：__macro(cufftXtExecDescriptorZ2Z);      \
// 删除这行：__macro(cufftXtExecDescriptorD2Z);      \
// 删除这行：__macro(cufftXtExecDescriptorZ2D);      \
// 删除这行：__macro(cufftXtQueryPlan);              \
// 删除这行：__macro(cufftXtGetSizeMany);            \
// 删除这行：__macro(cufftXtExecDescriptor);         \
// 删除这行：__macro(cufftXtSetWorkAreaPolicy)
```


（18）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=83f48aeddbe44ba19fdfc317fc278858&docGuid=o8_CyK4_2B8NcQ)
问题根因：

* **cuBLAS v1 与 v2 的历史遗留：**在原生的 NVIDIA CUDA 环境中，`cublas_v2.h` 头文件里存在大量的宏映射，比如 `#define cublasCreate cublasCreate_v2`。这是为了向下兼容。PaddlePaddle 在编写代码时，调用的是 `phi::dynload::cublasCreate`，它原本指望靠 NVIDIA 的宏把它自动替换成 `phi::dynload::cublasCreate_v2`（Paddle 的动态加载库里实际注册的只有 `_v2` 版本的函数）。
* `xtrans`** 工具缺失兼容宏：**你们的 `xtrans` 转换工具提供的 `cublas_v2.h` 比较“实在”，它没有包含这些向下兼容的宏替换。因此，当编译器去 `phi::dynload` 命名空间里找 `cublasCreate` 时，发现根本没有这个函数（只有 `cublasCreate_v2`），于是直接报错。
* **连环副作用：** 日志里出现的 `__CUDA_STATUS_TYPE__ was not declared` 只是一个幽灵报错。因为找不到函数，宏 `PADDLE_RETRY_CUDA_SUCCESS` 无法推导出函数返回类型，导致模板崩溃。一旦函数名写对了，这个错误会自动消失。

解决方案：

直接去 C++ 源码里，把函数名显式改成 `_v2` 版本。

```shell
1. 修改 cublasCreate (大约在 174 行)
PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasCreate(blas_handle));
修改为：
PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasCreate_v2(blas_handle));

2. 修改 cublasSetStream (大约在 176 行)
phi::dynload::cublasSetStream(*blas_handle, stream));
修改为：
phi::dynload::cublasSetStream_v2(*blas_handle, stream));

3. 修改 cublasDestroy (大约在 188 行)
phi::dynload::cublasDestroy(handle);
修改为：
phi::dynload::cublasDestroy_v2(handle);
```


(19) 此处为一个文件爆出的四个问题，因为都在一个文件修改就写在一起

修改方式（太长，折叠了）

```shell
--- a/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
+++ b/paddle/phi/kernels/funcs/blas/blas_impl.cu.h
@@ -40,27 +40,27 @@ template <>
 struct CUBlas<float> {
   template <typename... ARGS>
   static void GEMM(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm_v2(args...));
   }

   template <typename... ARGS>
   static void AXPY(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSaxpy(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasSaxpy(args...));
   }

   template <typename... ARGS>
   static void SCAL(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSscal(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasSscal(args...));
   }

   template <typename... ARGS>
   static void VCOPY(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasScopy(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasScopy(args...));
   }

   template <typename... ARGS>
   static void GEMV(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemv(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemv_v2(args...));
   }

   template <typename... ARGS>
@@ -185,7 +185,7 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void TRSM(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsm(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsm_v2(args...));
   }

   template <typename... ARGS>
@@ -200,7 +200,8 @@ struct CUBlas<float> {

   template <typename... ARGS>
   static void MATINV_BATCH(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSmatinvBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("SmatinvBatched is not supported by xtrans."));
+         //PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSmatinvBatched(args...));
   }

   template <typename... ARGS>
@@ -223,27 +224,27 @@ template <>
 struct CUBlas<double> {
   template <typename... ARGS>
   static void GEMM(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm_v2(args...));
   }

   template <typename... ARGS>
   static void AXPY(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDaxpy(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasDaxpy(args...));
   }

   template <typename... ARGS>
   static void SCAL(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDscal(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasDscal(args...));
   }

   template <typename... ARGS>
   static void VCOPY(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDcopy(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(cublasDcopy(args...));
   }

   template <typename... ARGS>
   static void GEMV(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemv(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemv_v2(args...));
   }

   template <typename... ARGS>
@@ -281,7 +282,7 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void TRSM(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsm(args...));
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsm_v2(args...));
   }

   template <typename... ARGS>
@@ -296,7 +297,8 @@ struct CUBlas<double> {

   template <typename... ARGS>
   static void MATINV_BATCH(ARGS... args) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDmatinvBatched(args...));
+    PADDLE_THROW(phi::errors::Unimplemented("DmatinvBatched is not supported by xtrans."));
+         // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDmatinvBatched(args...));
   }

   template <typename... ARGS>
@@ -340,13 +342,13 @@ struct CUBlas<phi::float16> {
                                   m,
                                   n,
                                   k,
-                                  reinterpret_cast<const __half *>(alpha),
-                                  reinterpret_cast<const __half *>(A),
+                                  (const cublasHalf*)alpha,
+                                  (const cublasHalf*)A,
                                   lda,
-                                  reinterpret_cast<const __half *>(B),
+                                  (const cublasHalf*)B,
                                   ldb,
-                                  reinterpret_cast<const __half *>(beta),
-                                  reinterpret_cast<__half *>(C),
+                                  (const cublasHalf*)beta,
+                                  (cublasHalf*)C,
                                   ldc));
   }

@@ -439,15 +441,15 @@ struct CUBlas<phi::float16> {
         m,
         n,
         k,
-        reinterpret_cast<const __half *>(alpha),
-        reinterpret_cast<const __half *>(A),
+        (const cublasHalf*)alpha,
+        (const cublasHalf*)A,
         lda,
         strideA,
-        reinterpret_cast<const __half *>(B),
+        (const cublasHalf*)B,
         ldb,
         strideB,
-        reinterpret_cast<const __half *>(beta),
-        reinterpret_cast<__half *>(C),
+        (const cublasHalf*)beta,
+        (cublasHalf*)C,
         ldc,
         strideC,
         batchCount));
@@ -606,7 +608,7 @@ struct CUBlas<phi::complex64> {
                    const phi::complex64 *beta,
                    phi::complex64 *C,
                    int ldc) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemv(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemv_v2(
         handle,
         transa,
         m,
@@ -628,14 +630,15 @@ struct CUBlas<phi::complex64> {
                    const int incX,
                    phi::complex64 *Y,
                    const int incY) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCaxpy(
-        handle,
-        n,
-        reinterpret_cast<const cuFloatComplex *>(alpha),
-        reinterpret_cast<const cuFloatComplex *>(X),
-        incX,
-        reinterpret_cast<cuFloatComplex *>(Y),
-        incY));
+    PADDLE_THROW(phi::errors::Unimplemented("Complex AXPY is not supported by xtrans yet."));
+      //PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCaxpy(
+    //    handle,
+    //    n,
+    //    reinterpret_cast<const cuFloatComplex *>(alpha),
+    //    reinterpret_cast<const cuFloatComplex *>(X),
+    //    incX,
+    //    reinterpret_cast<cuFloatComplex *>(Y),
+    //    incY));
   }

   static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
@@ -696,7 +699,7 @@ struct CUBlas<phi::complex64> {
                    const phi::complex64 *beta,
                    phi::complex64 *C,
                    int ldc) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm_v2(
         handle,
         transa,
         transb,
@@ -725,7 +728,7 @@ struct CUBlas<phi::complex64> {
                    int lda,
                    phi::complex64 *B,
                    int ldb) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsm(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsm_v2(
         handle,
         side,
         uplo,
@@ -875,10 +878,10 @@ struct CUBlas<phi::complex64> {
         diag,
         m,
         n,
-        reinterpret_cast<const cuFloatComplex *>(alpha),
-        reinterpret_cast<const cuFloatComplex **>(A),
+        (const cublasComplex *)alpha,
+        (cublasComplex* const*)A,
         lda,
-        reinterpret_cast<cuFloatComplex **>(B),
+        (cublasComplex **)B,
         ldb,
         batch_size));
   }
@@ -912,10 +915,10 @@ struct CUBlas<phi::complex64> {
     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetriBatched(
         handle,
         n,
-        reinterpret_cast<const cuFloatComplex **>(A),
+        (cublasComplex* const*)A,
         lda,
-        ipiv,
-        reinterpret_cast<cuFloatComplex **>(Ainv),
+        const_cast<int*>(ipiv),
+        (cublasComplex **)Ainv,
         ldc,
         info,
         batch_size));
@@ -929,15 +932,16 @@ struct CUBlas<phi::complex64> {
                            int lda_inv,
                            int *info,
                            int batch_size) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCmatinvBatched(
-        handle,
-        n,
-        reinterpret_cast<const cuFloatComplex **>(A),
-        lda,
-        reinterpret_cast<cuFloatComplex **>(Ainv),
-        lda_inv,
-        info,
-        batch_size));
+    PADDLE_THROW(phi::errors::Unimplemented("Complex MATINV_BATCH is not supported by xtrans yet."));
+      //PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCmatinvBatched(
+    //    handle,
+    //    n,
+    //    reinterpret_cast<const cuFloatComplex **>(A),
+    //    lda,
+    //    reinterpret_cast<cuFloatComplex **>(Ainv),
+    //    lda_inv,
+    //    info,
+    //    batch_size));
   }

   static void DOT(cublasHandle_t handle,
@@ -972,7 +976,7 @@ struct CUBlas<phi::complex128> {
                    const phi::complex128 *beta,
                    phi::complex128 *C,
                    int ldc) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemv(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemv_v2(
         handle,
         transa,
         m,
@@ -994,14 +998,15 @@ struct CUBlas<phi::complex128> {
                    const int incX,
                    phi::complex128 *Y,
                    const int incY) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZaxpy(
-        handle,
-        n,
-        reinterpret_cast<const cuDoubleComplex *>(alpha),
-        reinterpret_cast<const cuDoubleComplex *>(X),
-        incX,
-        reinterpret_cast<cuDoubleComplex *>(Y),
-        incY));
+    PADDLE_THROW(phi::errors::Unimplemented("Complex AXPY is not supported by xtrans yet."));
+      //PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZaxpy(
+    //    handle,
+    //    n,
+    //    reinterpret_cast<const cuDoubleComplex *>(alpha),
+    //    reinterpret_cast<const cuDoubleComplex *>(X),
+    //    incX,
+    //    reinterpret_cast<cuDoubleComplex *>(Y),
+    //    incY));
   }

   static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
@@ -1062,7 +1067,7 @@ struct CUBlas<phi::complex128> {
                    const phi::complex128 *beta,
                    phi::complex128 *C,
                    int ldc) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm_v2(
         handle,
         transa,
         transb,
@@ -1091,7 +1096,7 @@ struct CUBlas<phi::complex128> {
                    int lda,
                    phi::complex128 *B,
                    int ldb) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsm(
+    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsm_v2(
         handle,
         side,
         uplo,
@@ -1127,10 +1132,10 @@ struct CUBlas<phi::complex128> {
         diag,
         m,
         n,
-        reinterpret_cast<const cuDoubleComplex *>(alpha),
-        reinterpret_cast<const cuDoubleComplex **>(A),
+        (const cublasDoubleComplex *)alpha,
+        (cublasDoubleComplex* const*)A,
         lda,
-        reinterpret_cast<cuDoubleComplex **>(B),
+        (cublasDoubleComplex **)B,
         ldb,
         batch_size));
   }
@@ -1278,10 +1283,10 @@ struct CUBlas<phi::complex128> {
     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgetriBatched(
         handle,
         n,
-        reinterpret_cast<const cuDoubleComplex **>(A),
+        (cublasDoubleComplex* const*)A,
         lda,
-        ipiv,
-        reinterpret_cast<cuDoubleComplex **>(Ainv),
+        const_cast<int*>(ipiv),
+        (cublasDoubleComplex **)Ainv,
         ldc,
         info,
         batch_size));
@@ -1295,15 +1300,16 @@ struct CUBlas<phi::complex128> {
                            int lda_inv,
                            int *info,
                            int batch_size) {
-    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZmatinvBatched(
-        handle,
-        n,
-        reinterpret_cast<const cuDoubleComplex **>(A),
-        lda,
-        reinterpret_cast<cuDoubleComplex **>(Ainv),
-        lda_inv,
-        info,
-        batch_size));
+      PADDLE_THROW(phi::errors::Unimplemented("Complex MATINV_BATCH is not supported by xtrans yet."));
+      //PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZmatinvBatched(
+    //    handle,
+    //    n,
+    //    reinterpret_cast<const cuDoubleComplex **>(A),
+    //    lda,
+    //    reinterpret_cast<cuDoubleComplex **>(Ainv),
+    //    lda_inv,
+    //    info,
+    //    batch_size));
   }

   static void DOT(cublasHandle_t handle,
@@ -2881,7 +2887,7 @@ inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                                    static_cast<int>(ldc),
                                                    strideC,
                                                    static_cast<int>(batchCount),
-                                                   CUBLAS_COMPUTE_32F,
+                                                   (cudaDataType_t)CUBLAS_COMPUTE_32F,
                                                    algo));
     });
   }
@@ -2986,7 +2992,7 @@ inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                                    static_cast<int>(ldc),
                                                    strideC,
                                                    static_cast<int>(batchCount),
-                                                   CUBLAS_COMPUTE_32F,
+                                                  (cudaDataType_t)CUBLAS_COMPUTE_32F,
                                                    algo));
     });
   }
```


（20）

发现了一些由于定制化编译工具链（Clang + `-Werror` ）下导致检查极其严厉的问题。

解决方案：关闭`-Werror`

`cmake/flags.cmake`

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=fb8d34eab99c469f9ca646a68ef9e330&docGuid=o8_CyK4_2B8NcQ)
当前没效果，怀疑是缓存机制导致，后续删除后重编尝试。



（21）理论上20搞定应该没这个问题

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=796c29c385a647da801ace0c01e863ef&docGuid=o8_CyK4_2B8NcQ)
* **问题原因：** 编译器在 Eigen 的第三方源码中发现了一个不规范的写法——在 `for` 循环里，把一个有符号的 `int` 变量和无符号的 `size_t` 变量放在一起比较了（比如 `p < numPlanes`）。
* **为什么会中断：** 这通常只是一个 `Warning`（警告），绝大多数环境下编译器会直接忽略并继续编译。但是，你的编译环境开启了 `-Werror` 编译选项，**这个选项会把所有 Warning 强制视为 Error 并中断编译**。因为 Eigen 是第三方库，它的源码本来就带有一点历史遗留的不规范写法，这就导致了编译“卡壳”。

修改方式：

把用于循环计数的 `int` 改成 `size_t`（因为等号右边的变量都是 `size_t`）

```shell
vim /home/xuanyuan/Paddle/third_party/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=883d718220fa43658bb2c0540086f113&docGuid=o8_CyK4_2B8NcQ)


（22）理论上20搞定应该没这个问题

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0e27062641014e2f8df5d8492fcf6f83&docGuid=o8_CyK4_2B8NcQ)
问题根因：

`/home/xuanyuan/Paddle/paddle/phi/backends/gpu/cuda/cuda_device_function.h` 中，Paddle 试图对 `bfloat16`（16 位浮点数）执行 `__shfl_down` 和 `__shfl_xor` 线程束洗牌操作。但是在你的定制化环境（`xtdk_warp_functions.h`）里，底层并没有提供专为 `__nv_bfloat16` 设计的重载函数，导致编译器在面对 `int`、`float`、`double` 等一堆备选函数时“患了选择困难症”，报出 `ambiguous`（歧义）。

修改方式：

在进行线程束操作前，**先把 **`bfloat16`** 转换为 **`float`，利用 `float` 版本的洗牌函数传递数据，传完后再通过外层的 `phi::dtype::bfloat16(...)` 强转回 `bfloat16`。

```shell
第一处（约第 68 行）：
// 原始代码：
return phi::dtype::bfloat16(__shfl_down(val.to_nv_bfloat16(), delta, width));

// 修改为：
return phi::dtype::bfloat16(__shfl_down(static_cast<float>(val), delta, width));

第二处（约第 110 行）：
// 原始代码：
return phi::dtype::bfloat16(__shfl_xor(val.to_nv_bfloat16(), lane_mask, width));

// 修改为：
return phi::dtype::bfloat16(__shfl_xor(static_cast<float>(val), lane_mask, width));
```
此处跟源码不同，需要确认。



（23）clang编译器严格导致

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2800ccf138e9480fbfc6e595b3304df4&docGuid=o8_CyK4_2B8NcQ)
问题根因：

编译参数开启了 `-Werror`（视警告为错误）和 `-Winconsistent-missing-override`。Clang 编译器极度严格，它发现子类重写了父类的虚函数 `cudnn_handle()`，但却没有在末尾写上 `override` 关键字，因此强行中断了编译。

修改方式：补上丢失的 override 关键字

```shell
// 原始代码：
dnnHandle_t cudnn_handle() const;

// 修改为（在行尾加上 override）：
dnnHandle_t cudnn_handle() const override;
```


（24）clang编译器严格导致

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=023e4e4d9b054e95a9650601956fd5c2&docGuid=o8_CyK4_2B8NcQ)
问题根因：

`cub` (CUDA Unbound) 是 NVIDIA 提供的一个用于 CUDA 编程的高性能底层原语库。你的编译器在处理 `bert_encoder_functor.cu` 这个文件时，遇到了 `cub::BlockReduce` 和 `cub::Sum`，但是由于代码文件顶部忘记了 `#include` 相关的 CUB 头文件，导致编译器根本不认识这两个词。

在老版本的 `nvcc` 编译器中，某些上层头文件可能会隐式（悄悄地）帮你引入 CUB 库，但你现在使用的是极其严格的 Clang 19，它要求必须显式引入。

解决方式：

```shell
vim /home/xuanyuan/Paddle/paddle/phi/kernels/funcs/math/bert_encoder_functor.cu

vim /home/xuanyuan/Paddle/paddle/phi/kernels/funcs/emb_eltwise_layer_norm_functor.cu

//前方加入引入
#include <cub/cub.cuh>
```


(25) clang编译器严格导致

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=405756c907ca4939981255fa74155514&docGuid=o8_CyK4_2B8NcQ)
问题根因：

**头文件缺失**，导致编译器不认识 `memory_utils`。

在 Paddle 的架构中，`memory_utils::Copy` 这个函数通常定义在与内存处理相关的头文件中。在编译过程中，如果没有显式 `#include` 它，严格的 Clang 编译器就会抛出 `undeclared identifier`（未声明的标识符）错误。

解决方式：

```shell
vim /home/xuanyuan/Paddle/paddle/phi/kernels/funcs/fake_quantize_functor.cu

#include "paddle/phi/common/memory_utils.h"
```


(26) 需修改xtrans的json namespace

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=db27cc65c93745cd8244f16e51dd4a92&docGuid=o8_CyK4_2B8NcQ)
问题根因：

报错信息 `error: unexpected namespace name 'json': expected expression` 表明编译器在处理 `cudnn-frontend`（Paddle 的第三方库）时，把本该作为**类型名/别名**的 `json` 误认为了一个**命名空间**。

**命名空间冲突**：在 `xtrans` 的头文件中，定义了一个名为 `json` 的命名空间（或者通过 `using namespace` 引入了它），这与 `cudnn-frontend` 内部使用的 `using json = nlohmann::json;` 产生了冲突。



解决方式：优先修改xtrans，将里面的 `namespace json` 全局替换为 `namespace xpu_json`

```shell
# 将 namespace json 改为 namespace xpu_json
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/include/xpu/refactor/impl_public/json.h
line 22

# 调用时也需要做相应修改
vim /home/xuanyuan/xtrans_cuda_11.7_ubuntu2004_x86_64_mars/targets/x86_64-linux/include/xpu/refactor/context/xpu_act_type.h
line 98 100 102
```


同时由于`json` 这个简写本身并没有在每个文件中都包含 `using json = nlohmann::json;`，它依赖于某个顶层文件定义的别名。



```shell
vim paddle/phi/backends/dynload/cudnn_frontend.h
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2cbc1f7e1d54490fbfdc4d0ea9e31809&docGuid=o8_CyK4_2B8NcQ)


(27)

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=bace7e42c5b64d38bae367239fd4b237&docGuid=o8_CyK4_2B8NcQ)
问题根因：

你的类 CUDA 编译器（`xcn`）在处理自定义浮点类型（`bf16`, `fp8`）时，定义了非常多的全局 `operator==` 模板。而 `nlohmann::json` 为了方便用户，定义了类似 `operator T()` 的隐式转换。

当编译器看到 `json1 == json2` 时，它面临两个诱惑：

1. 使用 JSON 库自带的 `==`。
2. 将 `json1` 和 `json2` 都隐式转换成 `float8_e4m3`（或 `bf16` 等），然后调用你硬件 SDK 里的 `==`。

解决方式：

（1）尝试了不开启隐式转换，会导致后方其他报错。

CMakeLists.txt中加入 add_definitions(-DJSON_USE_IMPLICIT_CONVERSIONS=0)

（2）尝试在函数中重写宏，不奏效，最终在下方文件的宏中更改 == 为 sed::equal_to

```shell
vim /home/xuanyuan/Paddle/third_party/nlohmann_json/include/nlohmann/detail/macro_scope.hpp
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8e2e6ab7bea54102acd48680f4f4c8db&docGuid=o8_CyK4_2B8NcQ)


（28）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2608c83934334e9fb72017ffe4ab87d6&docGuid=o8_CyK4_2B8NcQ)
问题17在动态加载中删除多了，先把这两个恢复，引入新问题再解决。



（29）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=67bf14fd83a3484198edb2948d502f7b&docGuid=o8_CyK4_2B8NcQ)
问题根因：

在 C++ 中，`float**` 可以隐式转换为 `const float* const*`（增加一级和二级常性），但**不能**从 `const float**` 转换为 `float* const*`。因为你当前的输入参数 `A` 已经是 `const complex64**` 了，编译器认为你试图通过这个 API 去修改一个被声明为 `const` 的数据块，或者违反了指针层级的常性安全规则。

此外，由于 `substitution failure`，导致 `PADDLE_ENFORCE_GPU_SUCCESS` 宏内部无法推导出 `__CUDA_STATUS_TYPE__`，从而引发了连带的语法错误。

你的 SDK 头文件使用了更严格的 `* const*`（二级指针常量）定义，而 Paddle 源码默认使用的是标准的二级指针。

解决方案：

将参数强制转换为 SDK 严格要求的类型。

```shell
  860   static void TRSM_BATCH(cublasHandle_t handle,
   861                          cublasSideMode_t side,
   862                          cublasFillMode_t uplo,
   863                          cublasOperation_t transa,
   864                          cublasDiagType_t diag,
   865                          int m,
   866                          int n,
   867                          const phi::complex64 *alpha,
   868                          const phi::complex64 **A,
   869                          int lda,
   870                          phi::complex64 **B,
   871                          int ldb,
   872                          int batch_size) {
   873     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsmBatched(
   874         handle,
   875         side,
   876         uplo,
   877         transa,
   878         diag,
   879         m,
   880         n,
!  881         reinterpret_cast<const cuFloatComplex *>(alpha),
!  882         reinterpret_cast<cuFloatComplex* const*>(const_cast<complex64**>(A)),
!  883         lda,
!  884         reinterpret_cast<cuFloatComplex **>(B),
!  885         ldb,
   886         batch_size));
   887   }



   906   static void GETRI_BATCH(cublasHandle_t handle,
   907                           int n,
   908                           const phi::complex64 **A,
   909                           int lda,
   910                           const int *ipiv,
   911                           phi::complex64 **Ainv,
   912                           int ldc,
   913                           int *info,
   914                           int batch_size) {
   915     PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetriBatched(
   916         handle,
!  917         n,
!  918         reinterpret_cast<cuFloatComplex* const*>(const_cast<complex64**>(A)),
!  919         lda,
!  920         const_cast<int*>(ipiv),
!  921         reinterpret_cast<cuFloatComplex* const*>(Ainv),
!  922         ldc,
   923         info,
   924         batch_size));
   925   }
```




（30）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3a38fec777014378b9280cb9b66e807a&docGuid=o8_CyK4_2B8NcQ)
根因分析：

观察报错中的 `instantiation`：

* **A参数**：传入的是 `const float **`，SDK 要求的是 `const float* const*`。
* **C参数**（也就是 `a_inv`）：传入的是 `float **`，SDK 要求的是 `float* const*`。

解决方案：

需要把下方两个函数展开，强转参数；暂时先用PADDLE_THROW跳过。

```shell
vim paddle/phi/kernels/funcs/blas/blas_impl.cu.h
```
198

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f084ae1669c0460396babbc0880119af&docGuid=o8_CyK4_2B8NcQ)
296

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=79d17c7b9a4847a38998e1d8838a45a0&docGuid=o8_CyK4_2B8NcQ)
215

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=7fc6509d4ad840a1bcbacd2860d65f64&docGuid=o8_CyK4_2B8NcQ)
313

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b90606a157df43e98c24b08bffd1a8b1&docGuid=o8_CyK4_2B8NcQ)
2567

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8a04b528463c449892edf87de536865d&docGuid=o8_CyK4_2B8NcQ)




(31)

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=dc53d5db683349a684fee1ed53f3c8f1&docGuid=o8_CyK4_2B8NcQ)
根因分析：

matrix_reduce.cu会include该文件：#include "paddle/phi/kernels/funcs/reduce_function.h"

该文件中的编译选项是根据NVCC判断的，当前关闭NVCC，所以其中的命名空间和算子名称等都找不到。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=edf00a5eae79479f8e61e18934cc713a&docGuid=o8_CyK4_2B8NcQ)
把这块nvcc的编译换成gpu后，会报出特别多的错误，如果不用通讯，是不是不需要用到reduce，暂时先把该算子的编译去掉。

当前直接给相关算子改名为.bak。

```shell
paddle/phi/kernels/funcs/matrix_reduce.cu
paddle/phi/kernels/funcs/matrix_reduce.cc

paddle/phi/kernels/funcs/pooling.cu
paddle/phi/kernels/funcs/pooling.cc
```


(32)

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=988f3a8454084e61911be6e61bcc94a7&docGuid=o8_CyK4_2B8NcQ)
两个问题：

1是cub库中缺少DeviceScan,这个可以在cub中通过把nvcc宏转换为paddlewithgpu解决(vim paddle/phi/kernels/funcs/cub.h)

2是之前对matrix_reduce的注释影响到该算子。

```shell
paddle/phi/kernels/funcs/repeat_tensor2index_tensor.cu
paddle/phi/kernels/funcs/repeat_tensor2index_tensor.cc
```


（33）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=755aa070a9d74dc19eace7bb5a78f540&docGuid=o8_CyK4_2B8NcQ)
两个问题：

1.paddle的bfloat16没办法直接转换为xcn的bfloat16

2.`sync` 函数在定义时可能只写了 `float*` 参数，但在某些模板实例化路径下，它收到的参数类型推导失败。

```shell
paddle/phi/kernels/funcs/weight_only_gemv.cu
```


(34)

跳过fusion算子，跟bfloat16在xcuda和paddle上定义不同相关，改了很多次都有问题，先跳过。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=801d7cca55a1442c8ebd0f737b028155&docGuid=o8_CyK4_2B8NcQ)


(35)

发现有许多NVCC宏内的东西是缺少的，打开NVCC宏（即使会带来很多新问题）。

不清楚xcuda是怎么处理的nvcc，尝试做两个替换把NVCC换成CUDACC

defined(__NVCC__) -> defined(__NVCC__) || defined(__CUDACC__)

#ifdef __NVCC__-> #ifdef __CUDACC__ //1



（36）

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=4b9e60f70fb84f77afcb25bc7f0d8e00&docGuid=o8_CyK4_2B8NcQ)
在fake_quantize_functor.cc编译时遇到茫茫多的CUDA相关代码不支持的问题，经分析cc代码应该是host侧，可是此处最终调用到了device侧，暂时跳过该算子。



(37)

数据类型问题，xcuda的float16与bfloat16是通过cudnn引入进去的，cudnn即使是在设置了with_cudnn_frontend也会引入。



（38）

```shell
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.cu:182:17: error: non-constant-expression cannot be narrowed from type 'int' to 'unsigned int' in initializer list [-Wc++11-narrowing]
  182 |   uint3 grid = {(M + blockM - 1) / blockM, (N + blockN - 1) / blockN, 1};
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.cu:182:17: note: insert an explicit cast to silence this issue
  182 |   uint3 grid = {(M + blockM - 1) / blockM, (N + blockN - 1) / blockN, 1};
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~
      |                 static_cast<unsigned int>( )
```
问题根因：`uint3` 是 CUDA 内置结构体，其字段类型为 `unsigned int`。`M` 和 `N` 是 `int`，整个表达式 `(M + blockM - 1) / blockM` 的结果也是 `int`。C++11 规定聚合初始化列表中不允许窄化转换（`int` → `unsigned int`），因此编译器报错。

解决方式：

```shell
  uint3 grid = {static_cast<unsigned int>((M + blockM - 1) / blockM),
                static_cast<unsigned int>((N + blockN - 1) / blockN),
                1};
```


（39）

```shell
/home/xuanyuan/Paddle/paddle/phi/kernels/funcs/weight_only_gemv.cu:1024:3: error: reference to overloaded function could not be resolved; did you mean to call it?
 1024 |   Details::Layout::sync<Num, WarpSize>(reses, sm);
      |   ^~~~~~~~~~~~~~~~~~~~~
/home/xuanyuan/Paddle/paddle/phi/kernels/funcs/weight_only_gemv.cu:1072:7: note: in instantiation of function template specialization 'phi::(anonymous namespace)::weight_only_batched_gemv_multi_warp<__half, phi::(anonymous namespace)::WeightOnlyQuantType::Int4b, phi::(anonymous namespace)::WeightOnlyPerChannel, true, false, true, 1, 1, 192>' requested here
 1072 |       weight_only_batched_gemv_multi_warp<T,
      |       ^
/home/xuanyuan/Paddle/paddle/phi/kernels/funcs/weight_only_gemv.cu:1151:9: note: in instantiation of function template specialization 'phi::(anonymous namespace)::select_activation_and_bias<__half, phi::(anonymous namespace)::WeightOnlyQuantType::Int4b, phi::(anonymous namespace)::WeightOnlyPerChannel, 1, 1, 192>' requested here
 1151 |         select_activation_and_bias<T,
      |         ^
/home/xuanyuan/Paddle/paddle/phi/kernels/funcs/weight_only_gemv.cu:1274:5: note: in instantiation of function template specialization 'phi::(anonymous namespace)::weight_only_batched_gemv_launcher<__half, phi::(anonymous namespace)::WeightOnlyPerChannel>' requested here
 1274 |     weight_only_batched_gemv_launcher<DataType, WeightOnlyPerChannel>(
      |     ^
/home/xuanyuan/Paddle/paddle/phi/kernels/funcs/weight_only_gemv.cu:415:81: note: possible target for call
  415 |   __attribute__((device)) __inline__ __attribute__((always_inline)) static void sync(float* res,
      |                                                                                 ^
```
`etails` 是依赖模板参数 `QType` 的类型（`WeightOnlyKernelDetails<QType>`），因此 `Details::Layout` 也是依赖名称（dependent name）。在依赖名称的作用域下调用其模板成员函数时，C++ 标准要求必须使用 `template` 关键字消歧，否则编译器会把 `<` 解析为小于号，导致无法正确识别这是一个模板函数调用。

问题解决：

```shell
 // Details::Layout::sync<Num, WarpSize>(reses, sm);
 Details::Layout::template sync<Num, WarpSize>(reses, sm);
```


（40）

```shell
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/strings/gpu/strings_copy_kernel.cu:22:
In file included from /home/xuanyuan/Paddle/paddle/phi/common/pstring.h:27:
/home/xuanyuan/Paddle/paddle/phi/common/cpstring_impl.h:187:3: error: reference to __host__ function 'free' in __host__ __device__ function
  187 |   free(ptr);
      |   ^
/home/xuanyuan/Paddle/paddle/phi/common/cpstring_impl.h:270:5: note: called by 'PD_PString_Dealloc'
  270 |     PD_Free(str->u.large.ptr, str->u.large.cap + 1);
      |     ^
/home/xuanyuan/Paddle/paddle/phi/common/cpstring_impl.h:530:3: note: called by 'PD_PString_Move'
  530 |   PD_PString_Dealloc(dst);
      |   ^
/home/xuanyuan/Paddle/paddle/phi/common/pstring.h:226:3: note: called by 'operator='
  226 |   PD_PString_Move(&pstr_, &str.pstr_);
      |   ^
/home/xuanyuan/Paddle/paddle/phi/kernels/strings/gpu/copy_utils.h:99:16: note: called by 'DeserializeCUDAKernel'
   99 |     dst_str[i] = phi::dtype::pstring(strings_data + strings_offset[i], len);
      |                ^
/usr/include/stdlib.h:565:13: note: 'free' declared here
  565 | extern void free (void *__ptr) throw ();
      |             ^
```
device调用了host 暂时跳过。

```shell
Paddle/paddle/phi/kernels/strings/gpu/strings_copy_kernel.cu
Paddle/paddle/phi/kernels/strings/gpu/strings_lower_upper_kernel.cu

```


（41）

```shell
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen/generic_mixed_gemm_kernelLauncher_bf16_sm80_stages2_bias.cu:4:
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h:43:
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/fpA_intB_gemm_split_k.h:538:12: error: missing 'typename' prior to dependent type name 'Mma::IteratorA'
  538 |     return Mma::IteratorA(params.params_A,
      |            ^~~~~~~~~~~~~~
```
问题根因：`Mma` 是类模板参数，`Mma::IteratorA` 是**依赖类型名（dependent type name）**。在 C++ 模板中，编译器无法在解析模板定义时确定 `Mma::IteratorA` 是一个类型还是一个静态成员/函数，因此必须用 `typename` 关键字显式告知编译器这是一个类型。

解决方案：

```shell
// 改前
return Mma::IteratorA(params.params_A,
// 改后
return typename Mma::IteratorA(params.params_A,
```




（42）

```shell
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.cu:480:47: error: function template partial specialization is not allowed
  480 | void CutlassFpAIntBGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag,
      |                                               ^               ~~~~~~~~~~~~~
  481 |                                                                FineGrained>(
      |                                                                ~~~~~~~~~~~~
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.cu:581:47: error: function template partial specialization is not allowed
  581 | void CutlassFpAIntBGemmRunner<T, WeightType>::run_gemm<EpilogueTag,
      |                                               ^       ~~~~~~~~~~~~~
  582 |                                                        FineGrained>(
      |                                                        ~~~~~~~~~~~~
2 warnings and 2 errors generated when compiling for xcn.
```
C++ 标准**禁止函数模板的偏特化（partial specialization）**。上面的写法在类外定义成员函数模板时，函数名后面跟了 `<EpilogueTag, FineGrained>`，编译器将其解析为偏特化语法，因此报错。

实际上这是对成员函数模板的**类外定义（out-of-class definition）**，正确写法**不应该**在函数名后面加模板参数列表。

```shell
// 错误写法（当前代码）
template <typename T, typename WeightType>
template <typename EpilogueTag, bool FineGrained>
void CutlassFpAIntBGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag,
                                                               FineGrained>(...)

// 正确写法（移除函数名后的模板参数）
template <typename T, typename WeightType>
template <typename EpilogueTag, bool FineGrained>
void CutlassFpAIntBGemmRunner<T, WeightType>::dispatch_to_arch(...)
```


（43）

```shell
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/impl/cutlass_forward_bf16_aligned_sm_ma_rf_32x128.cu:3:
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/memory_efficient_variable_attention.h:15:
In file included from /home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/default_fmha_grouped.h:56:
/home/xuanyuan/Paddle/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/gemm/fmha_grouped.h:780:172: error: reference to overloaded function could not be resolved; did you mean to call it?
  780 |         { if (iter_key_start == 0) { constexpr bool kIsFirst = true; ([&] { { if (num_keys - iter_key_start >= kKeysPerBlock) { constexpr bool kFullColumns = true; ([&] { MM0::ScalingCoefsUpdater::update< kQueriesPerBlock, MM0::MmaCore::WarpCount::kCount, MM0::MmaCore::WarpCount::kN, kFullColumns, kIsFirst, kKeepOutputInRF>( accum_o, accum, mi, m_prime, s_prime, shared_storage.addition_storage, lane_id(), thread_id(), warp_id(), num_keys - iter_key_start, iteratorC_tile_offset, kAddMask ? 1.0f : params.scale); })(); } else { constexpr bool kFullColumns = false; ([&] { MM0::ScalingCoefsUpdater::update< kQueriesPerBlock, MM0::MmaCore::WarpCount::kCount, MM0::MmaCore::WarpCount::kN, kFullColumns, kIsFirst, kKeepOutputInRF>( accum_o, accum, mi, m_prime, s_prime, shared_storage.addition_storage, lane_id(), thread_id(), warp_id(), num_keys - iter_key_start, iteratorC_tile_offset, kAddMask ? 1.0f : params.scale); })(); } }; })(); } else { constexpr bool kIsFirst = false; ([&] { { if (num_keys - iter_key_start >= kKeysPerBlock) { constexpr bool kFullColumns = true; ([&] { MM0::ScalingCoefsUpdater::update< kQueriesPerBlock, MM0::MmaCore::WarpCount::kCount, MM0::MmaCore::WarpCount::kN, kFullColumns, kIsFirst, kKeepOutputInRF>( accum_o, accum, mi, m_prime, s_prime, shared_storage.addition_storage, lane_id(), thread_id(), warp_id(), num_keys - iter_key_start, iteratorC_tile_offset, kAddMask ? 1.0f : params.scale); })(); } else { constexpr bool kFullColumns = false; ([&] { MM0::ScalingCoefsUpdater::update< kQueriesPerBlock, MM0::MmaCore::WarpCount::kCount, MM0::MmaCore::WarpCount::kN, kFullColumns, kIsFirst, kKeepOutputInRF>( accum_o, accum, mi, m_prime, s_prime, shared_storage.addition_storage, lane_id(), thread_id(), warp_id(), num_keys - iter_key_start, iteratorC_tile_offset, kAddMask ? 1.0f : params.scale); })(); } }; })(); } };
      |                                                                                                                                                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
`ScalingCoefsUpdater` 是从 `MM0`（一个模板参数依赖的类型）中取出的嵌套类型，`update` 是它的成员函数模板。在这种情况下，C++ 标准要求在调用依赖类型的成员函数模板时，必须加 `template`** 关键字**来消除歧义（告诉编译器 `<` 是模板参数列表的开始，而不是小于号）。

```shell
// 错误（当前写法）
MM0::ScalingCoefsUpdater::update<kQueriesPerBlock, ...>(...)

// 正确写法（需要加 template 关键字）
MM0::ScalingCoefsUpdater::template update<kQueriesPerBlock, ...>(...)
```
