# 基于mpi+cuda加速的霍夫曼编码压缩

## 项目说明

该项目的实现初衷是当前Linux服务器的压缩软件应用一般都是只运行在cpu的，并且甚至一些比较常用的gzip、bzip2、unzip等压缩软件甚至全部都是单核运行的，这对于一些比较大的文件的压缩的速度就会非常受限，并且一些比较知名的压缩软件，比如tar、xz、unrar(RAR5的版本才支持多核、但仍然不支持GPU)等，虽然在多核情况下表现良好，但是对一些比较大的文件（超过1G），它们的性能并不如使用GPU的压缩方式，而一般个人的系统中，CPU、GPU对用户自身都是可见的，个人很容易平衡他们之上的负载，所以为用户提供一个可以支持多核和GPU加速的压缩、解压对于用户根据自身使用的应用情况和设备负载可以自由选择一个合适的压缩/解压缩方式非常有意义

目前本项目的工作主要是：

+ 借鉴了Huffman串行的思想
+ 参考了MPI-Huffman对于Compress/Decompress实现
+ 完成了CUDACompress的实现
+ 完成了MPI对CUDACompress的串行部分的加速
+ 相比如tar在一个比较实用场景的nvhpc的压缩包解压有10.16倍的加速比（tar 36.731s，本项目3.613s）

## 安装运行

### 依赖包含：

```shell
HPCX
CUDA
infiniband mellanox ofed 5.x
```

### 目前我测试成功的依赖版本：

#### 以下环境为后面测试使用的标准环境

软件环境

```
CUDA 11.8
GPU Driver version 520.61.05
MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64
HPCX 2.13.0
```

硬件环境

```
CPU Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
GPU 2 x P100-16G
IB Connet-X 3
```

#### 以下环境为做迁移测试的环境（额外支持，同时在该机器上测试成功了NVLink通信加速支持）

软件环境

```
CUDA 12.0
GPU Driver Version 525.105.17
MLNX_OFED_LINUX-5.4-ubuntu20.04-x86_64
HPCX (bind-to nvhpc 23.1)
```

硬件环境

```
CPU AMD EPYC 7742 64-Core Processor
GPU 8 x A100-SXM-40G
IB Connet-X 5
```

CPU

```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   43 bits physical, 48 bits virtual
CPU(s):                          256
On-line CPU(s) list:             0-255
Thread(s) per core:              2
Core(s) per socket:              64
Socket(s):                       2
NUMA node(s):                    8
Vendor ID:                       AuthenticAMD
CPU family:                      23
Model:                           49
Model name:                      AMD EPYC 7742 64-Core Processor
Stepping:                        0
Frequency boost:                 enabled
CPU MHz:                         3388.216
CPU max MHz:                     2250.0000
CPU min MHz:                     1500.0000
BogoMIPS:                        4491.25
Virtualization:                  AMD-V
L1d cache:                       4 MiB
L1i cache:                       4 MiB
L2 cache:                        64 MiB
L3 cache:                        512 MiB
NUMA node0 CPU(s):               0-15,128-143
NUMA node1 CPU(s):               16-31,144-159
NUMA node2 CPU(s):               32-47,160-175
NUMA node3 CPU(s):               48-63,176-191
NUMA node4 CPU(s):               64-79,192-207
NUMA node5 CPU(s):               80-95,208-223
NUMA node6 CPU(s):               96-111,224-239
NUMA node7 CPU(s):               112-127,240-255
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Not affected
Vulnerability Retbleed:          Vulnerable
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP conditional, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl non
                                 stop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy
                                  abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate sme ssbd mba sev ibrs ibpb sti
                                 bp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero
                                  irperf xsaveerptr wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif umip rdpid overflo
                                 w_recov succor smca

```

GPU topo

```
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	NIC8	NIC9	CPU Affinity	NUMA Affinity
GPU0	 X 	NV12	NV12	NV12	NV12	NV12	NV12	NV12	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	48-63,176-191	3
GPU1	NV12	 X 	NV12	NV12	NV12	NV12	NV12	NV12	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	48-63,176-191	3
GPU2	NV12	NV12	 X 	NV12	NV12	NV12	NV12	NV12	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	16-31,144-159	1
GPU3	NV12	NV12	NV12	 X 	NV12	NV12	NV12	NV12	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	16-31,144-159	1
GPU4	NV12	NV12	NV12	NV12	 X 	NV12	NV12	NV12	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	112-127,240-255	7
GPU5	NV12	NV12	NV12	NV12	NV12	 X 	NV12	NV12	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	112-127,240-255	7
GPU6	NV12	NV12	NV12	NV12	NV12	NV12	 X 	NV12	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	80-95,208-223	5
GPU7	NV12	NV12	NV12	NV12	NV12	NV12	NV12	 X 	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	80-95,208-223	5
NIC0	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS		
NIC1	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	PXB	 X 	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS		
NIC2	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PXB	SYS	SYS	SYS	SYS	SYS	SYS		
NIC3	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	PXB	 X 	SYS	SYS	SYS	SYS	SYS	SYS		
NIC4	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PXB	SYS	SYS	SYS	SYS		
NIC5	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	PXB	 X 	SYS	SYS	SYS	SYS		
NIC6	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PXB	SYS	SYS		
NIC7	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	PXB	 X 	SYS	SYS		
NIC8	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PIX		
NIC9	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PIX	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9

```

![image-20231203174232169](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20231203174232169.png)

### 依赖的安装方式

当前应用部署的操作系统均为Ubuntu20.04，以下安装方式均已Ubuntu20.04为例

（CUDA的安装需要有GPU支持，Infiniband Driver的安装需要机器本身配有IB网卡，HPCX的安装需要本机的IB Driver和CUDA的版本满足要求，也算是进一步要求本机需要配有GPU和IB网卡）

#### CUDA

Driver驱动官网下载（可以直接使用下方的wget命令get到对应的驱动包）

[CUDA Toolkit 11.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux)

这里推荐使用11.8，下载run_file文件

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

然后根据提示安装即可

#### Infiniband Driber

驱动使用官网的下载[Linux InfiniBand Drivers (nvidia.com)](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/)

选择对应的版本和型号后解压，然后进文件夹执行相应的安装脚本，而后按照提示安装即可

```shell
tar -vxzf MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64.tgz
cd MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64
./mlnxofedinstall
```

安装完成后，需要在本机的网络管理器中设置一下IB的IP配置，应用完毕IB的配置后，使用一以下命令重启IB服务

```shell
/etc/init.d/openibd restart
/etc/init.d/opensmd restart
```

而后可以使用ibstat查看ib状态

```
CA 'mlx4_0'
	CA type: MT4099
	Number of ports: 1
	Firmware version: 2.42.5000
	Hardware version: 1
	Node GUID: 0x98039b0300dccab0
	System image GUID: 0x98039b0300dccab3
	Port 1:
		State: Active
		Physical state: LinkUp
		Rate: 56
		Base lid: 1
		LMC: 0
		SM lid: 1
		Capability mask: 0x0251486a
		Port GUID: 0x98039b0300dccab1
		Link layer: InfiniBand
```

#### HPCX

HPCX官网下载[HPC-X | NVIDIA | NVIDIA Developer](https://developer.nvidia.com/networking/hpc-x)

![image-20231203175812788](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20231203175812788.png)

安装过程解压即可

```
tar -xvf hpcx.tbz
```

具体的安装过程和使用方式可以参考[Installing and Loading HPC-X - NVIDIA Docs](https://docs.nvidia.com/networking/display/hpcxv212/installing+and+loading+hpc-x)

个人推荐使用的方式是modulefile，通过指定module_path和HPCX_HOME环境变量来加载管理

```shell
export HPCX_HOME=/home/oyhd/hpcx-v2.13-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.12-x86_64
export MODULEPATH=$MODULEPATH:$HPCX_HOME/modulefiles
```

而后使用时

```shell
module load hpcx
```

不使用时

```shell
module unload hpcx
```

### 项目的安装方式

项目因为压缩需要测试的文件是直接写入和随机写入的，因为为了测试大文件性能，这些随机生成文件的自身大小都非常大，无法放到项目目录中上传，同时直接通过上传下载的方式并不如直接生成一个磁盘读写的文件速度快，因此写了一个生成测试文件的脚本，通过运行脚本生成对应的测试文件和目录

```shell
mkdir bin
mkdir logs
mkdir Testfiles
cd Testfiles
python3 ../scripts/generateFiles.py
cd ..
make
```

(需要事先module load hpcx，同时将CUDA toolkit加入环境变量)

然后运行可以在项目文件夹目录下，使用以下脚本即可运行相应测试程序（以下脚本本身也是基于Gaussian16的作业批处理管理系统（PBS），可以直接使用该脚本提交到PBS上运行）

```shell
./scripts/CUDAMPI.pbs
./scripts/CUDA.pbs
./scripts/MPI.pbs
./scripts/Serial.pbs
```

## 代码分析

### 代码架构

为了更好的演示和对比测试，在项目里添加了串行（Serial）、MPI、CUDA、CUDAMPI的支持

```shell
MPI-GPU-Compress/
├── bin
│   ├── compress
│   ├── CUDA_compress
│   ├── CUDAMPI_compress
│   ├── decompress
│   ├── MPI_compress
│   └── MPI_decompress
├── CUDA
│   ├── CUDACompress.cu
│   └── Makefile
├── CUDAMPI
│   ├── CUDAMPICompress.cu
│   └── Makefile
├── include
│   ├── GPUWrapper.cu
│   ├── kernel.cu
│   ├── parallelFunctions.cu
│   ├── parallelHeader.h
│   ├── serialFunctions.c
│   └── serialHeader.h
├── logs
│   ├── CUDAMPI.txt
│   ├── CUDA.txt
│   ├── MPI.txt
│   └── Serial_compress.txt
├── Makefile
├── MPI
│   ├── Makefile
│   ├── MPICompress.c
│   └── MPIDecompress.c
├── scripts
│   ├── CUDAMPI.pbs
│   ├── CUDA.pbs
│   ├── generateFiles.py
│   ├── MPI.pbs
│   └── Serial.pbs
├── Serial
│   ├── compress.c
│   ├── decompress.c
│   └── Makefile
└── TestFiles
    ├── mb1024
    ├── mb128
    ├── mb1280
    ├── mb1280_comp
    ├── mb128_comp
    ├── mb1536
    ├── mb1536_comp
    ├── mb1792
    ├── mb1792_comp
    ├── mb2048
    ├── mb2048_comp
    ├── mb256
    ├── mb512
    ├── mb5120
    ├── mb5120_comp
    ├── mb64
    └── mb768
```

### Common Design

#### 架构和Makefile

考虑到均衡各个src的设计，就将多个src设计均放到了顶层，共享一套include files，并根据这一规则书写了Makefile

+ 顶层

设计为CUDAMPI  CUDA  MPI  Serial实现，为四者均实现了一个子目录的递归编译

```makefile
SUBDIRS = CUDAMPI CUDA MPI Serial

all:
	@for dir in $(SUBDIRS); do \
		make -C $$dir; \
	done

clean:
	@for dir in $(SUBDIRS); do \
		make clean -C $$dir; \
	done

.PHONY: all clean $(SUBDIRS)
```

+ 项目内（CUDAMPI举例）

书写相应的编译链接规则，以及所包含的编译文件树

```makefile
NVCC = nvcc
SRC_DIR = .
INCLUDE_DIR = ../include
BIN_DIR = ../bin

CU_FILES = $(SRC_DIR)/CUDAMPICompress.cu $(INCLUDE_DIR)/parallelFunctions.cu $(INCLUDE_DIR)/GPUWrapper.cu $(INCLUDE_DIR)/kernel.cu

all: CUDAMPI_compress

CUDAMPI_compress: $(CU_FILES)
	$(NVCC) -dc $^
	$(NVCC) *.o -lmpi -o $(BIN_DIR)/CUDAMPI_compress
	rm -f *.o

clean:
	rm -f $(BIN_DIR)/CUDAMPI_compress

.PHONY: all clean
```

#### include

+ GPUWrapper.cu

  为了对文件情况的支持性考虑，需要简单对可能的情况分一个类（数据长度使用unsigned int方式存储，可能会导致int溢出；单个gpu显存有限，可能需要多次运行kernel分批处理）

  故在创建数组时和运行kernel时做了以下处理

  ```c++
  // generate offset
  if (integerOverflowFlag == 0) {
      // only one time run of kernel
      if (numKernelRuns == 1) {
          createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength);
      }
      // multiple run of kernel due to larger file or smaller gpu memory
      else {
          // ...
          createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, gpuMemoryOverflowIndex, gpuBitPaddingFlag, mem_req);
      }
  } else {
      // overflow occurs and single run
      if (numKernelRuns == 1) {
          // ...
          createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, integerOverflowIndex, bitPaddingFlag, 10240);
      }
      // overflow occurs and multiple run
      else {
          // ...
          createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, integerOverflowIndex, bitPaddingFlag, gpuMemoryOverflowIndex, gpuBitPaddingFlag, 10240, mem_req);
      }
  }
  ```

  内存申请和拷贝

  ```c++
  // GPU initiation
  {	
      // allocate memory for input data, offset information and dictionary
      error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
      // ...
      error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
      // ...
      error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDictionary));
      // ...
      // memory copy input data, offset information and dictionary
      error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
      // ...
      error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
      // ...
      error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDictionary), cudaMemcpyHostToDevice);
      // ...
      // copy constant memory if required for dictionary
      if (constMemoryFlag == 1) {
          error = cudaMemcpyToSymbol(d_bitSequenceConstMemory, bitSequenceConstMemory, 256 * 255 * sizeof(unsigned char));
          // ...
      }
  }
  ```

  而后便是执行kernel和释放内存

+ kernel.cu

  kernel.cu中便是实现compress函数的位置，分别需要实现以上提到的4种kernel

  以multiple run and with overflow（出现整数表示溢出和GPU显存溢出）为例讲解目前所实现的功能

  首先期望可以支持块内的share_memory，因此需要完成块内的共享内存和溢出判断：

  ```c++
  	// when shared memory is sufficient
    	if(*constMemoryFlag* == 0){
  		// ...
      }
  	// use constant memory and shared memory
  	else{
          // ...
      }
  	__syncthreads();
  ```

  然后在每种情况下，根据table中的字段对数据进行压缩

  ```c++
  for(i = pos + d_lowerPosition; i < overflowPosition; i += blockDim.x){
  			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
  				d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
  			}
  		}
  		for(i = overflowPosition + pos; i < d_upperPosition - 1; i += blockDim.x){
  			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i + 1]]; k++){
  				d_temp_overflow[d_compressedDataOffset[i + 1] + k] = table.bitSequence[d_inputFileData[i + 1]][k];
  			}
  		}
  		if(pos == 0){
  			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], 
  				   &table.bitSequence[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
  		}
  		if(pos == 0 && d_lowerPosition != 0){
  			memcpy(&d_byteCompressedData[d_compressedDataOffset[(d_lowerPosition)] - table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]], 
  				   &table.bitSequence[d_inputFileData[d_lowerPosition - 1]], table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]);
  		}
  ```

  将压缩后的数据进行位操作，将各个位按字节组合

  ```c++
  	for(i = pos * 8; i < d_compressedDataOffset[overflowPosition]; i += blockDim.x * 8){
  		for(j = 0; j < 8; j++){
  			if(d_byteCompressedData[i + j] == 0){
  				d_inputFileData[(i / 8)] = d_inputFileData[(i / 8)] << 1;
  			}
  			else{
  				d_inputFileData[(i / 8)] = (d_inputFileData[i / 8] << 1) | 1;
  			}
  		}
  	}
  
  	offset_overflow = d_compressedDataOffset[overflowPosition] / 8;
  	
  	for(i = pos * 8; i < d_compressedDataOffset[d_upperPosition]; i += blockDim.x * 8){
  		for(j = 0; j < 8; j++){
  			if(d_temp_overflow[i + j] == 0){
  				d_inputFileData[(i / 8) + offset_overflow] = d_inputFileData[(i / 8) + offset_overflow] << 1;
  			}
  			else{
  				d_inputFileData[(i / 8) + offset_overflow] = (d_inputFileData[(i / 8) + offset_overflow] << 1) | 1;
  			}
  		}
  	}
  ```

+ parallelFunctions.cu

  sortHuffmanTree 函数:

  ```
  cppCopy code
  void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes)
  ```

  - 该函数根据频率对 Huffman 树节点进行排序
  - huffmanTreeNode 是一个全局数组，包含 Huffman 树节点的信息，节点结构为 struct huffmanTree
  - i 表示已合并的 Huffman 节点的数量，distinctCharacterCount 表示不同字符的数量，combinedHuffmanNodes 表示已合并的 Huffman 节点的起始位置

  

  buildHuffmanTree 函数:

  ```
  cppCopy code
  void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes)
  ```

  - 该函数根据 sortHuffmanTree 的结果构建 Huffman 树
  - huffmanTreeNode 是一个全局数组，包含 Huffman 树节点的信息，节点结构为 struct huffmanTree
  - i 表示已合并的 Huffman 节点的数量，distinctCharacterCount 表示不同字符的数量，combinedHuffmanNodes 表示已合并的 Huffman 节点的起始位置
  - 函数通过合并频率最低的两个节点构建 Huffman 树

  

  buildHuffmanDictionary 函数:

  ```
  cppCopy code
  void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength)
  ```

  - 该函数根据 Huffman 树构建 Huffman 编码表
  - root 是 Huffman 树的根节点，bitSequence 用于存储当前字符的 Huffman 编码，bitSequenceLength 表示编码的长度
  - 递归地遍历 Huffman 树，生成每个字符的 Huffman 编码，并将结果存储在全局变量 huffmanDictionary 中
  - 如果编码长度小于 192，将完整的编码存储在 huffmanDictionary 中；否则，存储在 bitSequenceConstMemory 中，同时设置 constMemoryFlag

  

  createDataOffsetArray 函数 (单次运行，无溢出):

  ```
  cppCopy code
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength)
  ```

  - 生成数据偏移数组，用于压缩数据
  - compressedDataOffset 存储每个字符对应的压缩后数据的起始偏移
  - inputFileData 是输入的原始数据
  - 如果最后一个偏移不是 8 的倍数，会进行相应的填充

  

  createDataOffsetArray 函数 (单次运行，有溢出):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, int numBytes)
  ```

  - 生成数据偏移数组，考虑到溢出情况
  - compressedDataOffset 存储每个字符对应的压缩后数据的起始偏移
  - inputFileData 是输入的原始数据
  - integerOverflowIndex 存储整数溢出的位置，bitPaddingFlag 存储相应位置是否需要填充
  - numBytes 表示每个线程块可处理的最大字节数

  

  createDataOffsetArray 函数 (多次运行，无溢出):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, long unsigned int mem_req)
  ```

  - 生成数据偏移数组，考虑到多次运行和 GPU 内存的限制，无溢出情况
  - compressedDataOffset 存储每个字符对应的压缩后数据的起始偏移
  - inputFileData 是输入的原始数据
  - gpuMemoryOverflowIndex 存储 GPU 内存溢出的位置，gpuBitPaddingFlag 存储相应位置是否需要填充
  - mem_req 表示每次运行的 GPU 内存限制

  

  createDataOffsetArray 函数 (多次运行，有溢出):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, 
                              unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, int numBytes, long unsigned int mem_req)
  ```

  - 生成数据偏移数组，考虑到多次运行和 GPU 内存的限制，有溢出情况
  - compressedDataOffset 存储每个字符对应的压缩后数据的起始偏移
  - inputFileData 是输入的原始数据
  - integerOverflowIndex 存储整数溢出的位置，bitPaddingFlag 存储相应位置是否需要填充
  - gpuMemoryOverflowIndex 存储 GPU 内存溢出的位置，gpuBitPaddingFlag 存储相应位置是否需要填充
  - numBytes 表示每个线程块可处理的最大字节数
  - mem_req 表示每次运行的 GPU 内存限制

+ parallelHeader.h
  parallelHeader 中是定义了一些共用型的函数和变量，并引入了一些通用性质的头文件，期望可以在CUDA内部的程序和CUDAMPI程序中可以仅实现相应的启动程序

  ```c++
  #ifndef PARALLEL_HEADER_H
  #define PARALLEL_HEADER_H
  
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <limits.h>
  #include <time.h>
  #include <math.h>
  
  #define BLOCK_SIZE 1024
  #define MIN_SCRATCH_SIZE (50 * 1024 * 1024)
  
  struct huffmanDictionary{
  	unsigned char bitSequence[256][191];
  	unsigned char bitSequenceLength[256];
  };
  
  struct huffmanTree{
  	unsigned char letter;
  	unsigned int count;
  	struct huffmanTree *left, *right;
  };
  
  extern struct huffmanTree *head_huffmanTreeNode;
  extern struct huffmanTree huffmanTreeNode[512];
  extern struct huffmanDictionary huffmanDictionary;
  extern unsigned int constMemoryFlag;
  extern unsigned char bitSequenceConstMemory[256][255];
  extern __constant__ unsigned char d_bitSequenceConstMemory[256][255];
  
  void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes);
  void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes);
  void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength);
  int wrapperGPU(char **file, unsigned char *inputFileData, int inputFileLength);
  
  __global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int constMemoryFlag);
  __global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_inputFileLength, unsigned int constMemoryFlag, unsigned int overflowPosition);
  __global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_lowerPosition, unsigned int constMemoryFlag, unsigned int d_upperPosition);
  __global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_lowerPosition, unsigned int constMemoryFlag, unsigned int d_upperPosition, unsigned int overflowPosition);
  
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength);
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, long unsigned int mem_req);
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, int numBytes);
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, int numBytes, long unsigned int mem_req);
  
  void lauchCUDAHuffmanCompress(unsigned char *inputFileData, unsigned int *compressedDataOffset, unsigned int inputFileLength, int numKernelRuns, unsigned int integerOverflowFlag, long unsigned int mem_req);
  
  #endif // PARALLEL_HEADER_H
  ```

+ 其他的文件（serialFunctions.c serialHeader.h）
  也是类似于cuda部分，将一些CPU串行的函数部分放在了include下（由于不是本项目的工作，不详细讲解了）

#### tests

tests是为每个样例（MPI分了一次PROC，然后总体均分成了基于不同的FILE_SIZE，并且对每一种情况都运行了3次）

```shell
#!/bin/bash
#PBS -N CUDAMPI_job
#PBS -l select=16:ncpus=16:mpiprocs=1:ngpus=2:mem=256gb:phase=15
#PBS -l walltime=04:00:00

NUM_RUNS=3
SIZES=(64 128 256 512 768 1024)

echo "Run: CUDAMPI" >> logs/CUDAMPI.txt
echo "Resource: select=32:ncpus=2:mpiprocs=1:mem=124gb:phase=10" >> logs/CUDAMPI.txt

for FILE_SIZE in "${SIZES[@]}"; do
    echo ' ' >> logs/CUDAMPI.txt
    echo "FileSize: ${FILE_SIZE}MB" >> logs/CUDAMPI.txt

    for PROCS in 1 2 4 8 16; do
        echo "MPIPROCS: ${PROCS}" >> logs/CUDAMPI.txt
        for ((i=0; i<$NUM_RUNS; i++)); do
            mpirun -np ${PROCS} ./bin/CUDAMPI_compress "TestFiles/mb${FILE_SIZE}" "TestFiles/mb${FILE_SIZE}_comp" >> logs/CUDAMPI.txt
            rm "TestFiles/mb${FILE_SIZE}_comp"
        done
    done
done
```

### CUDA

CUDA部分主要为CUDACompress.cu 代码主要实现了对输入文件进行 Huffman 压缩的流程，本项目的代码中将各个功能均分离成单个模块

1. **读取输入文件：**
   - readInputFile 函数用于从文件中读取输入数据
   - 打开文件，获取文件长度，分配内存，读取文件内容
2. **计算字符频率：**
   - calculateFrequency 函数统计输入数据中每个字符的频率
   - 使用数组 frequency记录每个字符的出现次数
3. **初始化 Huffman 树节点：**
   - initializeHuffmanTreeNodes函数初始化 Huffman 树的节点
   - 根据字符频率创建叶子节点，并记录叶子节点的数量
4. **构建 Huffman 树：**
   - buildHuffmanTreeNodes函数通过合并频率最低的两个节点来构建 Huffman 树
   - 通过调用 sortHuffmanTree和 buildHuffmanTree函数，循环构建 Huffman 树
5. **构建 Huffman 编码表：**
   - buildHuffmanDictionary函数根据 Huffman 树构建 Huffman 编码表
   - 通过递归遍历 Huffman 树，生成每个字符的 Huffman 编码，并记录编码长度
   - Huffman 编码存储在全局变量 huffmanDictionary中
6. **计算内存需求：**
   - calculateMemoryRequirements函数计算压缩过程中所需的 GPU 内存
   - 计算数据偏移数组的内存需求，检查 GPU 是否有足够的内存来存储编码后的数据
   - 调用 lauchCUDAHuffmanCompress 函数执行 GPU 压缩
7. **写入输出文件：**
   - writeOutputFile函数将压缩后的数据写入输出文件
   - 写入输入文件长度、字符频率和压缩后的数据
8. **主函数 - `main`：**
   - 从命令行参数中获取输入和输出文件名
   - 读取输入文件，计算字符频率，初始化 Huffman 树节点，构建 Huffman 树，构建 Huffman 编码表
   - 计算 GPU 内存需求，执行 GPU 压缩
   - 写入输出文件，输出程序运行时间

### CUDAMPI

1. **MPI 初始化：**
   - MPI_Init 函数初始化 MPI 环境
   - 获取当前进程的 rank 和总进程数
2. **读取输入文件：**
   - 使用 MPI 文件 I/O 打开文件，读取本地进程负责的数据块
   - 使用 MPI_File_open、MPI_File_seek 和 MPI_File_read 函数
3. **统计字符频率：**
   - 每个进程统计其负责的数据块中字符的频率
   - frequency数组记录每个字符的出现次数
4. **构建 Huffman 树：**
   - 各进程独立统计频率后，通过 MPI 函数进行归约操作，将各进程的频率相加，得到全局频率
   - 主进程（rank == 0）构建 Huffman 树，然后通过广播 head_huffmanTreeNode 给其他进程
5. **计算内存需求：**
   - cudaMemGetInfo函数获取 GPU 内存信息
   - 计算偏移数组的内存需求，检查 GPU 是否有足够的内存存储编码后的数据
   - 使用 MPI_Bcast将内存需求信息广播给所有进程
6. **调用 GPU 压缩 Kernel：**
   - 各进程根据其负责的数据块调用 GPU 压缩的 Kernel
   - lauchCUDAHuffmanCompress 函数进行 GPU 压缩
7. **计算各进程压缩数据的长度：**
   - 主进程计算每个进程压缩后数据的长度，然后通过 MPI_Gather将这些长度收集到主进程
   - 更新数据长度以反映偏移
8. **MPI 文件 I/O：写入输出文件：**
   - 主进程使用 MPI 文件 I/O 将压缩后的数据写入输出文件
   - MPI_File_open、MPI_File_seek和 MPI_File_write函数用于写入文件
9. **退出环境，终止MPI ：**
   - MPI_Finalize终止 MPI 环境，并返回

## 性能评估

由于经常可以使用的测试机器上的用户运行内存只可以支持到使用768MB的文件，所以目前的测试都是对（64,128,256,512,768）而进行的性能评估

### 串行性能

```shell
Run: Serial
Resource: select=1:ncpus=20:mem=124gb:phase=15
 
FileSize: 64MB
Time taken: 7:349 s
Time taken: 7:350 s
Time taken: 7:349 s
 
FileSize: 128MB
Time taken: 14:695 s
Time taken: 14:677 s
Time taken: 14:683 s
 
FileSize: 256MB
Time taken: 29:379 s
Time taken: 29:344 s
Time taken: 29:381 s
 
FileSize: 512MB
Time taken: 58:969 s
Time taken: 58:986 s
Time taken: 58:969 s
 
FileSize: 768MB
Time taken: 88:924 s
Time taken: 88:931 s
Time taken: 88:909 s
```

### CUDA性能

```shell
Run: CUDA
Resource: select=1:ncpus=16:ngpus=2:mem=256gb:phase=15

FileSize: 64MB
Time taken: 2:289 s
Time taken: 2:296 s
Time taken: 2:286 s
 
FileSize: 128MB
Time taken: 3:506 s
Time taken: 3:505 s
Time taken: 3:513 s
 
FileSize: 256MB
Time taken: 6:187 s
Time taken: 6:177 s
Time taken: 6:175 s
 
FileSize: 512MB
Time taken: 13:224 s
Time taken: 13:241 s
Time taken: 13:223 s
 
FileSize: 768MB
Time taken: 19:434 s
Time taken: 19:442 s
Time taken: 19:620 s
```

### CUDAMPI性能

```shell
Run: CUDAMPI
Resource: select=32:ncpus=2:mpiprocs=1:mem=124gb:phase=10
 
FileSize: 64MB
MPIPROCS: 1
Time taken: 1:294 s
Time taken: 1:295 s
Time taken: 1:298 s
MPIPROCS: 2
Time taken: 1:14 s
Time taken: 1:17 s
Time taken: 1:17 s
MPIPROCS: 4
Time taken: 0:857 s
Time taken: 0:853 s
Time taken: 0:854 s
MPIPROCS: 8
Time taken: 0:806 s
Time taken: 0:807 s
Time taken: 0:808 s
MPIPROCS: 16
Time taken: 0:766 s
Time taken: 0:776 s
Time taken: 0:760 s
 
FileSize: 128MB
MPIPROCS: 1
Time taken: 2:534 s
Time taken: 2:533 s
Time taken: 2:534 s
MPIPROCS: 2
Time taken: 1:943 s
Time taken: 1:944 s
Time taken: 1:943 s
MPIPROCS: 4
Time taken: 1:629 s
Time taken: 1:629 s
Time taken: 1:635 s
MPIPROCS: 8
Time taken: 1:491 s
Time taken: 1:490 s
Time taken: 1:499 s
MPIPROCS: 16
Time taken: 1:434 s
Time taken: 1:447 s
Time taken: 1:430 s
 
FileSize: 256MB
MPIPROCS: 1
Time taken: 5:19 s
Time taken: 5:16 s
Time taken: 5:14 s
MPIPROCS: 2
Time taken: 3:857 s
Time taken: 3:857 s
Time taken: 3:856 s
MPIPROCS: 4
Time taken: 3:155 s
Time taken: 3:158 s
Time taken: 3:153 s
MPIPROCS: 8
Time taken: 2:881 s
Time taken: 2:891 s
Time taken: 2:896 s
MPIPROCS: 16
Time taken: 2:776 s
Time taken: 2:757 s
Time taken: 2:736 s
 
FileSize: 512MB
MPIPROCS: 1
Time taken: 11:554 s
Time taken: 11:562 s
Time taken: 11:554 s
MPIPROCS: 2
Time taken: 7:602 s
Time taken: 7:680 s
Time taken: 7:685 s
MPIPROCS: 4
Time taken: 6:292 s
Time taken: 6:295 s
Time taken: 6:271 s
MPIPROCS: 8
Time taken: 5:680 s
Time taken: 5:706 s
Time taken: 5:701 s
MPIPROCS: 16
Time taken: 5:394 s
Time taken: 5:395 s
Time taken: 5:418 s
 
FileSize: 768MB
MPIPROCS: 1
Time taken: 17:539 s
Time taken: 17:534 s
Time taken: 17:540 s
MPIPROCS: 2
Time taken: 11:439 s
Time taken: 11:486 s
Time taken: 11:437 s
MPIPROCS: 4
Time taken: 9:310 s
Time taken: 9:429 s
Time taken: 9:426 s
MPIPROCS: 8
Time taken: 8:486 s
Time taken: 8:470 s
Time taken: 8:516 s
MPIPROCS: 16
Time taken: 7:994 s
Time taken: 8:50 s
Time taken: 8:12 s
```

可以看到CUDA的直接实现相比与串行方式就获得了4.53倍的加速比，在CUDAMPI版本内，由于多进程的IO处理，使得IO和压缩后的Gather部分的性能有明显提高，并且可以发现使用MPI内建函数实现的IO优化，即便是单核性能相比于CUDA也有一定的提高（MPI的内建函数很多会基于宏，并且对于文件seek、find都是做了对数级的优化的）

## 未来的工作

目前项目是使用MPI + OpenMP实现串行部分的某些操作，目前由于其多进程特征比较明显，以及期望练习一下MPI + CUDA的分布式训练，但最终没有完成多卡版本，同时也没有进一步在多线程角度进行优化。但目前的实现方案，对于后续可能启动的多机支持的效果会更好，只不过当下最实用的场景还是多核 + 单卡的一般性设备，同时也可以方便一些日常使用Linux作为主要工作系统的用户有一个普遍的压缩加速支持

未来的工作可能集中在以下几个方向：

+ 对于编码部分的处理，目前是均使用串行在MPI上实现，可以引入OpenMP可以大幅提高编码部分的速度，同时对于一些比较大的文件，可以考虑在编码部分就启用GPU支持
+ 目前在多卡的设备上仍然没有一个完整的支持，原计划的确希望可以对于目前服务器这种单机双卡的架构进行一个分布式优化，考虑过将MPI进程同时绑定在一个GPU上，但实际的压缩/解压性能其实目前的瓶颈并不在于GPU的压缩计算上，通过MPI的多进程改善IO的效果要比使用两张卡的性能好很多（最开始使用MPI + CUDA的时候就是使用两个线程绑定，但是后续发现性能不如直接改善IO，而也因此同时发现了多进程的方案并没那么好，或许当下多线程会是优选，同时也是因此而发现了MPI + CUDA想对于多卡支持更好则应当需要一个灵活配置的过程，这对于目前的代码来说在支持性的工作量有些过高，目前还未能成功实现）
+ 中间尝试过使用NVLink的GPU Direct加速GPU之间的通信，可以显著提高多卡GPU压缩性能，但对于没有NVLink的时候则没那么通用，后期项目Debug的时候并没有保持相关的支持，现被遗弃，后面有时间期望可以补充上
+ 目前对于整个项目的部署支持上感觉还是不够好，目前正在考虑docker方案，以及应用打包方案等等。当前的项目是使用Nvidia的整套MPI + CUDA的工具库，当下HPCX对于项目的MPI封装来说仍然是一个必需品，直接使用OpenMPI并不能直接完成相关代码的编译运行，同时MPI部分代码到后期因为环境问题，我自己都已经跑不起来实验了，目前影响因素仍然未知，纯MPI部分代码还有待后续测试
+ 对于压缩、解压缩的文件格式需要额外支持，这一点目前仅是将文件当做字节码的方式输入、还原，并不支持一个普遍的压缩、解压缩的文件格式。换句话说就是，当前只能支持两台同样配置了本项目的机器上通过该项目可以压缩后传输文件到另一台机器上并使用该项目解压缩，想变得更实用则需要对相关的文件类型提供一个额外的支持