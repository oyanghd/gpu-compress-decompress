# gpu-compress-decompress

[English](project_description_EN.md) | [中文](project_description_ZH.md)

## Project Description

The initial intent of this project is to address the fact that most compression software applications on Linux servers run only on CPUs. Even commonly used compression tools like gzip, bzip2, and unzip operate on a single core, which significantly limits the speed of compressing large files. While some well-known compression software such as tar, xz, and unrar (RAR5 version supports multicore but still not GPU), perform well with multicore systems, they are not as efficient as GPU-based compression methods for very large files (over 1GB). In most personal systems, both the CPU and GPU are accessible to the user, making it easy for individuals to balance the load between them. Therefore, providing users with a compression/decompression tool that supports multicore and GPU acceleration would be very meaningful, allowing users to choose a suitable compression/decompression method based on their application needs and device load.

The current work of this project primarily includes:

- Drawing inspiration from the serial idea of Huffman coding
- Referring to MPI-Huffman for Compress/Decompress implementation
- Completing the implementation of CUDACompress
- Accelerating the serial part of CUDACompress with MPI
- Comparing with tar in a practical scenario

## Installation and Operation

### Dependencies include:

```shell
HPCX
CUDA
infiniband mellanox ofed 5.x
```

### Currently tested successful dependency versions:

#### The following environment is used as the standard for later tests

Software environment:

```
CUDA 11.8
GPU Driver version 520.61.05
MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64
HPCX 2.13.0
```

Hardware environment:

```
CPU Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
GPU 2 x P100-16G
IB Connet-X 3
```

#### The following environment is the environment for migration testing (additional support, and NVLink communication acceleration support was successfully tested on this machine)

Software Environment

```
CUDA 12.0
GPU Driver Version 525.105.17
MLNX_OFED_LINUX-5.4-ubuntu20.04-x86_64
HPCX (bind-to nvhpc 23.1)
```

Hardware environment:

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

### Dependency installation method

The operating systems currently used for application deployment are Ubuntu20.04. The following installation methods are all based on Ubuntu20.04 as an example.

(The installation of CUDA requires GPU support, the installation of Infiniband Driver requires the machine itself to be equipped with an IB network card, and the installation of HPCX requires that the version of the IB Driver and CUDA of the machine meet the requirements, which is also a further requirement that the machine needs to be equipped with a GPU and IB network card. )

#### CUDA

Download the Driver from the official website (you can directly use the wget command below to get the corresponding driver package)

[CUDA Toolkit 11.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux)

It is recommended to use 11.8 here, download the run_file file

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Then follow the prompts to install it

#### Infiniband Driber

Download the driver from the official website [Linux InfiniBand Drivers (nvidia.com)](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/)

Select the corresponding version and model, unzip it, then enter the folder to execute the corresponding installation script, and then follow the prompts to install it.

```shell
tar -vxzf MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64.tgz
cd MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64
./mlnxofedinstall
```

After the installation is complete, you need to set the IP configuration of IB in the local network manager. After applying the IB configuration, use the following command to restart the IB service.

```shell
/etc/init.d/openibd restart
/etc/init.d/opensmd restart
```

Then you can use ibstat to view the ib status

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

HPCX official website download [HPC-X | NVIDIA | NVIDIA Developer](https://developer.nvidia.com/networking/hpc-x)

![image-20231203175812788](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20231203175812788.png)

Just unzip the installation process

```
tar -xvf hpcx.tbz
```

For the specific installation process and usage, please refer to [Installing and Loading HPC-X - NVIDIA Docs](https://docs.nvidia.com/networking/display/hpcxv212/installing+and+loading+hpc-x)

The personally recommended method is modulefile, which is loaded and managed by specifying module_path and HPCX_HOME environment variables.

```shell
export HPCX_HOME=/home/oyhd/hpcx-v2.13-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.12-x86_64
export MODULEPATH=$MODULEPATH:$HPCX_HOME/modulefiles
```

and then use

```shell
module load hpcx
```

when not in use

```shell
module unload hpcx
```

### How to install the project

The files that the project needs to test due to compression are written directly and randomly. Because in order to test the performance of large files, the sizes of these randomly generated files are very large and cannot be uploaded in the project directory. At the same time, they can be directly uploaded and downloaded. It is not as fast as directly generating a disk read and write file, so I wrote a script to generate test files, and generated the corresponding test files and directories by running the script.

```shell
mkdir bin
mkdir logs
mkdir Testfiles
cd Testfiles
python3 ../scripts/generateFiles.py
cd ..
make
```

(You need to module load hpcx in advance and add CUDA toolkit to the environment variable)

Then you can run the corresponding test program in the project folder directory and use the following script (the following script itself is also based on the Gaussian16 job batch management system (PBS) and can be directly submitted to PBS for running using this script)

```shell
./scripts/CUDAMPI.pbs
./scripts/CUDA.pbs
./scripts/MPI.pbs
./scripts/Serial.pbs
```

## Code analysis

### Code structure

For better demonstration and comparative testing, support for Serial, MPI, CUDA, and CUDAMPI has been added to the project.

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

#### Architecture and Makefile

Considering the balanced design of each src, I put multiple src designs on the top level, shared a set of include files, and wrote the Makefile according to this rule

+ top
  Designed to be implemented as CUDAMPI CUDA MPI Serial, recursive compilation of a subdirectory is implemented for all four

  ```shell
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

+ Within the project (CUDAMPI example)
  Write the corresponding compilation and linking rules, as well as the included compilation file tree

  ```shell
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

  In order to support the file situation, it is necessary to simply classify the possible situations into one category (the data length is stored in unsigned int mode, which may cause int overflow; a single GPU memory is limited, and the kernel batch processing may need to be run multiple times)

  Therefore, the following processing is done when creating the array and when running the kernel.

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

  Memory alloc and copying

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

  Then the kernel is executed and the memory is released.

+ kernel.cu

  Kernel.cu is where the compress function is implemented. The four kernels mentioned above need to be implemented respectively.

  Taking multiple run and with overflow (the occurrence of integers indicating overflow and GPU memory overflow) as an example to explain the currently implemented functions

  First, it is expected that share_memory within the block can be supported, so shared memory and overflow judgment within the block need to be completed:

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

  Then in each case, the data is compressed based on the fields in the table

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

  Perform bit operations on the compressed data and combine each bit into bytes

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

  sortHuffmanTree  function:

  ```
  cppCopy code
  void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes)
  ```

  - This function sorts Huffman tree nodes according to frequency
  - huffmanTreeNode is a global array containing information about Huffman tree nodes. The node structure is struct huffmanTree
  - i represents the number of merged Huffman nodes, distinctCharacterCount represents the number of distinct characters, combinedHuffmanNodes represents the starting position of the merged Huffman nodes

  

  buildHuffmanTree function:

  ```
  cppCopy code
  void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes)
  ```

  - This function builds a Huffman tree based on the results of sortHuffmanTree
  - huffmanTreeNode is a global array containing information about Huffman tree nodes. The node structure is struct huffmanTree
  - i represents the number of merged Huffman nodes, distinctCharacterCount represents the number of distinct characters, combinedHuffmanNodes represents the starting position of the merged Huffman nodes
  - Function builds a Huffman tree by merging the two nodes with the lowest frequency

  

  buildHuffmanDictionary function:

  ```
  cppCopy code
  void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength)
  ```

  - This function builds a Huffman coding table based on a Huffman tree
  - root is the root node of the Huffman tree, bitSequence is used to store the Huffman encoding of the current character, bitSequenceLength represents the length of the encoding
  - Recursively traverse the Huffman tree, generate the Huffman encoding of each character, and store the result in the global variable huffmanDictionary
  - If the encoding length is less than 192, store the complete encoding in huffmanDictionary; otherwise, store it in bitSequenceConstMemory and set constMemoryFlag

  

  createDataOffsetArray Function (single run, no overflow):

  ```
  cppCopy code
  void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength)
  ```

  - Generate data offset array for compressing data
  - compressedDataOffset stores the starting offset of the compressed data corresponding to each character
  - inputFileData is the original data entered
  - If the last offset is not a multiple of 8, padding will be done accordingly

  

  createDataOffsetArray function (single run, with overflow):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, int numBytes)
  ```

  - Generate data offset array, taking into account overflow situations
  - compressedDataOffset stores the starting offset of the compressed data corresponding to each character
  - inputFileData is the original data entered
  - integerOverflowIndex stores the position where the integer overflows, and bitPaddingFlag stores whether the corresponding position needs to be filled.
  - numBytes represents the maximum number of bytes that each thread block can process

  

  createDataOffsetArray function (run multiple times, no overflow):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, long unsigned int mem_req)
  ```

  - Generate data offset arrays, taking into account multiple runs and GPU memory limitations, without overflow situations
  - compressedDataOffset stores the starting offset of the compressed data corresponding to each character
  - inputFileData is the original data entered
  - gpuMemoryOverflowIndex stores the location where GPU memory overflows, and gpuBitPaddingFlag stores whether the corresponding location needs to be filled.
  - mem_req represents the GPU memory limit per run

  

  createDataOffsetArray function (run multiple times, with overflow):

  ```
  cppCopy codevoid createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, 
                              unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, 
                              unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, int numBytes, long unsigned int mem_req)
  ```

  - Generate data offset array, taking into account multiple runs and GPU memory limitations, there is an overflow situation
  - compressedDataOffset stores the starting offset of the compressed data corresponding to each character
  - inputFileData is the original data entered
  - integerOverflowIndex stores the position where the integer overflows, and bitPaddingFlag stores whether the corresponding position needs to be filled.
  - gpuMemoryOverflowIndex stores the location where GPU memory overflows, and gpuBitPaddingFlag stores whether the corresponding location needs to be filled.
  - numBytes represents the maximum number of bytes that each thread block can process
  - mem_req represents the GPU memory limit per run

+ parallelHeader.h
  parallelHeader defines some common functions and variables, and introduces some common header files. It is expected that only the corresponding startup program can be implemented in CUDA internal programs and CUDAMPI programs.

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

+ Other files (serialFunctions.c serialHeader.h)
  It is also similar to the cuda part. Some CPU serial function parts are placed under include (since it is not the work of this project, I will not explain it in detail)

#### tests

The tests are for each sample (MPI divides the PROC once, and then divides the whole into equal parts based on different FILE_SIZE, and runs 3 times for each case)

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

The CUDA part is mainly CUDACompress.cu. The code mainly implements the process of Huffman compression of input files. In the code of this project, each function is separated into a single module.

1. **Read input file:**
    - The readInputFile function is used to read input data from a file
    - Open the file, get the file length, allocate memory, read the file content
2. **Calculate character frequency:**
    - calculateFrequency function counts the frequency of each character in the input data
    - Use the array frequency to record the number of occurrences of each character
3. **Initialize Huffman tree nodes:**
    - The initializeHuffmanTreeNodes function initializes the nodes of the Huffman tree
    - Create leaf nodes based on character frequency and record the number of leaf nodes
4. **Construct Huffman tree:**
    - The buildHuffmanTreeNodes function builds a Huffman tree by merging the two nodes with the lowest frequency
    - Loop to build the Huffman tree by calling the sortHuffmanTree and buildHuffmanTree functions
5. **Build Huffman coding table:**
    - The buildHuffmanDictionary function builds the Huffman coding table based on the Huffman tree
    - Generate the Huffman code of each character by recursively traversing the Huffman tree, and record the code length
    - Huffman encoding is stored in the global variable huffmanDictionary
6. **Calculation memory requirements:**
    - calculateMemoryRequirements function calculates the GPU memory required during compression
    - Calculate the memory requirements of the data offset array and check whether the GPU has enough memory to store the encoded data
    - Call lauchCUDAHuffmanCompress function to perform GPU compression
7. **Write to output file:**
    - The writeOutputFile function writes the compressed data to the output file
    - Write input file length, character frequency and compressed data
8. **Main function - `main`:**
    - Get input and output filenames from command line arguments
    - Read input files, calculate character frequencies, initialize Huffman tree nodes, build Huffman trees, and build Huffman coding tables
    - Calculate GPU memory requirements and perform GPU compression
    - Write output file and output program running time

### CUDAMPI

1. **MPI initialization:**
    - MPI_Init function initializes the MPI environment
    - Get the rank of the current process and the total number of processes
2. **Read input file:**
    - Use MPI file I/O to open the file and read the data blocks responsible for the local process
    - Use MPI_File_open, MPI_File_seek and MPI_File_read functions
3. **Statistical character frequency:**
    - Each process counts the frequency of characters in the data block it is responsible for
    - The frequency array records the number of occurrences of each character
4. **Construct Huffman tree:**
    - After each process counts the frequency independently, the reduction operation is performed through the MPI function, and the frequencies of each process are added to obtain the global frequency.
    - The main process (rank == 0) builds the Huffman tree and then broadcasts head_huffmanTreeNode to other processes
5. **Calculation memory requirements:**
    - cudaMemGetInfo function obtains GPU memory information
    - Calculate the memory requirements of the offset array and check whether the GPU has enough memory to store the encoded data
    - Use MPI_Bcast to broadcast memory requirement information to all processes
6. **Calling GPU compression Kernel:**
    - Each process calls the GPU compressed Kernel according to the data block it is responsible for
    - lauchCUDAHuffmanCompress function for GPU compression
7. **Calculate the length of compressed data of each process:**
    - The main process calculates the length of the compressed data of each process, and then collects these lengths to the main process through MPI_Gather
    - Update data length to reflect offset
8. **MPI File I/O: Write output file:**
    - The main process uses MPI file I/O to write the compressed data to the output file
    - MPI_File_open, MPI_File_seek and MPI_File_write functions are used to write files
9. **Exit the environment and terminate MPI:**
    - MPI_Finalize terminates the MPI environment and returns

## Performance evaluation

Since the user running memory on the commonly used test machine can only support the use of 768MB files, the current tests are all performance evaluations for (64,128,256,512,768)

### Serial performace

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

### CUDA performance

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

### CUDAMPI performance

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

It can be seen that the direct implementation of CUDA achieves an acceleration ratio of 4.53 times compared with the serial method. In the CUDAMPI version, due to multi-process IO processing, the performance of the IO and compressed Gather parts is significantly improved, and can It was found that using MPI built-in functions to achieve IO optimization, even single-core performance has been improved to a certain extent compared to CUDA (many of MPI's built-in functions are based on macros, and file seek and find are logarithmically optimized. of)

## Future Work

The current project employs MPI + OpenMP to implement certain operations serially. Due to its prominent multi-process characteristics and the desire to practice MPI + CUDA distributed training, a multi-card version has not been completed yet, nor has there been further optimization in a multi-threading context. However, the current implementation is likely to be more effective for future multi-machine support scenarios. The most practical current application remains for multicore + single-card general-purpose devices, also providing widespread compression acceleration support for users who primarily use Linux as their main operating system.

Future work may focus on the following directions:

- For the coding part, which is currently implemented serially in MPI, introducing OpenMP could significantly enhance the speed of the coding process. Additionally, for larger files, considering GPU support in the coding phase could be beneficial.
- There is still no complete support for multi-card devices. The original plan indeed aimed at distributed optimization for the current server-type single-machine dual-card architecture. There was consideration of binding MPI processes to a single GPU, but the actual bottleneck in compression/decompression performance is not in GPU computation. Improving IO through MPI multi-processes has shown better results than using two cards (originally, MPI + CUDA used two threads, but it was later found that improving IO was more effective). This discovery led to the realization that a multi-threaded approach might be preferable at present and that MPI + CUDA might need a more flexible configuration process for better multi-card support, which is currently a high workload for support and has not been successfully implemented.
- Attempts were made to use NVLink's GPU Direct to accelerate communication between GPUs, significantly improving multi-GPU compression performance. However, this is not as universal without NVLink, and this support was not maintained during later project debugging. There are plans to add this feature in the future.
- The deployment support for the entire project still feels inadequate. Currently, considering solutions like Docker and application packaging. The project uses Nvidia's complete MPI + CUDA toolkit, and HPCX is still essential for the project's MPI wrapper. Direct use of OpenMPI does not support the compilation and execution of the related code. Additionally, due to environmental issues, some MPI code is no longer runnable, and the influencing factors are still unknown. Pure MPI part of the code requires further testing.
- Support for compressed and decompressed file formats needs additional consideration. Currently, the project treats files as bytecode for input and restoration and does not support a general compression/decompression file format. In other words, currently, only two machines configured with this project can compress and transfer files to each other and decompress them using the project. To become more practical, additional support for related file types is needed.