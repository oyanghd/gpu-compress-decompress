# gpu-compress-decompress
Graphics card often idling? Is the decompression speed of common tools too slow? This project is a GPU + multi-process, multi-thread compression and decompression solution for Linux users.

[English](README.md) | [中文](README_ZH.md)

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

```
shellCopy codeHPCX
CUDA
Infiniband Mellanox OFED 5.x
```

### Currently tested successful dependency versions:

#### The following environment is used as the standard for later tests

Software environment:

```
Copy codeCUDA 11.8
GPU Driver version 520.61.05
MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu20.04-x86_64
HPCX 2.13.0
```

Hardware environment:

```
scssCopy codeCPU Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
GPU 2 x P100-16G
IB Connect-X 3
```

### Installation Method of the Project

Since the project requires testing files with direct and random write operations, and to test the performance with large files, these randomly generated files are quite large in size. It's impractical to include them in the project directory for uploading. Moreover, uploading and downloading these files is not as fast as directly generating a disk read/write file. Therefore, a script has been written to generate test files and directories by running the script:

```
shellCopy codemkdir bin
mkdir logs
mkdir Testfiles
cd Testfiles
python3 ../scripts/generateFiles.py
cd ..
make
```

(Before running, it's necessary to load the HPCX module and add the CUDA toolkit to the environment variables.)

Then, to run the project, you can use the following script in the project folder directory. This script itself is also based on the Gaussian16 Job Batch Processing Management System (PBS) and can be directly submitted to PBS for execution:

```
./scripts/CUDAMPI.pbs
./scripts/CUDA.pbs
./scripts/MPI.pbs
./scripts/Serial.pbs
```

### Serial Performance

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

### CUDA Performance

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

### CUDAMPI Performance

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
