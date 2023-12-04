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