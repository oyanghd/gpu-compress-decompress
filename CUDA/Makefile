NVCC = nvcc
SRC_DIR = .
INCLUDE_DIR = ../include
BIN_DIR = ../bin

CU_FILES = $(SRC_DIR)/CUDACompress.cu $(INCLUDE_DIR)/parallelFunctions.cu $(INCLUDE_DIR)/kernel.cu $(INCLUDE_DIR)/GPUWrapper.cu

all: CUDA_compress

CUDA_compress: $(CU_FILES)
	$(NVCC) -dc $^
	$(NVCC) *.o -o $(BIN_DIR)/CUDA_compress
	rm -rf *.o

clean:
	rm -f $(BIN_DIR)/CUDA_compress

.PHONY: all clean