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