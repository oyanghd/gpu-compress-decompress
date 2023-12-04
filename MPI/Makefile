CC = mpicc
SRC_DIR = .
BIN_DIR = ../bin

MPI_compress_SRC = $(SRC_DIR)/MPICompress.c $(SRC_DIR)/../include/serialFunctions.c
MPI_decompress_SRC = $(SRC_DIR)/MPIDecompress.c $(SRC_DIR)/../include/serialFunctions.c

all: MPI_compress MPI_decompress

MPI_compress: $(MPI_compress_SRC)
	$(CC) $^ -o $(BIN_DIR)/MPI_compress

MPI_decompress: $(MPI_decompress_SRC)
	$(CC) $^ -o $(BIN_DIR)/MPI_decompress

clean:
	rm -f $(BIN_DIR)/MPI_compress
	rm -f $(BIN_DIR)/MPI_decompress

.PHONY: all clean