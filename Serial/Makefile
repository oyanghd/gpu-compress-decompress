CC = gcc
SRC_DIR = .
BIN_DIR = ../bin

compress_SRC = $(SRC_DIR)/compress.c $(SRC_DIR)/../include/serialFunctions.c
decompress_SRC = $(SRC_DIR)/decompress.c $(SRC_DIR)/../include/serialFunctions.c

all: compress decompress

compress: $(compress_SRC)
	$(CC) $^ -o $(BIN_DIR)/compress

decompress: $(decompress_SRC)
	$(CC) $^ -o $(BIN_DIR)/decompress

clean:
	rm -f $(BIN_DIR)/compress
	rm -f $(BIN_DIR)/decompress

.PHONY: all clean