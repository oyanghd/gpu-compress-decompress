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