#include "../include/parallelHeader.h"

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;
unsigned char bitSequenceConstMemory[256][255];

void readInputFile(const char *filename, unsigned char **inputFileData, unsigned int *inputFileLength) {
    FILE *inputFile = fopen(filename, "rb");
    if (!inputFile) {
        printf("Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(inputFile, 0, SEEK_END);
    *inputFileLength = ftell(inputFile);
    fseek(inputFile, 0, SEEK_SET);

    *inputFileData = (unsigned char *)malloc(*inputFileLength * sizeof(unsigned char));
    if (!(*inputFileData)) {
        printf("Memory allocation error\n");
        fclose(inputFile);
        exit(EXIT_FAILURE);
    }

    fread(*inputFileData, sizeof(unsigned char), *inputFileLength, inputFile);
    fclose(inputFile);
}

void calculateFrequency(const unsigned char *inputFileData, unsigned int inputFileLength, unsigned int *frequency) {
    for (unsigned int i = 0; i < 256; i++) {
        frequency[i] = 0;
    }

    for (unsigned int i = 0; i < inputFileLength; i++) {
        frequency[inputFileData[i]]++;
    }
}

void initializeHuffmanTreeNodes(unsigned int *frequency, unsigned int *distinctCharacterCount) {
    *distinctCharacterCount = 0;

    for (unsigned int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            huffmanTreeNode[*distinctCharacterCount].count = frequency[i];
            huffmanTreeNode[*distinctCharacterCount].letter = i;
            huffmanTreeNode[*distinctCharacterCount].left = NULL;
            huffmanTreeNode[*distinctCharacterCount].right = NULL;
            (*distinctCharacterCount)++;
        }
    }
}

void buildHuffmanTreeNodes(unsigned int distinctCharacterCount) {
    for (unsigned int i = 0; i < distinctCharacterCount - 1; i++) {
        unsigned int combinedHuffmanNodes = 2 * i;
        sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
        buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
    }

    if (distinctCharacterCount == 1) {
        head_huffmanTreeNode = &huffmanTreeNode[0];
    }
}

void calculateMemoryRequirements(unsigned char *inputFileData, unsigned int inputFileLength, const unsigned int *frequency) {
    unsigned long int mem_free, mem_total, mem_req, mem_offset, mem_data;
    int numKernelRuns;
    unsigned int integerOverflowFlag;

    cudaMemGetInfo(&mem_free, &mem_total);

    // debug
    // if (1) {
    //     printf("Free Mem: %lu\n", mem_free);
    // }

    // offset array requirements
    mem_offset = 0;
    for (unsigned int i = 0; i < 256; i++) {
        mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
    }
    mem_offset = (mem_offset % 8 == 0) ? mem_offset : mem_offset + 8 - mem_offset % 8;

    // other memory requirements
    mem_data = inputFileLength + (inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);

    if (mem_free - mem_data < MIN_SCRATCH_SIZE) {
        printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
        exit(EXIT_FAILURE);
    }

    mem_req = mem_free - mem_data - 10 * 1024 * 1024;
    numKernelRuns = ceil((double)mem_offset / mem_req);
    integerOverflowFlag = (mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX) ? 0 : 1;

    // debug
    // if (1) {
    //     printf("	InputFileSize      =%u\n\
	// OutputSize         =%lu\n\
	// NumberOfKernel     =%d\n\
	// integerOverflowFlag=%d\n", inputFileLength, mem_offset / 8, numKernelRuns, integerOverflowFlag);
    // }

    // generate data offset array
    unsigned int *compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));

    // launch kernel
    lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, inputFileLength, numKernelRuns, integerOverflowFlag, mem_req);

    free(compressedDataOffset);
}

void writeOutputFile(const char *filename, unsigned int inputFileLength, const unsigned int *frequency, const unsigned char *inputFileData) {
    FILE *compressedFile = fopen(filename, "wb");
    if (!compressedFile) {
        printf("Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
    fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
    fwrite(inputFileData, sizeof(unsigned char), inputFileLength / 8, compressedFile);

    fclose(compressedFile);
}

int main(int argc, char **argv) {
    unsigned int distinctCharacterCount;
    unsigned char bitSequenceLength = 0, bitSequence[255];
    unsigned int frequency[256];
    unsigned char *inputFileData;
    unsigned int inputFileLength;
    clock_t start, end;

    // check number of args
    if (argc != 3) {
        printf("try with arguments InputFile and OutputFile");
        return -1;
    }

    readInputFile(argv[1], &inputFileData, &inputFileLength);

    // calculate run duration
    start = clock();

    // find the frequency of each symbols
    calculateFrequency(inputFileData, inputFileLength, frequency);

    // initialize nodes of huffman tree
    initializeHuffmanTreeNodes(frequency, &distinctCharacterCount);

    // build tree
    buildHuffmanTreeNodes(distinctCharacterCount);

    // build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

    // calculate memory requirements
    calculateMemoryRequirements(inputFileData, inputFileLength, frequency);

    // calculate run duration
    end = clock();

    // write src inputFileLength, header and compressed data to output file
    writeOutputFile(argv[2], inputFileLength, frequency, inputFileData);

    unsigned int cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);

    free(inputFileData);
    return 0;
}