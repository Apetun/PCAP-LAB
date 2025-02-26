#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

#define N 1024  

__global__ void countWordKernel(const char *sentence, const char *word, unsigned int *d_count, int sentence_len, int word_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx <= sentence_len - word_len) {
        bool match = true;
        
        for (int i = 0; i < word_len; ++i) {
            if (sentence[idx + i] != word[i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            atomicAdd(d_count, 1);
        }
    }
}

int main() {
    char sentence[N];
    char word[50];  
    char *d_sentence, *d_word;
    unsigned int *d_count;
    unsigned int *count = (unsigned int*)malloc(sizeof(unsigned int));
    *count = 0;  
    unsigned int *result;

    printf("Enter a sentence: ");
    fgets(sentence, N, stdin); 
    printf("Enter a Word: ");
    fgets(word, 50, stdin); 
   
    sentence[strcspn(sentence, "\n")] = 0;
    word[strcspn(word, "\n")] = 0;

    int sentence_len = strlen(sentence);
    int word_len = strlen(word);

    unsigned int *host_count = (unsigned int*)malloc(sizeof(unsigned int));
    *host_count = 0; 

    cudaMalloc((void**)&d_sentence, (sentence_len + 1) * sizeof(char));
    cudaMalloc((void**)&d_word, (word_len + 1) * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMalloc((void**)&result, sizeof(unsigned int));

    cudaMemcpy(d_sentence, sentence, (sentence_len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, (word_len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (sentence_len + blockSize - 1) / blockSize;
    countWordKernel<<<numBlocks, blockSize>>>(d_sentence, d_word, d_count, sentence_len, word_len);

    cudaDeviceSynchronize();

    cudaMemcpy(host_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Total occurrences of the word '%s': %u\n", word, *host_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    cudaFree(result);

    free(host_count);

    return 0;
}
