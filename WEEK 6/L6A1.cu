#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to convert integers to octal values
__global__ void convert_to_octal(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int num = input[idx];
        int octal = 0, place = 1;

        // Convert to octal using modulo and division
        while (num > 0) {
            octal += (num % 8) * place;
            num /= 8;
            place *= 10;
        }

        output[idx] = octal;
    }
}

void convertIntegersToOctal(int *h_input, int *h_output, int n) {
    int *d_input, *d_output;

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block size and number of blocks
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the kernel to convert integers to octal
    convert_to_octal<<<numBlocks, blockSize>>>(d_input, d_output, n);

    // Synchronize to ensure kernel execution is completed
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int h_input[] = {64, 25, 12, 22, 11, 90, 45, 33}; // Example input
    int n = sizeof(h_input) / sizeof(h_input[0]);    // Number of elements
    int h_output[n];

    printf("Original integers: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Convert integers to octal using CUDA
    convertIntegersToOctal(h_input, h_output, n);

    printf("Converted octal values: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    return 0;
}
