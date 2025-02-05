#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>  // For sinf

__global__ void computeSine(float *input, float *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    if (idx < N) {
        output[idx] = sinf(input[idx]);  // Compute sine of the angle
    }
}

int main(void) {
    int N = 1000;  // Size of the array
    int size = N * sizeof(float);

    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *d_input, *d_output;

    // Initialize host input array with some angles in radians
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i * M_PI / 180.0);  // Angles in radians (1 degree step)
    }

    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block size and number of blocks
    int THREADS_PER_BLOCK = 256;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    computeSine<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the first 10 results for verification
    printf("First 10 Sine Values:\n");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("sin(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
