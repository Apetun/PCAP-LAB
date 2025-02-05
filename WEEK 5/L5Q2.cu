#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Add corresponding elements
    }
}

int main(void) {
    int N = 1000;  // You can change N to any size
    int size = N * sizeof(int);

    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    int *d_a, *d_b, *d_c;

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1;  // 1, 2, 3, 4, ...
        h_b[i] = (i + 1) * 2;  // 2, 4, 6, 8, ...
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel with dynamic number of blocks
    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result (optional, printing first 10 elements)
    printf("Result: ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
