#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vectorAddBlockSizeN(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x;  // Thread index within the block
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Add corresponding elements
    }
}

__global__ void vectorAddNThreads(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x;  // Thread index within the block
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Add corresponding elements
    }
}

int main(void) {
    int N = 5;  // You can change N to any size
    int size = N * sizeof(int);

    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    int *d_a, *d_b, *d_c;

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1;  // 1, 2, 3, 4, 5
        h_b[i] = (i + 1) * 2;  // 2, 4, 6, 8, 10
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Add vectors using Block Size N approach
    vectorAddBlockSizeN<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Result using Block Size N:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    // Reinitialize device memory
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Add vectors using N Threads approach
    vectorAddNThreads<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Result using N Threads:\n");
    for (int i = 0; i < N; i++) {
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
