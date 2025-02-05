#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Define length of vectors
#define THREADS_PER_BLOCK 256

// CUDA kernel 1: Block size as N (1 block with N threads)
__global__ void addVectorsBlockSizeN(int* A, int* B, int* C) {
    int idx = threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel 2: N threads (N blocks with 1 thread each)
__global__ void addVectorsNThreads(int* A, int* B, int* C) {
    int idx = blockIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel 3: 256 threads per block (number of blocks varies)
__global__ void addVectors256ThreadsPerBlock(int* A, int* B, int* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int *A, *B, *C;       // Host vectors
    int *d_A, *d_B, *d_C; // Device vectors

    size_t size = N * sizeof(int);

    // Allocate memory on host
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Kernel 1: Launch with block size as N (1 block with N threads)
    addVectorsBlockSizeN<<<1, N>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result (Block Size N):\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Kernel 2: Launch with N threads (N blocks with 1 thread each)
    addVectorsNThreads<<<N, 1>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResult (N Threads):\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Kernel 3: Launch with 256 threads per block (vary number of blocks)
    

    // Define the grid and block dimensions using dim3
    dim3 blockDim(THREADS_PER_BLOCK,1,1);  // Block of 256 threads
    dim3 gridDim(ceil(N/256),1,1);  // Variable number of blocks

    addVectors256ThreadsPerBlock<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResult (256 Threads Per Block):\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Clean up
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
