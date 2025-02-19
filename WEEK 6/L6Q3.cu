#include <stdio.h>
#include <cuda_runtime.h>

__global__ void odd_phase_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 1 && idx < n - 1) {
        if (data[idx] > data[idx + 1]) {
            // Swap elements
            int temp = data[idx];
            data[idx] = data[idx + 1];
            data[idx + 1] = temp;
        }
    }
}

__global__ void even_phase_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0 && idx < n - 1) {
        if (data[idx] > data[idx + 1]) {
            // Swap elements
            int temp = data[idx];
            data[idx] = data[idx + 1];
            data[idx + 1] = temp;
        }
    }
}

void parallelOddEvenSort(int *h_arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int phase = 0; phase < n; ++phase) {
        // Odd phase
        odd_phase_kernel<<<numBlocks, blockSize>>>(d_arr, n);
        cudaDeviceSynchronize();

        // Even phase
        even_phase_kernel<<<numBlocks, blockSize>>>(d_arr, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    const int n = 8;
    int h_arr[n] = {64, 25, 12, 22, 11, 90, 45, 33};

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    parallelOddEvenSort(h_arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    return 0;
}
