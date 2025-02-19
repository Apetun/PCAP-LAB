#include <stdio.h>
#include <cuda_runtime.h>

__device__ void selection_sort(int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        int min_val = data[i];
        int min_idx = i;

        for (int j = i + 1; j <= right; ++j) {
            int val_j = data[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void selection_sort_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        selection_sort(data, idx, n - 1);
    }
}

void parallelSelectionSort(int *h_arr, int n) {
    int *d_arr;

    cudaMalloc((void**)&d_arr, n * sizeof(int));

    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1; 
    int numBlocks = n;
    selection_sort_kernel<<<numBlocks, blockSize>>>(d_arr, n);
    cudaDeviceSynchronize();

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

    parallelSelectionSort(h_arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    return 0;
}
