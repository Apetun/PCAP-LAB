#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for Inclusive Scan
__global__ void inclusiveScanKernel(float *d_input, float *d_output, int n) {
    __shared__ float temp[1024]; // Shared memory for block-level scan
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    int index = blockIdx.x * blockDim.x + tid;
    if (index < n) {
        temp[tid] = d_input[index];
    } else {
        temp[tid] = 0.0f; // Handle out-of-bound threads
    }
    __syncthreads();

    // Perform the scan operation in shared memory
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Clear the last element and propagate down the tree
    if (tid == 0) {
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();

    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (index < n) {
        d_output[index] = temp[tid];
    }
}

void inclusiveScan(float *h_input, float *h_output, int n) {
    float *d_input, *d_output;

    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    inclusiveScanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int n;

    // Input array size
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    float *h_input = (float *)malloc(n * sizeof(float));
    float *h_output = (float *)malloc(n * sizeof(float));

    // Input array elements
    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &h_input[i]);
    }

    // Perform inclusive scan
    inclusiveScan(h_input, h_output, n);

    // Print results
    printf("Inclusive Scan Result:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    free(h_input);
    free(h_output);

    return 0;
}
