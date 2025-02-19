#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}

int main() {
    // Define the size of the input array and the mask
    int Width = 10; // Size of the input array N
    int Mask_Width = 3; // Size of the mask array M

    // Host arrays
    float *h_N = (float*)malloc(Width * sizeof(float));  // Input array
    float *h_M = (float*)malloc(Mask_Width * sizeof(float)); // Mask array
    float *h_P = (float*)malloc(Width * sizeof(float));  // Output array

    // Initialize the input array N and mask M with some values
    for (int i = 0; i < Width; i++) {
        h_N[i] = i + 1;  // Example input values: 1, 2, 3, ..., Width
    }

    for (int i = 0; i < Mask_Width; i++) {
        h_M[i] = 1.0f;  // Example mask values: all ones
    }

    // Device arrays
    float *d_N, *d_M, *d_P;

    // Allocate memory on the device
    cudaMalloc((void**)&d_N, Width * sizeof(float));
    cudaMalloc((void**)&d_M, Mask_Width * sizeof(float));
    cudaMalloc((void**)&d_P, Width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (Width + blockSize - 1) / blockSize; // Calculate the number of blocks

    convolution_1D_basic_kernel<<<numBlocks, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result from device to host
    cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Output array P:\n");
    for (int i = 0; i < Width; i++) {
        printf("%.2f ", h_P[i]);
    }
    printf("\n");

    // Free the allocated memory
    free(h_N);
    free(h_M);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
