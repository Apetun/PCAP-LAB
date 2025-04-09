#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 5

// Constant memory for convolution mask
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution_1D_tiled_kernel(float *N, float *P, int Mask_Width, int Width) {
    // Shared memory declaration with halo elements
    extern __shared__ float N_ds[];
    
    // Calculate thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate radius of the mask
    int n = Mask_Width / 2;
    
    // Load main tile elements
    if (i < Width) {
        N_ds[threadIdx.x + n] = N[i];
    } else {
        N_ds[threadIdx.x + n] = 0.0f;
    }

    // Load left halo elements
    if (threadIdx.x >= blockDim.x - n) {
        int halo_index_left = i - blockDim.x;
        N_ds[threadIdx.x - (blockDim.x - n)] = 
            (halo_index_left >= 0) ? N[halo_index_left] : 0.0f;
    }

    // Load right halo elements
    if (threadIdx.x < n) {
        int halo_index_right = i + blockDim.x;
        N_ds[threadIdx.x + blockDim.x + n] = 
            (halo_index_right < Width) ? N[halo_index_right] : 0.0f;
    }

    __syncthreads();

    // Calculate convolution
    if (i < Width) {
        float Pvalue = 0.0f;
        for (int j = 0; j < Mask_Width; j++) {
            Pvalue += N_ds[threadIdx.x + j] * M[j];
        }
        P[i] = Pvalue;
    }
}

void convolution_1D(float *h_N, float *h_P, float *h_M, 
                    int width, int mask_width) {
    float *d_N, *d_P;
    
    // Allocate device memory
    cudaMalloc(&d_N, width * sizeof(float));
    cudaMalloc(&d_P, width * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy mask to constant memory
    cudaMemcpyToSymbol(M, h_M, mask_width * sizeof(float));
    
    // Calculate grid dimensions
    int numBlocks = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    // Launch kernel with shared memory allocation
    convolution_1D_tiled_kernel<<<numBlocks, TILE_SIZE, 
        (TILE_SIZE + mask_width - 1) * sizeof(float)>>>(d_N, d_P, mask_width, width);
    
    // Copy result back to host
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_N);
    cudaFree(d_P);
}

int main() {
    const int width = 1<<20;  // 1M elements
    const int mask_width = 5;
    
    // Host allocations
    float *h_N = (float*)malloc(width * sizeof(float));
    float *h_P = (float*)malloc(width * sizeof(float));
    float h_M[mask_width] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    
    // Initialize input
    for (int i = 0; i < width; i++) {
        h_N[i] = (float)(i % 10);
    }
    
    // Perform convolution
    convolution_1D(h_N, h_P, h_M, width, mask_width);
    
    // Print first 10 results
    printf("First 10 output elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("P[%d] = %.2f\n", i, h_P[i]);
    }
    
    free(h_N);
    free(h_P);
    return 0;
}
