#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 5
#define TILE_SIZE 256

__constant__ float d_Mask[MASK_WIDTH];

__global__ void convolution1D(float *input, float *output, int width) {
    __shared__ float s_Data[TILE_SIZE + MASK_WIDTH - 1];
    
    int tx = threadIdx.x;
    int start = blockIdx.x * TILE_SIZE;
    int gx = start + tx;
    
    // Load input elements into shared memory
    if (gx < width) {
        s_Data[tx] = input[gx];
    } else {
        s_Data[tx] = 0.0f;
    }
    
    // Load halo elements
    if (tx < MASK_WIDTH - 1) {
        if (start + TILE_SIZE + tx < width)
            s_Data[TILE_SIZE + tx] = input[start + TILE_SIZE + tx];
        else
            s_Data[TILE_SIZE + tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute convolution
    float result = 0.0f;
    if (gx < width) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            result += s_Data[tx + i] * d_Mask[i];
        }
        output[gx] = result;
    }
}

int main() {
    const int width = 1024;
    float h_Input[width];
    float h_Output[width];
    float h_Mask[MASK_WIDTH] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    
    // Initialize input data
    for (int i = 0; i < width; i++) {
        h_Input[i] = static_cast<float>(i);
    }
    
    float *d_Input, *d_Output;
    cudaMalloc(&d_Input, width * sizeof(float));
    cudaMalloc(&d_Output, width * sizeof(float));
    
    cudaMemcpy(d_Input, h_Input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Mask, h_Mask, MASK_WIDTH * sizeof(float));
    
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE);
    
    convolution1D<<<gridSize, blockSize>>>(d_Input, d_Output, width);
    
    cudaMemcpy(h_Output, d_Output, width * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results (printing first 10 elements)
    printf("First 10 output elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_Output[i]);
    }
    printf("\n");
    
    cudaFree(d_Input);
    cudaFree(d_Output);
    
    return 0;
}
