#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mmk(float *a, float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        float s = 0.0f;
        for (int k = 0; k < n; k++)
            s += a[i * n + k] * b[k * n + j];
        c[i * n + j] = s;
    }
}

void mm(float *a, float *b, float *c, int n) {
    size_t sz = n * n * sizeof(float);
    float *p, *q, *r;
    cudaMalloc((void **)&p, sz);
    cudaMalloc((void **)&q, sz);
    cudaMalloc((void **)&r, sz);
    cudaMemcpy(p, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(q, b, sz, cudaMemcpyHostToDevice);

    dim3 t(16, 16);
    dim3 g((n + t.x - 1) / t.x, (n + t.y - 1) / t.y);

    mmk<<<g, t>>>(p, q, r, n);
    cudaMemcpy(c, r, sz, cudaMemcpyDeviceToHost);

    cudaFree(p);
    cudaFree(q);
    cudaFree(r);
}

int main() {
    int n;
    printf("n: ");
    scanf("%d", &n);
    size_t sz = n * n * sizeof(float);
    float *a = (float *)malloc(sz);
    float *b = (float *)malloc(sz);
    float *c = (float *)malloc(sz);

    printf("a:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            scanf("%f", &a[i * n + j]);
    }

    printf("b:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            scanf("%f", &b[i * n + j]);
    }

    mm(a, b, c, n);

    printf("c:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%.2f ", c[i * n + j]);
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
