//code to test the centroid kernel
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#define NUM_POINTS 2048

__global__
void Centroid(float* cloud, float* bar)
{
    __shared__ float sdatax[NUM_POINTS];
    __shared__ float sdatay[NUM_POINTS];
    __shared__ float sdataz[NUM_POINTS];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdatax[tid] = cloud[i * 3 + 0];
    sdatay[tid] = cloud[i * 3 + 1];
    sdataz[tid] = cloud[i * 3 + 2];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0) 
        {
            sdatax[tid] += sdatax[tid + s];
            sdatay[tid] += sdatay[tid + s];
            sdataz[tid] += sdataz[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        bar[0 + 3 * blockIdx.x] = sdatax[tid];
        bar[1 + 3 * blockIdx.x] = sdatay[tid];
        bar[2 + 3 * blockIdx.x] = sdataz[tid];
        //printf("bar[2]: %.3f\n", bar[2 + 3 * blockIdx.x]);
    }
}

int main(void)
{
    
    float* h_cloud = NULL, * d_cloud = NULL;
    float* h_bar = NULL, * d_bar = NULL;
    float* h_bar_2 = NULL, * d_bar_2 = NULL;

    int GridSize = 8;
    int BlockSize = NUM_POINTS / GridSize;

    size_t size_cloud = 3 * NUM_POINTS * sizeof(float);
    //size_t size_bar = 3 * GridSize * sizeof(float);

    h_cloud = (float*)malloc(size_cloud);
    cudaMalloc(&d_cloud, size_cloud);

    h_bar = (float*)malloc(size_cloud);
    cudaMalloc(&d_bar, size_cloud);

    for (unsigned int i = 0; i < NUM_POINTS; i++)
    {
        h_cloud[0 + i * 3] = 1.0f;//all the x values get 1
        h_cloud[1 + i * 3] = 2.0f;//all the y values get 2
        h_cloud[2 + i * 3] = 3.0f;//all the z values get 3
    }

    cudaMemcpy(d_cloud, h_cloud, size_cloud, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Centroid <<< GridSize, BlockSize >>> (d_cloud, d_bar);
    Centroid << < 1, BlockSize >> > (d_bar, d_bar);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error in kernel: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float miliseconds1 = 0;
    cudaEventElapsedTime(&miliseconds1, start, stop);

    cudaMemcpy(h_bar, d_bar, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("bar 1:\n");
    for (int i = 0; i < 3; i++) printf("%.3f ", h_bar[i]);
    printf("\n");

    //cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* d_x = NULL;
    cudaMalloc(&d_x, 3 * sizeof(float));
    cudaMemset(d_x, 1, 3 * sizeof(float));
    h_bar_2 = (float*)malloc(3 * sizeof(float));
    cudaMalloc(&d_bar_2, 3 * sizeof(float));

    cudaEventRecord(start);
    cublasSgemv(handle, CUBLAS_OP_N, 3, NUM_POINTS, &alpha, d_cloud, 3, d_x, 1, &beta, d_bar_2, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float miliseconds2 = 0;
    cudaEventElapsedTime(&miliseconds2, start, stop);

    cudaMemcpy(h_bar_2, d_bar_2, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("bar 2:\n");
    for (int i = 0; i < 3; i++) printf("%.3f ", h_bar_2[i]);
    printf("\n");

    printf("\nElapsed time 1: %f ms\n", miliseconds1);
    printf("\nElapsed time 2: %f ms\n", miliseconds2);

    cublasDestroy(handle);
    cudaFree(d_cloud), cudaFree(d_bar), cudaFree(d_bar_2);
    free(h_cloud), free(h_bar), free(h_bar_2);
    cudaFree(d_x);

    return 1;
}
