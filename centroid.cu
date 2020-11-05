//code to test the centroid kernel
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void Centroid(float* cloud, int n, float* bar)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for(int s = 1; s <= n; s *= 2)
	{
		if(i % (2 * s) == 0)
		{
			cloud[i*3 + 0] += cloud[(i+s)*3 + 0];
			cloud[i*3 + 1] += cloud[(i+s)*3 + 1];
			cloud[i*3 + 2] += cloud[(i+s)*3 + 2];
        }
        __syncthreads();
	}

	if(i==0) bar[0] = cloud[0];
	if(i==1) bar[1] = cloud[1];
	if(i==2) bar[2] = cloud[2];
	
}

int main(void)
{
    int NUM_POINTS = 1<<9;//512 points
    float* h_cloud, *d_cloud;
    float* h_bar, *d_bar;

    size_t size_cloud = 3 * NUM_POINTS * sizeof(float);
    size_t size_bar = 3 * sizeof(float);

    h_cloud = (float*)malloc(size_cloud);
    cudaMalloc(&d_cloud,size_cloud);
    cudaMemset(d_cloud, 0, size_cloud);

    h_bar = (float*)malloc(size_bar);
    cudaMalloc(&d_bar,size_bar);

    for(int i=0;i<NUM_POINTS;i++)
    {
        h_cloud[0 + i * 3] = 1;//all the x values get 1
        h_cloud[1 + i * 3] = 2;//all the y values get 2
        h_cloud[2 + i * 3] = 3;//all the z values get 3
    }

    cudaMemcpy(d_cloud,h_cloud,3 * NUM_POINTS * sizeof(float),cudaMemcpyHostToDevice);
	
    int GridSize = 8;
    int BlockSize = NUM_POINTS / GridSize;
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	
    Centroid<<<GridSize,BlockSize>>>(d_cloud,NUM_POINTS,d_bar);
	
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error in kernel: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_bar,d_bar,size_bar,cudaMemcpyDeviceToHost);

    printf("bar:\n");
    for(int i=0;i<3;i++) printf("%.3f ",h_bar[i]);
    printf("\n");
    printf("Elapsed time: %f ms\n",milliseconds);

    cudaFree(d_cloud);cudaFree(d_bar);
    free(h_cloud);free(h_bar);

    return 1;
}
