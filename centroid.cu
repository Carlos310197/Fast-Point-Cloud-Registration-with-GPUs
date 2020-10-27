//code to test the centroid kernel
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define NUM_POINTS 1<<5

__global__
void Centroid(double* cloud, int n, double* bar)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    //if(i==0) printf("%f\n",log2(n))
	for(int s=1;s<=n;s*=2)
	{
		if(i%(2*s)==0)
		{
			cloud[i*3 + 0] += cloud[(i+s)*3 + 0];
			cloud[i*3 + 1] += cloud[(i+s)*3 + 1];
			cloud[i*3 + 2] += cloud[(i+s)*3 + 2];
		}
	}

	if(i==0) bar[0] = cloud[0];
	if(i==1) bar[1] = cloud[1];
	if(i==2) bar[2] = cloud[2];
	
}

int main(void)
{
    unsigned long int NUM_POINTS = 1<<25;
    double* h_cloud, *d_cloud;
    double* h_bar, *d_bar;
    size_t size_cloud = 3*NUM_POINTS*sizeof(double);
    size_t size_bar = 3*sizeof(double);

    h_cloud = (double*)malloc(size_cloud);
    cudaMalloc(&d_cloud,size_cloud);

    h_bar = (double*)malloc(size_bar);
    cudaMalloc(&d_bar,size_bar);

    for(int i=0;i<NUM_POINTS;i++) h_cloud[0+i*3] = 1;
    for(int i=0;i<NUM_POINTS;i++) h_cloud[1+i*3] = 2;
    for(int i=0;i<NUM_POINTS;i++) h_cloud[2+i*3] = 3;

    cudaMemcpy(d_cloud,h_cloud,size_cloud,cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int factor = sqrt(NUM_POINTS);
    Centroid<<<factor,NUM_POINTS/factor>>>(d_cloud,NUM_POINTS,d_bar);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_bar,d_bar,size_bar,cudaMemcpyDeviceToHost);

    for(int i=0;i<3;i++) printf("%.3f ",h_bar[i]);
    printf("\n");
    printf("Elapsed time: %f ms\n",milliseconds);

    cudaFree(d_cloud);cudaFree(d_bar);
    free(h_cloud);free(h_bar);

    return 1;
}