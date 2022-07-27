//For debugging:
//nvcc ICP_standard.cu -lcublas -lcurand -lcusolver -L/home/carlos/Desktop/ICP/my_libraries/ -o ICP_cuda

//ICP standard implementation in CUDA
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
#include "my_lib.h"

//constants
#define WIDTH 12
#define NUM_POINTS WIDTH*WIDTH //width of grid
#define XY_min -2.0
#define XY_max 2.0
#define MAX_ITER 30

__global__
void cudaMatching(double* data, double* model, double* distance, int* pos)
{
	//distances between x, y, z in position "pos" from data AND all the x, y, z from model 
	double x, y, z;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	x = data[0 + pos[0] * 3] - model[0 + i * 3];
	y = data[1 + pos[0] * 3] - model[1 + i * 3];
	z = data[2 + pos[0] * 3] - model[2 + i * 3];
	distance[i] = pow(x,2) + pow(y,2) + pow(z,2);
}

__global__
void Index_model(double* cloud, int* idx, double* result)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index = idx[i]-1;//the minimum fcn was 1-based index
	result[0 + i * 3] = cloud[0 + index * 3];
	result[1 + i * 3] = cloud[1 + index * 3];
	result[2 + i * 3] = cloud[2 + index * 3];
}

__global__
void centroid(double* cloud, int n, double* bar)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d ",i);
	for(int s=1;s<=n;s*=2)//parallel reduction
	{
		if(i%(2*s)==0)
		{
			cloud[i*3 + 0] += cloud[(i+s)*3 + 0];
			cloud[i*3 + 1] += cloud[(i+s)*3 + 1];
			cloud[i*3 + 2] += cloud[(i+s)*3 + 2];
		}
	}
	if(i>=0 && i<=2) bar[i] = cloud[i]/n;

	/*_syncthreads();
	dev[0 + i * 3] = cloud[0 + i * 3] - bar[0];
	dev[1 + i * 3] = cloud[1 + i * 3] - bar[1];
	dev[2 + i * 3] = cloud[2 + i * 3] - bar[2];*/
}

__global__
void deviation(double* cloud, double* bar, double* dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	dev[0 + i * 3] = cloud[0 + i * 3] - bar[0];
	dev[1 + i * 3] = cloud[1 + i * 3] - bar[1];
	dev[2 + i * 3] = cloud[2 + i * 3] - bar[2];
}

__global__
void repmat(double* vector, double* matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	matrix[0 + i * 3] = vector[0];
	matrix[1 + i * 3] = vector[1];
	matrix[2 + i * 3] = vector[2];
}

__global__
void Error(double* aux, double* D, double* M, double* error, int* iteration, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	aux[i] = pow(M[i]-D[i],2);
	for(int s=1;s<=3*n;s*=2)//parallel reduction
	{
		if(i%(2*s)==0)
		{
			aux[i] += aux[i + s];
		}
	}
	if(i==0) error[iteration[0]] = sqrt(aux[i]/n);
}

int main(void)
{
    ////////////////////////////////////1st:Creation of dataset////////////////////////////

	int num_points, i, j, k;
	double ti[3], ri[3];
	double lin_space[WIDTH], lenght;
	double* mesh_x, * mesh_y, * z;
	double* h_D, * h_M;

	num_points = WIDTH * WIDTH;//number of points
	lenght = XY_max - XY_min;
	
	size_t bytesD = num_points * 3 * sizeof(double);
	size_t bytesM = num_points * 3 * sizeof(double);

    //create an array with all points equally separated
    int n = WIDTH;
	for (i = 0; i < WIDTH; i++)
	{
		lin_space[i] = XY_min + double(i) * (lenght) / (double(n) - 1.0f);
	}

	//create meshgrid for x and y coordinates
	mesh_x = (double*)malloc(num_points * sizeof(double));
	mesh_y = (double*)malloc(num_points * sizeof(double));
	i = 0;
	k = 0;
	while (i < num_points)
	{
		j = 0;
		while (j < WIDTH)
		{
			mesh_y[i] = lin_space[j];
			mesh_x[i] = lin_space[k];
			i++; j++;
		}
		k++;
	}

	//Create the function z = f(x,y) = x^2-y^2
	z = (double*)malloc(num_points * sizeof(double));
	for (i = 0; i < num_points; i++) z[i] = pow(mesh_x[i], 2) - pow(mesh_y[i], 2);
	
	//Create data point cloud matrix (source point cloud)
	//points are stored in this order: x1y1z1, x2y2z2, x3y3z3, ....
	h_D = (double*)malloc(bytesD);
	k = 0;
	for(i = 0; i < num_points; i++)
	{
		for(j = 0; j < 3; j++)
		{
			if(j == 0) h_D[k] = mesh_x[i];
			if(j == 1) h_D[k] = mesh_y[i];
			if(j == 2) h_D[k] = z[i];
			k++;
		}
	}
	
	/*printf("Data point cloud\n");
	print_cloud(h_D, num_points, num_points);*/

	//Create model point cloud matrix (target point cloud)
	//every matrix is defined using the colum-major order
	h_M = (double*)malloc(bytesM);

	//Translation values
	ti[0] = 1.0f;//x
	ti[1] = -0.3f;//y
	ti[2] = 0.2f;//z

	//Rotation values (rad)
	ri[0] = 1.0f;//axis x
	ri[1] = -0.5f;//axis y
	ri[2] = 0.05f;//axis z

	double h_rx[9] = {
		1.0f, 0.0f, 0.0f,
		0.0f, cos(ri[0]), sin(ri[0]),
		0.0f, -sin(ri[0]), cos(ri[0])};
	double h_ry[9] = {
		cos(ri[1]), 0.0f, -sin(ri[1]),
		0.0f, 1, 0.0f,
		sin(ri[1]), 0.0f, cos(ri[1])};
	double h_rz[9] = {
		cos(ri[2]), sin(ri[2]),0.0f,
		-sin(ri[2]), cos(ri[2]), 0.0f,
		0.0f, 0.0f, 1.0f};

	//calculate the rotation matrix h_r
	double h_r[9]={0.8765,-0.3759,0.3008,-0.0439,0.5598,0.8275,-0.4794,-0.7385,0.4742};
	//dmatrixMul(h_rx, h_ry, h_r, 3, 3, 3);
	//dmatrixMul(h_r, h_rz, h_r, 3, 3, 3);
	
	/*printf("Rx:\n");
	print_cloud(h_rx,3,3);
	printf("Ry:\n");
	print_cloud(h_ry,3,3);
	printf("Rz:\n");
	print_cloud(h_rz,3,3);

	printf("Rotation Matrix:\n");
	print_cloud(h_r,3,3);*/

	//h_M = h_r*h_D
	dmatrixMul(h_r, h_D, h_M, 3, num_points, 3);
	//h_M = h_M + t
	for(i = 0; i < num_points; i++)
	{
		for(j = 0; j < 3; j++)
		{
			h_M[j + i * 3] += ti[j];
		}
	}

	/*printf("\nModel point cloud\n");
	print_cloud(h_M, num_points, num_points);*/

	free(mesh_x); free(mesh_y); free(z);
    /////////////////////////////////////////////End of 1st//////////////////////////////////////

	//////////////////////////////////////////2nd:ICP Algorithm//////////////////////////////////
	
	int D_size = num_points;//data cloud number of points
	int M_size = num_points;//model cloud number of points
	double* d_M = NULL;//model cloud (device)
	double* d_Dt = NULL;//data cloud transformed (device)
	double* h_Dt = NULL;//data cloud transformed (host)
	int* d_idx = NULL;//index vector (used for correspondence)
	int* h_idx = NULL;//index vector (used for correspondence)
	double* distance = NULL;//array to store temporary distances
	double* d_M_idx = NULL;//model cloud indexed (device)
	double* d_barD = NULL,* d_barM = NULL;//centroids (device)
	double* d_devD = NULL,* d_devM = NULL;//deviations (device)
	double* d_W = NULL;//3x3 matrix used in SVD
	double *d_S = NULL;
    double *d_U = NULL;
    double *d_VT = NULL;
	double *d_work = NULL;
	double *d_rwork = NULL;
	int *devInfo = NULL;
	int lwork = 0;
	double alpha, beta;
	double* d_temp_r = NULL;
	double* d_temp_T = NULL;
	double* rep_T = NULL;
	int *d_i = NULL;
	int *h_i = NULL;
	double* d_error = NULL;
	double* h_error = NULL;
	int* d_iteration = NULL;
	double* d_aux = NULL;//auxiliar matrices to compute some operations
	
	h_idx = (int *)malloc(D_size * sizeof(int));
	h_i = (int *)malloc(D_size * sizeof(int));
	for(i = 0; i < D_size; i++)h_i[i] = i;
	cudaMemcpy(d_i, h_i, D_size * sizeof(int), cudaMemcpyHostToDevice);

	//cuBLAS handle
    cublasHandle_t handle;
	cublasCreate(&handle);

	cudaMalloc(&d_Dt,bytesD);
	cudaMalloc(&d_M,bytesM);
	cudaMalloc(&d_idx, D_size*sizeof(int));
	cudaMalloc(&distance, M_size*sizeof(double));
	cudaMalloc(&d_M_idx,bytesD);
	h_Dt = (double*)malloc(bytesD);

	cudaMalloc(&d_barD, 3 * sizeof(double));
	cudaMalloc(&d_barM, 3 * sizeof(double));
	cudaMalloc(&d_devD, bytesD);
	cudaMalloc(&d_devM, bytesD);

	cudaMalloc(&d_S  , sizeof(double) * 3);
    cudaMalloc(&d_U  , sizeof(double) * 9);
    cudaMalloc(&d_VT , sizeof(double) * 9);
    cudaMalloc(&devInfo, sizeof(int));
	cudaMalloc(&d_W  , sizeof(double) * 9);
	cusolverDnHandle_t cusolverH;//cuSolver handle
	cusolverDnCreate(&cusolverH);
	cusolverDnDgesvd_bufferSize(cusolverH, 3, 3, &lwork);
	cudaMalloc((void**)&d_work , sizeof(double)*lwork);

	cudaMalloc(&d_temp_r,sizeof(double) * 9);
	cudaMalloc(&d_temp_T,sizeof(double) * 3);
	cudaMalloc(&rep_T,sizeof(double) * 3 * D_size);

	cudaMalloc(&d_i,sizeof(int));

	cudaMalloc(&d_iteration,sizeof(int));

	cudaMalloc(&d_error,MAX_ITER*sizeof(double));
	h_error = (double*)malloc(MAX_ITER*sizeof(double));

	cudaMalloc(&d_aux,bytesD);

	cudaMemcpy(d_Dt,h_D,bytesD,cudaMemcpyHostToDevice);
	cudaMemcpy(d_M,h_M,bytesM,cudaMemcpyHostToDevice);

	free(h_D);free(h_M);

	//printf("Here starts the main loop!\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int iterations = 0;
	cudaEventRecord(start);
	//MAIN LOOP
	while(1)
	{
		//Matching step (brute force)
		cudaMemset(&d_idx, 0, D_size*sizeof(int));
		for(i = 0; i < D_size; i++)
		{
			cudaMemcpy(d_i,&i,sizeof(int),cudaMemcpyHostToDevice);
			cudaMatching<<<1, M_size>>>(d_Dt, d_M, distance, d_i);//find another way to do this!!!!!!!
			cudaDeviceSynchronize();
			//checkCudaErrors(cudaGetLastError());
			cublasIdamin(handle, M_size, distance, 1, h_idx + i);//get the minimum index
		}
		/*printf("\nIndex values:\n");
		print_iarray(h_idx,D_size);*/
		cudaMemcpy(d_idx,h_idx,D_size*sizeof(int),cudaMemcpyHostToDevice);
		
		Index_model<<<WIDTH,D_size/WIDTH>>>(d_M, d_idx, d_M_idx);//calculate which points from the model cloud will be used based on the index array
		cudaDeviceSynchronize(); 
		//cudaMemcpy(h_M_idx,d_M_idx,bytesD,cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_Dt,d_Dt,bytesD,cudaMemcpyDeviceToHost);
		//printf("\nSelected model points: \n");
		//print_cloud(h_M_idx,D_size,D_size);
		//checkCudaErrors(cudaGetLastError());*/
		//printf("Matching step done!\n");

		//Minimization step (point-to-point)
		cublasDcopy(handle,3*D_size,d_Dt,1,d_aux,1);
		centroid<<<WIDTH,D_size/WIDTH>>>(d_aux, D_size, d_barD);//for data cloud
		cudaDeviceSynchronize();
		deviation<<<WIDTH,D_size/WIDTH>>>(d_Dt,d_barD,d_devD);
		cudaDeviceSynchronize();

		cublasDcopy(handle,3*D_size,d_M_idx,1,d_aux,1);
		centroid<<<WIDTH,D_size/WIDTH>>>(d_aux, D_size, d_barM);//for model cloud
		cudaDeviceSynchronize();
		deviation<<<WIDTH,D_size/WIDTH>>>(d_M_idx,d_barM,d_devM);
		cudaDeviceSynchronize();

		/*cudaMemcpy(h_barD,d_barD,3*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_barM,d_barM,3*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_devD,d_devD,bytesD,cudaMemcpyDeviceToHost);
		cudaMemcpy(h_devM,d_devM,bytesD,cudaMemcpyDeviceToHost);
		printf("D centroid:\n");
		print_darray(h_barD,3);
		printf("M centroid:\n");
		print_darray(h_barM,3);
		printf("D deviations:\n");
		print_cloud(h_devD,D_size,D_size);
		printf("M deviations:\n");
		print_cloud(h_devM,D_size,D_size);*/

		//d_W = d_devM * d_devD(t)
		alpha = 1; beta = 0;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, D_size,
					&alpha, d_devM, 3, d_devD, 3, &beta, d_W, 3);
		/*cudaMemcpy(h_W,d_W,9*sizeof(double),cudaMemcpyDeviceToHost);
		printf("W:\n");
		print_cloud(h_W,3,3);*/

		//SVD
		//d_W = d_U * d_S * d_VT
		cusolverDnDgesvd(cusolverH, 'A', 'A', 3, 3, d_W, 3, d_S, d_U, 3, d_VT, 3, 
						d_work, lwork, d_rwork, devInfo);
		
		//Calculate the temporary rotation matrix (d_temp_r)
		//R = U*Vt
		alpha = 1; beta = 0;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, 
					&alpha, d_U, 3, d_VT, 3, &beta, d_temp_r, 3);
		
		//Calculate the temporary translation vector (d_temp_t)
		//T = uM - R*uD
		alpha = -1; beta = 1;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 3, 
					&alpha, d_temp_r, 3, d_barD, 3, &beta, d_barM, 3);
		cublasDcopy(handle, 3, d_barM, 1, d_temp_T, 1);//copy the result to d_temp_t
		//printf("Minimization step done!\n");
		
		//cudaMemcpy(h_temp_r,d_temp_r,9*sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_temp_T,d_temp_T,3*sizeof(double),cudaMemcpyDeviceToHost);
		/*printf("Rotation matrix:\n");
		print_cloud(h_temp_r,3,3);
		printf("Translation vector:\n");
		print_darray(h_temp_T,3);*/

		//Transformation step
		//D = R * D + T
		repmat<<<WIDTH,D_size/WIDTH>>>(d_temp_T, rep_T);
		cudaDeviceSynchronize();
		alpha = 1; beta = 1;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, D_size, 3, 
					&alpha, d_temp_r, 3, d_Dt, 3, &beta, rep_T, 3);
		cublasDcopy(handle, 3 * D_size, rep_T, 1, d_Dt, 1);

		cudaMemcpy(h_Dt,d_Dt,bytesD,cudaMemcpyDeviceToHost);
		/*printf("\nTransformed data point cloud (iteration #%d):\n\n",iterations+1);
		print_cloud(h_Dt, D_size, D_size);*/

		/*cudaMemcpy(h_M_idx,d_M_idx,bytesD,cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Dt,d_Dt,bytesD,cudaMemcpyDeviceToHost);
		printf("\nSelected data points: \n");
		print_cloud(h_Dt,D_size,D_size);
		printf("\nSelected model points: \n");
		print_cloud(h_M_idx,D_size,D_size);*/

		//Error
		cudaMemcpy(d_iteration, &iterations, sizeof(int), cudaMemcpyHostToDevice);
		Error<<<3, D_size>>>(d_aux,d_Dt,d_M_idx,d_error,d_iteration,D_size);
		cudaDeviceSynchronize();

		iterations++;
		if (iterations > MAX_ITER-1) break;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cublasDestroy(handle);

	cudaMemcpy(h_error, d_error, MAX_ITER*sizeof(double), cudaMemcpyDeviceToHost);
	// printf("Error:\n");
	// print_darray(h_error,iterations);

	printf("Elapsed time: %f ms\n",milliseconds);

	/////////////////////////////////////////////End of 2nd//////////////////////////////////////
	free(h_idx);free(h_i);free(h_error);free(h_Dt);
	cudaFree(d_Dt);
	cudaFree(d_M);
	cudaFree(d_idx);
	cudaFree(d_i);
	cudaFree(distance);
	cudaFree(d_M_idx);
	cudaFree(d_barD);
	cudaFree(d_barM);
	cudaFree(d_devD);
	cudaFree(d_devM);
	cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(devInfo);
	cudaFree(d_W);
	cudaFree(d_work);
	cudaFree(d_temp_r);
	cudaFree(d_temp_T);
	cudaFree(rep_T);
	cudaFree(d_error);
	cudaFree(d_aux);

    return 0;
}