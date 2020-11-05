//For debugging:
//nvcc ICP_standard.cu -lcublas -lcurand -lcusolver -o ICP_cuda

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
//#include <device_functions.h>

//constants
#define WIDTH 1<<5
#define NUM_POINTS WIDTH*WIDTH //width of grid (1024 points)
#define XY_min -2.0
#define XY_max 2.0
#define MAX_ITER 2

void dmatrixMul(double* A, double* B, double* C, int m, int n, int k);
void print_cloud(double* cloud, int num_points, int points2show);
void print_darray(double* array, int points2show);
void print_iarray(int* array, int points2show);

__global__
void Matching(double* Dt, double* M, int m, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < NUM_POINTS)
	{
		double min = 100000;
		double d;
		for (int j = 0; j < m; j++)
		{
			d = sqrt(pow((Dt[0 + i * 3] - M[0 + j * 3]), 2) + pow((Dt[1 + i * 3] - M[1 + j * 3]), 2) + pow((Dt[2 + i * 3] - M[2 + j * 3]), 2));
			if (d < min)
			{
				min = d;
				idx[i] = j;
			}
		}
	}
}

__global__
void centr_dev(double* D, double* M, int* idx, double* barD, double* barM, double* devD, double* devM)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int size = blockDim.x * gridDim.x;

	if(i < NUM_POINTS)
	{
		//copy D cloud to devD
		devD[0 + i * 3] = D[0 + i * 3];
		devD[1 + i * 3] = D[1 + i * 3];
		devD[2 + i * 3] = D[2 + i * 3];

		//copy M cloud to devM using the correspondence(idx)
		devM[0 + i * 3] = M[0 + idx[i] * 3];
		devM[1 + i * 3] = M[1 + idx[i] * 3];
		devM[2 + i * 3] = M[2 + idx[i] * 3];
	}
	__syncthreads();

	for (int s = 1; s <= size; s *= 2)//parallel reduction
	{
		if (i % (2 * s) == 0)
		{
			//D
			devD[0 + i * 3] += devD[0 + (i + s) * 3];
			devD[1 + i * 3] += devD[1 + (i + s) * 3];
			devD[2 + i * 3] += devD[2 + (i + s) * 3];

			//M
			devM[0 + i * 3] += devM[0 + (i + s) * 3];
			devM[1 + i * 3] += devM[1 + (i + s) * 3];
			devM[2 + i * 3] += devM[2 + (i + s) * 3];
		}
		__syncthreads();
	}

	if (i >= 0 && i <= 2)
	{
		barD[i] = devD[i] / NUM_POINTS;
		barM[i] = devM[i] / NUM_POINTS;
		printf("barM[%d]: %lf\n",i,barD[i]);
	}
	__syncthreads();

	if(i < NUM_POINTS)
	{
		devD[0 + i * 3] = D[0 + i * 3] - barD[0];
		devD[1 + i * 3] = D[1 + i * 3] - barD[1];
		devD[2 + i * 3] = D[2 + i * 3] - barD[2];

		devM[0 + i * 3] = M[0 + idx[i] * 3] - barM[0];
		devM[1 + i * 3] = M[1 + idx[i] * 3] - barM[1];
		devM[2 + i * 3] = M[2 + idx[i] * 3] - barM[2];
	}
}

__global__
void repmat(double* vector, double* matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < NUM_POINTS)
	{
		matrix[0 + i * 3] = vector[0];
		matrix[1 + i * 3] = vector[1];
		matrix[2 + i * 3] = vector[2];
	}
}

__global__
void Error(double* aux, double* D, double* M, int* idx, double* error, int iteration)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int size = blockDim.x * gridDim.x;

	if(i < NUM_POINTS)
	{
		aux[0 + i * 3] = pow(M[0 + idx[i] * 3]-D[0 + i * 3],2);
		aux[1 + i * 3] = pow(M[1 + idx[i] * 3]-D[1 + i * 3],2);
		aux[2 + i * 3] = pow(M[2 + idx[i] * 3]-D[2 + i * 3],2);
	}
	__syncthreads();

	for (int s = 1; s <= size; s *= 2)//parallel reduction
	{
		if (i % (2 * s) == 0)
		{
			aux[0 + i * 3] += aux[0 + (i + s) * 3];
			aux[1 + i * 3] += aux[1 + (i + s) * 3];
			aux[2 + i * 3] += aux[2 + (i + s) * 3];
		}
		__syncthreads();
	}

	if (i == 0) error[iteration] = sqrt( (aux[0] + aux[1] + aux[2]) / NUM_POINTS);
}

int main()
{
	int i, j, k;
	float ti[3], ri[3];
	float lin_space[WIDTH], lenght;
	float* mesh_x = NULL, * mesh_y = NULL, * z = NULL;
	float* h_D = NULL, * h_M = NULL;

	lenght = XY_max - XY_min;

	size_t bytesD = NUM_POINTS * 3 * sizeof(float);
	size_t bytesM = NUM_POINTS * 3 * sizeof(float);

	//create an array with all points equally separated
	int n = WIDTH;
	for (i = 0; i < WIDTH; i++)
	{
		lin_space[i] = XY_min + float(i) * (lenght) / (float(n) - 1.0f);
	}

	//create meshgrid for x and y coordinates
	mesh_x = (float*)malloc(num_points * sizeof(float));
	mesh_y = (float*)malloc(num_points * sizeof(float));
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
	z = (float*)malloc(num_points * sizeof(float));
	for (i = 0; i < num_points; i++) z[i] = pow(mesh_x[i], 2) - pow(mesh_y[i], 2);

	//Create data point cloud matrix (source point cloud)
	//points are stored in this order: x1y1z1, x2y2z2, x3y3z3, ....
	h_D = (float*)malloc(bytesD);
	k = 0;
	for (i = 0; i < num_points; i++)
	{
		for (j = 0; j < 3; j++)
		{
			if (j == 0) h_D[k] = mesh_x[i];
			if (j == 1) h_D[k] = mesh_y[i];
			if (j == 2) h_D[k] = z[i];
			k++;
		}
	}

	/*printf("Data point cloud\n");
	print_cloud(h_D, num_points, num_points);*/

	//Create model point cloud matrix (target point cloud)
	//every matrix is defined using the colum-major order
	h_M = (float*)malloc(bytesM);

	//Translation values
	ti[0] = 1.0f;//x
	ti[1] = -0.3f;//y
	ti[2] = 0.2f;//z

	//Rotation values (rad)
	ri[0] = 1.0f;//axis x
	ri[1] = -0.5f;//axis y
	ri[2] = 0.05f;//axis z

	float h_rx[9] = {
		1.0f, 0.0f, 0.0f,
		0.0f, cos(ri[0]), sin(ri[0]),
		0.0f, -sin(ri[0]), cos(ri[0]) };
	float h_ry[9] = {
		cos(ri[1]), 0.0f, -sin(ri[1]),
		0.0f, 1, 0.0f,
		sin(ri[1]), 0.0f, cos(ri[1]) };
	float h_rz[9] = {
		cos(ri[2]), sin(ri[2]),0.0f,
		-sin(ri[2]), cos(ri[2]), 0.0f,
		0.0f, 0.0f, 1.0f };

	//calculate the rotation matrix h_r
	/*double h_r[9] = {};
	dmatrixMul(h_rx, h_ry, h_r, 3, 3, 3);
	dmatrixMul(h_r, h_rz, h_r, 3, 3, 3);*/
	float h_r[9] = {0.876485812,-0.37591464,0.300767018,
			-0.04386084,0.559789799,0.827473024,
			-0.47942553,-0.73846026,0.474159881};

	//h_M = h_r*h_D
	dmatrixMul(h_r, h_D, h_M, 3, NUM_POINTS, 3);
	//h_M = h_M + t
	for (i = 0; i < NUM_POINTS; i++)
	{
		for (j = 0; j < 3; j++)
		{
			h_M[j + i * 3] += ti[j];
		}
	}

	/*printf("\nModel point cloud\n");
	print_cloud(h_M, num_points, num_points);*/

	free(mesh_x); free(mesh_y); free(z);
	/////////////////////////////////////////////End of 1st//////////////////////////////////////

	//////////////////////////////////////////2nd:ICP Algorithm//////////////////////////////////

	int D_size = NUM_POINTS;//data cloud number of points
	int M_size = NUM_POINTS;//model cloud number of points

	float* d_M = NULL;//model cloud (device)
	float* d_Dt = NULL;//data cloud transformed (device)
	
	int* d_idx = NULL;//index vector (used for correspondence)
	int* h_idx = NULL;//index vector (used for correspondence)
	
	float* d_barD = NULL, * d_barM = NULL;//centroids (device)
	float* d_devD = NULL, * d_devM = NULL;//deviations (device)
	
	float* d_W = NULL;//3x3 matrix used in SVD
	float* d_S = NULL;
	float* d_U = NULL;
	float* d_VT = NULL;
	float* d_work = NULL;
	float* d_rwork = NULL;
	int* devInfo = NULL;
	int lwork = 0;
	float alpha, beta;

	float* d_temp_r = NULL;
	float* d_temp_T = NULL;
	float* rep_T = NULL;

	float* d_error = NULL;
	float* h_error = NULL;

	h_idx = (int*)malloc(D_size * sizeof(int));
	cudaMalloc(&d_idx, D_size * sizeof(int));

	//cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaMalloc(&d_Dt, bytesD);
	cudaMalloc(&d_M, bytesM);
	cudaMemcpy(d_Dt, h_D, bytesD, cudaMemcpyHostToDevice);//copy data cloud to d_Dt
	cudaMemcpy(d_M, h_M, bytesM, cudaMemcpyHostToDevice);//copy model cloud to d_M

	cudaMalloc(&d_barD, 3 * sizeof(double));
	cudaMalloc(&d_barM, 3 * sizeof(double));
	cudaMemset(d_barD, 0, 3 * sizeof(double));
	cudaMemset(d_barM, 0, 3 * sizeof(double));

	cudaMalloc(&d_devD, bytesD);
	cudaMalloc(&d_devM, bytesD);
	cudaMemset(d_devD, 0, bytesD);
	cudaMemset(d_devM, 0, bytesD);

	//double *h_Dt = (double*)malloc(3*n*sizeof(double));
	
	float* h_barD, * h_barM;
	float* h_devD, * h_devM;
	h_barD = (float*)malloc(3 * sizeof(float));
	h_barM = (float*)malloc(3 * sizeof(float));
	h_devD = (float*)malloc(bytesD);
	h_devM = (float*)malloc(bytesD);

	cudaMalloc(&d_S, sizeof(float) * 3);
	cudaMalloc(&d_U, sizeof(float) * 9);
	cudaMalloc(&d_VT, sizeof(float) * 9);
	cudaMalloc(&devInfo, sizeof(int));
	cudaMalloc(&d_W, sizeof(float) * 9);
	cusolverDnHandle_t cusolverH;//cuSolver handle
	cusolverDnCreate(&cusolverH);
	cusolverDnDgesvd_bufferSize(cusolverH, 3, 3, &lwork);
	cudaMalloc((void**)&d_work, sizeof(float) * lwork);

	cudaMalloc(&d_temp_r, sizeof(float) * 9);
	cudaMalloc(&d_temp_T, sizeof(float) * 3);
	cudaMalloc(&rep_T, sizeof(float) * 3 * D_size);

	cudaMalloc(&d_error, MAX_ITER * sizeof(float));
	cudaMemset(d_error, 0, MAX_ITER * sizeof(float));
	h_error = (float*)malloc(MAX_ITER * sizeof(float));

	cudaError_t err;//for checking errors in kernels

	//printf("Here starts the main loop!\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int iteration = 0;
	cudaEventRecord(start);
	//MAIN LOOP

	int factor = 16;
	int GridSize = factor;
	int BlockSize = n/factor;
	printf("Grid Size: %d, Block Size: %d\n",GridSize,BlockSize);

	while (iteration < MAX_ITER)
	{
		/////////////////matching step/////////////////
		Matching<<<GridSize,BlockSize>>>(d_Dt, d_M, M_size, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_idx, d_idx, D_size*sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index values:\n");
		print_iarray(h_idx, 20);*/
		/////////////////end of matching/////////////////

		/////////////////minimization step/////////////////
		//cudaMemcpy(d_devD, d_Dt, bytesD, cudaMemcpyDeviceToDevice);
		centr_dev<<<GridSize,BlockSize>>>(d_Dt, d_M, d_idx, d_barD, d_barM, d_devD, d_devM);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in centroid kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_D, d_devD, 3*n*sizeof(double), cudaMemcpyDeviceToHost);
		printf("D dev:\n");
		print_cloud(h_D, n, n);*/

		/*cudaMemcpy(h_barD, d_barD, 3*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_barM, d_barM, 3*sizeof(double), cudaMemcpyDeviceToHost);
		printf("D centroid:\n");
		print_darray(h_barD, 3);
		printf("M centroid:\n");
		print_darray(h_barM, 3);*/

		//d_W = d_devM * d_devD(t)
		alpha = 1; beta = 0;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, D_size,
			&alpha, d_devM, 3, d_devD, 3, &beta, d_W, 3);

		//SVD
		//d_W = d_U * d_S * d_VT
		cusolverDnDgesvd(cusolverH, 'A', 'A', 3, 3, d_W, 3, d_S, d_U, 3, d_VT, 3,
			d_work, lwork, d_rwork, devInfo);

		//Calculate the temporary rotation matrix (d_temp_r)
		//R = U*Vt
		alpha = 1; beta = 0;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3,
			&alpha, d_U, 3, d_VT, 3, &beta, d_temp_r, 3);

		//T = uM - R*uD
		alpha = -1; beta = 1;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 3,
			&alpha, d_temp_r, 3, d_barD, 3, &beta, d_barM, 3);
		cublasDcopy(handle, 3, d_barM, 1, d_temp_T, 1);//copy the result to d_temp_T
		/////////////////end of minimization/////////////////

		/////////////////transformation step/////////////////

		//D = R * D + T
		repmat <<<GridSize, BlockSize>>> (d_temp_T, rep_T);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in repmat kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		alpha = 1; beta = 1;
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, D_size, 3,
			&alpha, d_temp_r, 3, d_Dt, 3, &beta, rep_T, 3);
		cublasDcopy(handle, 3 * D_size, rep_T, 1, d_Dt, 1);

		/////////////////end of transformation step/////////////////

		//Error
		cudaMemset(d_devD, 0, 3 * n * sizeof(double));
		Error <<<GridSize, BlockSize>>> (d_devD, d_Dt, d_M, d_idx, d_error, iteration);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in error kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		iteration++;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cublasDestroy(handle);

	cudaMemcpy(h_error, d_error, MAX_ITER * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Error:\n");
	print_darray(h_error, iteration);

	printf("Elapsed time: %f ms\n", milliseconds);

	free(h_D), free(h_M), free(h_idx), free(h_error), free(h_barD), free(h_barM), free(h_devD), free(h_devM);
	cudaFree(d_Dt), cudaFree(d_M), cudaFree(d_idx), cudaFree(d_barM), cudaFree(d_barD), cudaFree(d_devD), cudaFree(d_devM);
	cudaFree(d_W), cudaFree(d_S), cudaFree(d_U), cudaFree(d_VT), cudaFree(d_work), cudaFree(d_rwork), cudaFree(devInfo);
	cudaFree(d_temp_r), cudaFree(d_temp_T), cudaFree(rep_T), cudaFree(d_error);

	return 0;
}

//double matrix multiplication
void dmatrixMul(double* A, double* B, double* C, int m, int n, int k)
{
	int i, j, q;
	double temp = 0.0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			temp = 0.0f;
			for (q = 0; q < k; q++) temp += A[j + q * m] * B[q + i * k];
			C[j + i * m] = (double)temp;
		}
	}
}

//print matrix
void print_cloud(double* cloud, int num_points, int points2show)//
{
	int i, j, offset;
	printf("x\ty\tz\n");
	if (points2show <= num_points)
	{
		for (i = 0; i < points2show; i++)
		{
			for (j = 0; j < 3; j++)
			{
				offset = j + i * 3;
				printf("%.4f\t", cloud[offset]);
				if (j % 3 == 2) printf("\n");
			}
		}
	}
	else printf("The cloud can't be printed\n\n");
}

//print vector with double values
void print_darray(double* array, int points2show)
{
	int i;
	for (i = 0; i < points2show; i++)
	{
		printf("%.3f ", array[i]);
	}
	printf("\n");
}

//print vector with integer values
void print_iarray(int* array, int points2show)
{
	int i;
	for (i = 0; i < points2show; i++)
	{
		printf("%d ", array[i]);
	}
	printf("\n");
}
