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
#define WIDTH 32
#define NUM_POINTS WIDTH*WIDTH //width of grid (1024 points)
#define XY_min -2.0
#define XY_max 2.0
#define MAX_ITER 2

void SmatrixMul(float* A, float* B, float* C, int m, int n, int k);
void printScloud(float* cloud, int num_points, int points2show);
void printSarray(float* array, int points2show);
void printIarray(int* array, int points2show);

__global__
void Matching(float* Dt, float* M, int m, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_POINTS)
	{
		float min = 100000;
		float d;
		for (int j = 0; j < m; j++)
		{
			d = (float)sqrt((Dt[0 + i * 3] - M[0 + j * 3]) * (Dt[0 + i * 3] - M[0 + j * 3]) +
				(Dt[1 + i * 3] - M[1 + j * 3]) * (Dt[1 + i * 3] - M[1 + j * 3]) +
				(Dt[2 + i * 3] - M[2 + j * 3]) * (Dt[2 + i * 3] - M[2 + j * 3]));
			if (d < min)
			{
				min = d;
				idx[i] = j;
			}
		}
	}
}

__global__
void centroid(int n, float* D, float* M, int* idx, float* barD, float* barM, float* devD, float* devM)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//copy D cloud to devD
	devD[0 + i * 3] = D[0 + i * 3];
	devD[1 + i * 3] = D[1 + i * 3];
	devD[2 + i * 3] = D[2 + i * 3];

	//copy M cloud to devM using the correspondence(idx)
	devM[0 + i * 3] = M[0 + idx[i] * 3];
	devM[1 + i * 3] = M[1 + idx[i] * 3];
	devM[2 + i * 3] = M[2 + idx[i] * 3];
	__syncthreads();

	for (int s = 1; s < n; s *= 2)//parallel reduction
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

	if (i < 3)
	{
		barD[i] = devD[i] / (float)n;
		barM[i] = devM[i] / (float)n;
	}
}

__global__
void deviation(float* D, float* M, int* idx, float* barD, float* barM, float* devD, float* devM)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	devD[0 + i * 3] = D[0 + i * 3] - barD[0];
	devD[1 + i * 3] = D[1 + i * 3] - barD[1];
	devD[2 + i * 3] = D[2 + i * 3] - barD[2];

	devM[0 + i * 3] = M[0 + idx[i] * 3] - barM[0];
	devM[1 + i * 3] = M[1 + idx[i] * 3] - barM[1];
	devM[2 + i * 3] = M[2 + idx[i] * 3] - barM[2];
}

__global__
void RyT(float* R, float* T, float* P, float* Q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] + R[0 + 2 * 3] * P[2 + i * 3] + T[0];
	Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] + R[1 + 2 * 3] * P[2 + i * 3] + T[1];
	Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] + R[2 + 2 * 3] * P[2 + i * 3] + T[2];
}

__global__
void Error(int n, float* aux, float* D, float* M, int* idx, float* error, int iteration)
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	aux[0 + i * 3] = (M[0 + idx[i] * 3] - D[0 + i * 3]) * (M[0 + idx[i] * 3] - D[0 + i * 3]);
	aux[1 + i * 3] = (M[1 + idx[i] * 3] - D[1 + i * 3]) * (M[1 + idx[i] * 3] - D[1 + i * 3]);
	aux[2 + i * 3] = (M[2 + idx[i] * 3] - D[2 + i * 3]) * (M[2 + idx[i] * 3] - D[2 + i * 3]);
	__syncthreads();
	//printf("aux[%d]: %.3f %.3f %.3f\n", i, aux[0 + i * 3], aux[1 + i * 3], aux[2 + i * 3]);

	sdata[tid] = aux[0 + i * 3] + aux[1 + i * 3] + aux[2 + i * 3];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2)//parallel reduction
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) aux[blockIdx.x] = (float)sqrt(sdata[0] / (float)n);
	__syncthreads();

	if (i == 0)
	{
		for (unsigned int j = 0; j < gridDim.x; j++)
			error[iteration] += aux[j];
	}
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
		lin_space[i] = (float)XY_min + float(i) * (lenght) / (float(n) - 1.0f);
	}

	//create meshgrid for x and y coordinates
	mesh_x = (float*)malloc(NUM_POINTS * sizeof(float));
	mesh_y = (float*)malloc(NUM_POINTS * sizeof(float));
	i = 0;
	k = 0;
	while (i < NUM_POINTS)
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
	z = (float*)malloc(NUM_POINTS * sizeof(float));
	for (i = 0; i < NUM_POINTS; i++) z[i] = (mesh_x[i])* (mesh_x[i]) - (mesh_y[i])* (mesh_y[i]);

	//Create data point cloud matrix (source point cloud)
	//points are stored in this order: x1y1z1, x2y2z2, x3y3z3, ....
	h_D = (float*)malloc(bytesD);
	k = 0;
	for (i = 0; i < NUM_POINTS; i++)
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

	float h_r[9] = {};
	float cx = (float)cos(ri[0]); float cy = (float)cos(ri[1]); float cz = (float)cos(ri[2]);
	float sx = (float)sin(ri[0]); float sy = (float)sin(ri[1]); float sz = (float)sin(ri[2]);
	h_r[0] = cy * cz; h_r[1] = (cz * sx * sy) + (cx * sz); h_r[2] = -(cx * cz * sy) + (sx * sz);
	h_r[3] = -cy * sz; h_r[4] = (cx * cz) - (sx * sy * sz); h_r[5] = (cx * sy * sz) + (cz * sx);
	h_r[6] = sy; h_r[7] = -cy * sx; h_r[8] = cx * cy;
	//printf("Ri:\n");
	//printScloud(h_r, 3, 3);

	//h_M = h_r*h_D
	SmatrixMul(h_r, h_D, h_M, 3, NUM_POINTS, 3);
	//h_M = h_M + t
	for (i = 0; i < NUM_POINTS; i++)
	{
		for (j = 0; j < 3; j++)
		{
			h_M[j + i * 3] += ti[j];
		}
	}

	//printf("\nModel point cloud\n");
	//printScloud(h_M, NUM_POINTS, NUM_POINTS);

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

	cudaMalloc(&d_barD, 3 * sizeof(float));
	cudaMalloc(&d_barM, 3 * sizeof(float));
	cudaMemset(d_barD, 0, 3 * sizeof(float));
	cudaMemset(d_barM, 0, 3 * sizeof(float));

	cudaMalloc(&d_devD, bytesD);
	cudaMalloc(&d_devM, bytesD);
	cudaMemset(d_devD, 0, bytesD);
	cudaMemset(d_devM, 0, bytesD);

	//float *h_Dt = (float*)malloc(3*n*sizeof(float));

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

	float* h_W = (float*)malloc(sizeof(float) * 9);

	cudaMalloc(&d_temp_r, sizeof(float) * 9);
	cudaMalloc(&d_temp_T, sizeof(float) * 3);
	cudaMalloc(&rep_T, sizeof(float) * 3 * D_size);

	cudaMalloc(&d_error, MAX_ITER * sizeof(float));
	cudaMemset(d_error, 0, MAX_ITER * sizeof(float));
	h_error = (float*)malloc(MAX_ITER * sizeof(float));

	cudaError_t err;//for checking errors in kernels

	//printf("Here starts the main loop!\n");
	int GridSize = 8;
	int BlockSize = NUM_POINTS / GridSize;
	printf("Grid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	cublasStatus_t cublas_error;

	//MAIN LOOP
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int iteration = 0;
	cudaEventRecord(start);

	while (iteration < MAX_ITER)
	{
		/////////////////matching step/////////////////
		Matching <<< GridSize, BlockSize >>> (d_Dt, d_M, M_size, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_idx, d_idx, D_size*sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index values:\n");
		printIarray(h_idx, D_size);*/
		/////////////////end of matching/////////////////

		/////////////////minimization step/////////////////
		//cudaMemcpy(d_devD, d_Dt, bytesD, cudaMemcpyDeviceToDevice);
		centroid << <GridSize, BlockSize >> > (D_size, d_Dt, d_M, d_idx, d_barD, d_barM, d_devD, d_devM);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in centroid kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		deviation << <GridSize, BlockSize >> > (d_Dt, d_M, d_idx, d_barD, d_barM, d_devD, d_devM);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in deviation kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		cudaMemcpy(h_D, d_devD, bytesD, cudaMemcpyDeviceToHost);
		printf("D dev[%d]:\n", iteration + 1);
		printScloud(h_D, NUM_POINTS, NUM_POINTS);
		cudaMemcpy(h_D, d_devM, bytesD, cudaMemcpyDeviceToHost);
		printf("M dev[%d]:\n", iteration + 1);
		printScloud(h_D, NUM_POINTS, NUM_POINTS);

		cudaMemcpy(h_barD, d_barD, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_barM, d_barM, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("D centroid[%d]:\n", iteration + 1);
		printSarray(h_barD, 3);
		printf("M centroid[%d]:\n", iteration + 1);
		printSarray(h_barM, 3);

		//d_W = d_devM * d_devD(t)
		alpha = 1; beta = 0;
		cublas_error = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, D_size,
			&alpha, d_devM, 3, d_devD, 3, &beta, d_W, 3);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
			printf("Error in W:  %s\n", cudaGetErrorString(err));

		/*cudaMemcpy(h_W, d_W, 9 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("W:\n");
		printScloud(h_W, 3, 3);*/

		//SVD
		//d_W = d_U * d_S * d_VT
		cusolverDnSgesvd(cusolverH, 'A', 'A', 3, 3, d_W, 3, d_S, d_U, 3, d_VT, 3,
			d_work, lwork, d_rwork, devInfo);

		//Calculate the temporary rotation matrix (d_temp_r)
		//R = U*Vt
		alpha = 1; beta = 0;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3,
			&alpha, d_U, 3, d_VT, 3, &beta, d_temp_r, 3);

		//T = uM - R*uD
		alpha = -1; beta = 1;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 3,
			&alpha, d_temp_r, 3, d_barD, 3, &beta, d_barM, 3);
		cublasScopy(handle, 3, d_barM, 1, d_temp_T, 1);//copy the result to d_temp_T
		/////////////////end of minimization/////////////////

		/////////////////transformation step/////////////////

		//D = R * D + T
		/*repmat << <GridSize, BlockSize >> > (d_temp_T, rep_T);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in repmat kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		alpha = 1; beta = 1;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, D_size, 3,a
			&alpha, d_temp_r, 3, d_Dt, 3, &beta, rep_T, 3);
		cublasScopy(handle, 3 * D_size, rep_T, 1, d_Dt, 1);*/
		RyT << <GridSize, BlockSize >> > (d_temp_r, d_temp_T, d_Dt, d_devD);
		cublasScopy(handle, 3 * D_size, d_devD, 1, d_Dt, 1);

		/////////////////end of transformation step/////////////////

		//Error
		cudaMemset(d_devD, 0, bytesD);
		Error << <GridSize, BlockSize, BlockSize * sizeof(float) >> > (D_size, d_devD, d_Dt, d_M, d_idx, d_error, iteration);
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

	cudaMemcpy(h_error, d_error, MAX_ITER * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Error:\n");
	printSarray(h_error, MAX_ITER);

	printf("Elapsed time: %f ms\n", milliseconds);

	free(h_D), free(h_M), free(h_idx), free(h_error), free(h_barD), free(h_barM), free(h_devD), free(h_devM);
	cudaFree(d_Dt), cudaFree(d_M), cudaFree(d_idx), cudaFree(d_barM), cudaFree(d_barD), cudaFree(d_devD), cudaFree(d_devM);
	cudaFree(d_W), cudaFree(d_S), cudaFree(d_U), cudaFree(d_VT), cudaFree(d_work), cudaFree(d_rwork), cudaFree(devInfo);
	cudaFree(d_temp_r), cudaFree(d_temp_T), cudaFree(rep_T), cudaFree(d_error);

	return 0;
}

//double matrix multiplication colum-major order
void SmatrixMul(float* A, float* B, float* C, int m, int n, int k)
{
	int i, j, q;
	float temp = 0.0f;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			temp = 0.0f;
			for (q = 0; q < k; q++) temp += A[j + q * m] * B[q + i * k];
			C[j + i * m] = temp;
		}
	}
}

//print matrix
void printScloud(float* cloud, int num_points, int points2show)
{
	int i, j, offset;
	printf("x\ty\tz\n");
	if (points2show <= num_points)
	{
		for (i = 0; i < points2show; i++)
		{
			printf("%d: ", i + 1);
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
void printSarray(float* array, int points2show)
{
	int i;
	for (i = 0; i < points2show; i++)
	{
		printf("%.4f ", array[i]);
	}
	printf("\n");
}

//print vector with integer values
void printIarray(int* array, int points2show)
{
	int i;
	for (i = 0; i < points2show; i++)
	{
		printf("%d ", array[i]);
	}
	printf("\n");
}
