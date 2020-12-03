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
#define MAX_ITER 40

void SmatrixMul(float* A, float* B, float* C, int m, int n, int k);
void printScloud(float* cloud, int num_points, int points2show);
void printSarray(float* array, int points2show);
void printIarray(int* array, int points2show);

__global__
void Matching(float* P, float* Q, int q_points, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float min = 100000;
	float d;
	for (int j = 0; j < q_points; j++)
	{
		d = (float)sqrt((P[0 + i * 3] - Q[0 + j * 3]) * (P[0 + i * 3] - Q[0 + j * 3]) +
			(P[1 + i * 3] - Q[1 + j * 3]) * (P[1 + i * 3] - Q[1 + j * 3]) +
			(P[2 + i * 3] - Q[2 + j * 3]) * (P[2 + i * 3] - Q[2 + j * 3]));
		if (d < min)
		{
			min = d;
			idx[i] = j;
		}
	}
}

__global__
void Q_index(float* Q, int* idx, float* Q_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Q_idx[0 + i * 3] = Q[0 + idx[i] * 3];
	Q_idx[1 + i * 3] = Q[1 + idx[i] * 3];
	Q_idx[2 + i * 3] = Q[2 + idx[i] * 3];
}

__global__
void deviation(float* P, float* Q, float* barP, float* barQ, float* devP, float* devQ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	devP[0 + i * 3] = P[0 + i * 3] - barP[0];
	devP[1 + i * 3] = P[1 + i * 3] - barP[1];
	devP[2 + i * 3] = P[2 + i * 3] - barP[2];

	devQ[0 + i * 3] = Q[0 + i * 3] - barQ[0];
	devQ[1 + i * 3] = Q[1 + i * 3] - barQ[1];
	devQ[2 + i * 3] = Q[2 + i * 3] - barQ[2];
}

__global__
void RyT(float* R, float* T, float* P, float* Q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] + R[0 + 2 * 3] * P[2 + i * 3] + T[0];
	Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] + R[1 + 2 * 3] * P[2 + i * 3] + T[1];
	Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] + R[2 + 2 * 3] * P[2 + i * 3] + T[2];
}

int main()
{
	int num_points, i, j, k;
	float ti[3], ri[3];
	float lin_space[WIDTH], lenght;

	num_points = WIDTH * WIDTH;//number of points
	lenght = XY_max - XY_min;

	//for this specific case the number of points of the 2 clouds are the same
	int d_points = num_points;
	int m_points = num_points;

	////////////////1st:Creation of the synthetic data//////////////

	//create an array with all points equally separated
	int n = WIDTH;
	for (i = 0; i < WIDTH; i++)
	{
		lin_space[i] = (float)XY_min + ((float)i * (float)lenght) / (float(n) - 1.0f);
	}

	//create the meshgrid
	float* mesh_x = (float*)malloc(num_points * sizeof(float));
	float* mesh_y = (float*)malloc(num_points * sizeof(float));

	if ((mesh_x != NULL) && (mesh_y != NULL))
	{
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
	}
	else return 0;

	//Create the function z = f(x,y) = x^2-y^2
	float* z = (float*)malloc(num_points * sizeof(float));
	for (i = 0; i < num_points; i++) z[i] = pow(mesh_x[i], 2) - pow(mesh_y[i], 2);

	//Create data point cloud matrix
	size_t bytesD = (size_t)d_points * (size_t)3 * sizeof(float);
	float* h_D = (float*)malloc(bytesD);

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

	//printf("Data point cloud\n");
	//printScloud(h_D, num_points, num_points);

	//Translation values
	ti[0] = 0.8f;//x
	ti[1] = -0.3f;//y
	ti[2] = 0.2f;//z

	//Rotation values (rad)
	ri[0] = 0.2f;//axis x
	ri[1] = -0.2f;//axis y
	ri[2] = 0.05f;//axis z

	float h_r[9] = {};
	float cx = (float)cos(ri[0]); float cy = (float)cos(ri[1]); float cz = (float)cos(ri[2]);
	float sx = (float)sin(ri[0]); float sy = (float)sin(ri[1]); float sz = (float)sin(ri[2]);
	h_r[0] = cy * cz; h_r[1] = (cz * sx * sy) + (cx * sz); h_r[2] = -(cx * cz * sy) + (sx * sz);
	h_r[3] = -cy * sz; h_r[4] = (cx * cz) - (sx * sy * sz); h_r[5] = (cx * sy * sz) + (cz * sx);
	h_r[6] = sy; h_r[7] = -cy * sx; h_r[8] = cx * cy;
	//printf("Ri:\n");
	//printScloud(h_r,3,3);

	//Create model point cloud matrix (target point cloud)
	//every matrix is defined using the colum-major order
	size_t bytesM = (size_t)m_points * (size_t)3 * sizeof(float);
	float* h_M = (float*)malloc(bytesM);

	//h_M = h_r*h_D
	SmatrixMul(h_r, h_D, h_M, 3, num_points, 3);
	//h_M = h_M + t
	for (i = 0; i < num_points; i++)
	{
		for (j = 0; j < 3; j++)
		{
			h_M[j + i * 3] += ti[j];
		}
	}
	//printf("\nModel point cloud\n");
	//printScloud(h_M, m_points, m_points);

	free(mesh_x); free(mesh_y); free(z);
	/////////////////////////////////////////////End of 1st//////////////////////////////////////

	//since this lines 
	//p assumes the values of D
	//q assumes the values of M
	//number of p and q points
	int p_points = d_points;
	int q_points = m_points;
	float* d_p, * d_q;
	cudaMalloc(&d_p, bytesD);//p point cloud
	cudaMalloc(&d_q, bytesM);//q point cloud
	//transfer data from D and M to p and q
	cudaMemcpy(d_p, h_D, bytesD, cudaMemcpyHostToDevice);//copy data cloud to p
	cudaMemcpy(d_q, h_M, bytesM, cudaMemcpyHostToDevice);//copy model cloud to q
	cudaError_t err = cudaSuccess;//for checking errors in kernels
	//for measuring time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float miliseconds = 0.0f;
	//cuBLAS handle
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	cublasStatus_t cublas_error;
	//cuSolver handle
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	cusolverStatus_t cusolver_error;

	//////////////////////////////////////////2nd:ICP Algorithm//////////////////////////////////

	//Index vector (used for correspondence)
	int* h_idx = (int*)malloc(p_points * sizeof(int));
	int* d_idx = NULL;
	cudaMalloc(&d_idx, p_points * sizeof(int));

	//Q index cloud
	float* h_q_idx = (float*)malloc(bytesD);
	float* d_q_idx = NULL;
	cudaMalloc(&d_q_idx, bytesD);

	//Centroids and deviations
	float* h_barP = (float*)malloc(3 * sizeof(float));
	float* h_barQ = (float*)malloc(3 * sizeof(float));
	float* h_devP = (float*)malloc(bytesD);
	float* h_devQ = (float*)malloc(bytesD);
	float* d_barP = NULL, * d_barQ = NULL;
	float* d_devP = NULL, * d_devQ = NULL;
	cudaMalloc(&d_barP, 3 * sizeof(float));
	cudaMalloc(&d_barQ, 3 * sizeof(float));
	cudaMalloc(&d_devP, bytesD);
	cudaMalloc(&d_devQ, bytesD);
	float* h_unit = (float*)malloc(p_points * sizeof(float));
	if (h_unit != NULL) for (i = 0; i < p_points; i++) h_unit[i] = 1.0f;
	float* d_unit = NULL;
	cudaMalloc(&d_unit, p_points * sizeof(float));
	cudaMemcpy(d_unit, h_unit, p_points * sizeof(float), cudaMemcpyHostToDevice);

	//SVD
	float* h_W = (float*)malloc(sizeof(float) * 9);
	float* d_W = NULL;//3x3 matrix used in SVD
	float* d_S = NULL, * d_U = NULL, * d_VT = NULL;
	float* d_work = NULL, * d_rwork = NULL;
	int* devInfo = NULL;
	int lwork = 0;
	cudaMalloc(&d_S, sizeof(float) * 3);
	cudaMalloc(&d_U, sizeof(float) * 9);
	cudaMalloc(&d_VT, sizeof(float) * 9);
	cudaMalloc(&devInfo, sizeof(int));
	cudaMalloc(&d_W, sizeof(float) * 9);
	cusolverDnDgesvd_bufferSize(cusolverH, 3, 3, &lwork);
	cudaMalloc((void**)&d_work, sizeof(float) * lwork);

	float alpha = 0, beta = 0;//for cublas routines

	//Rotation matrix and translation vector
	float* h_temp_r = (float*)malloc(9 * sizeof(float));
	float* h_temp_T = (float*)malloc(3 * sizeof(float));
	float* d_temp_r = NULL, * d_temp_T = NULL;
	cudaMalloc(&d_temp_r, sizeof(float) * 9);
	cudaMalloc(&d_temp_T, sizeof(float) * 3);

	//Error estimation
	float* h_error = (float*)malloc((MAX_ITER + 1) * sizeof(float));
	if (h_error != NULL) for (i = 0; i < (MAX_ITER + 1); i++) h_error[i] = 0;
	/*float* d_error = NULL;
	cudaMalloc(&d_error, (MAX_ITER + 1) * sizeof(float));
	cudaMemset(d_error, 0, (MAX_ITER + 1) * sizeof(float));*/
	float* d_aux = NULL;
	cudaMalloc(&d_aux, bytesD);
	float partial_error = 0;

	//printf("Here starts the main loop!\n");
	int GridSize = 8;
	int BlockSize = NUM_POINTS / GridSize;
	printf("Grid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	//MAIN LOOP
	int iteration = 0;
	cudaEventRecord(start);
	while (iteration < MAX_ITER)
	{
		/////////////////Matching step/////////////////
		Matching << < GridSize, BlockSize >> > (d_p, d_q, q_points, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_idx, d_idx, p_points * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index  values[%d]:\n", iteration + 1);
		printIarray(h_idx, p_points);*/
		/////////////////end of matching/////////////////

		//get the Q indexed cloud
		Q_index << < GridSize, BlockSize >> > (d_q, d_idx, d_q_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in Q index kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		/////////////////Minimization step/////////////////

		//get the centroids
		alpha = 1 / (float)p_points; beta = 0;
		//P centroid
		cublas_error = cublasSgemv(cublasH, CUBLAS_OP_N, 3, p_points, &alpha, d_p, 3,
			d_unit, 1, &beta, d_barP, 1);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - P centroid\n");
			return (-1);
		}

		//Q centroid
		cublas_error = cublasSgemv(cublasH, CUBLAS_OP_N, 3, p_points, &alpha, d_q_idx, 3,
			d_unit, 1, &beta, d_barQ, 1);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - Q centroid\n");
			return (-1);
		}
		/*cublasGetVector(3, sizeof(float), d_barP, 1, h_barP, 1);
		cublasGetVector(3, sizeof(float), d_barQ, 1, h_barQ, 1);
		printf("P bar[%d]:\n", iteration + 1);
		printSarray(h_barP, 3);
		printf("Q bar[%d]:\n", iteration + 1);
		printSarray(h_barQ, 3);*/

		//get the deviation
		deviation << <GridSize, BlockSize >> > (d_p, d_q_idx, d_barP, d_barQ, d_devP, d_devQ);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in deviation kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		/*cudaMemcpy(h_devP, d_devP, bytesD, cudaMemcpyDeviceToHost);
		printf("P dev[%d]:\n", iteration + 1);
		printScloud(h_devP, p_points, p_points);
		cudaMemcpy(h_devQ, d_devQ, bytesD, cudaMemcpyDeviceToHost);
		printf("Q dev[%d]:\n", iteration + 1);
		printScloud(h_devQ, p_points, p_points);*/

		//d_W = d_devM * d_devD(t)
		alpha = 1; beta = 0;
		cublas_error = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, p_points,
									&alpha, d_devQ, 3, d_devP, 3, &beta, d_W, 3);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - W\n");
			return (-1);
		}

		/*cudaMemcpy(h_W, d_W, 9 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("W[%d]:\n", iteration + 1);
		printScloud(h_W, 3, 3);*/

		//SVD W = U * S * VT
		cusolver_error = cusolverDnSgesvd(cusolverH, 'A', 'A', 3, 3, d_W, 3, d_S, d_U, 3, d_VT, 3,
			d_work, lwork, d_rwork, devInfo);
		if (cusolver_error != CUSOLVER_STATUS_SUCCESS)
		{
			printf("Error in SVD\n");
			return (-1);
		}

		//Calculate the temporary rotation matrix (d_temp_r)
		//R = U * VT
		alpha = 1; beta = 0;
		cublas_error = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3,
			&alpha, d_U, 3, d_VT, 3, &beta, d_temp_r, 3);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - Rotation Matrix\n");
			return (-1);
		}

		//T = uM - R*uD
		alpha = -1; beta = 1;
		cublas_error = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 3,
			&alpha, d_temp_r, 3, d_barP, 3, &beta, d_barQ, 3);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - Translation Vector\n");
			return (-1);
		}
		cublasScopy(cublasH, 3, d_barQ, 1, d_temp_T, 1);//copy the result to d_temp_T
		/////////////////end of minimization/////////////////

		/////////////////Transformation step/////////////////

		//D = R * D + T
		RyT << <GridSize, BlockSize >> > (d_temp_r, d_temp_T, d_p, d_aux);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in RyT kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		cublasScopy(cublasH, 3 * p_points, d_aux, 1, d_p, 1);

		/////////////////end of transformation step/////////////////

		/////////////////Error estimation step/////////////////
		alpha = -1;
		cublasScopy(cublasH, 3 * p_points, d_p, 1, d_aux, 1);
		cublasSaxpy(cublasH, 3 * p_points, &alpha, d_q_idx, 1, d_aux, 1);
		cublasSnrm2(cublasH, 3 * p_points, d_aux, 1, &partial_error);
		h_error[iteration + 1] = partial_error / (float)sqrt(p_points);
		//printf("Current error (%d): %.4f\n", iteration + 1, h_error[iteration + 1]);
		/////////////////end of error estimation step/////////////////

		if ((h_error[iteration + 1] < 0.000001) ||
			((float)fabs((double)h_error[iteration + 1] - (double)h_error[iteration]) < 0.000001)) break;
		iteration++;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	printf("Error:\n");
	printSarray(h_error, iteration + 1);

	printf("ICP converged successfully!\n\n");

	printf("Elapsed time: %f ms\n", miliseconds);

	//destroy handles
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	//Free memory
	free(h_D), free(h_M);
	cudaFree(d_p), cudaFree(d_q);

	free(h_idx), cudaFree(d_idx);

	free(h_q_idx), cudaFree(d_q_idx);

	free(h_barP), free(h_barQ), free(h_devP), free(h_devQ), free(h_unit);
	cudaFree(d_barP), cudaFree(d_barQ), cudaFree(d_devP), cudaFree(d_devQ), cudaFree(d_unit);

	free(h_W);
	cudaFree(d_W), cudaFree(d_S), cudaFree(d_U), cudaFree(d_VT);
	cudaFree(d_work), cudaFree(d_rwork), cudaFree(devInfo);

	free(h_temp_r), free(h_temp_T);
	cudaFree(d_temp_r), cudaFree(d_temp_T);

	free(h_error), cudaFree(d_aux);

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
		printf("%d: %.4f\n", i + 1, array[i]);
	}
	printf("\n");
}

//print vector with integer values
void printIarray(int* array, int points2show)
{
	int i;
	for (i = 0; i < points2show; i++)
	{
		printf("%d: %d\n", i + 1, array[i]);
	}
	printf("\n");
}
