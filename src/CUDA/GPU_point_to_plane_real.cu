//ICP algorithm using the point-to-plane error metric and the Real data from a LiDAR OS1-16
//GPU version using CUDA and CUDA APIs
//By: Carlos Huapaya
//Libraries to include: cublas.lib, curand.lib, cusolver.lib
//For compiling linux:nvcc ICP_standard.cu -lcublas -lcurand -lcusolver -o ICP_cuda
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include "mkl.h"

#define MAX_ITER 100 //maximum amount of iterations

__global__
void Conversion(float* r, unsigned long int* encoder_count, float* altitude, float* azimuth, float* point_cloud)
{
	int azimuth_block, channel;
	unsigned long int counter;
	float theta, phi;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	azimuth_block = i / 16;
	counter = (encoder_count[0] + azimuth_block * 88) % 90112;
	channel = i % 16;
	theta = (float)(2 * M_PI * (counter / 90112.0 + azimuth[channel] / 360.0));
	phi = (float)(2 * M_PI * altitude[channel] / 360.0);
	point_cloud[0 + 3 * i] = (float)(r[i] * cos(theta) * cos(phi));//x
	point_cloud[1 + 3 * i] = (float)(-r[i] * sin(theta) * cos(phi));//y
	point_cloud[2 + 3 * i] = (float)(r[i] * sin(phi));//z
}

__device__
int minimum(float* d, int n)
{
	float min = 10000.0;
	int idx = 0;
	for (int j = 0; j < n; j++)
	{
		if (d[j] < min)
		{
			min = d[j];
			idx = j;
		}
	}
	return idx;
}

__global__
void knn(float* P, int n, float* Q, int m, int* idx, int k, float* d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
	{
		int j;

		float xp = P[0 + i * 3];
		float yp = P[1 + i * 3];
		float zp = P[2 + i * 3];

		float xq, yq, zq;
		for (j = 0; j < m / 2; j++)
		{
			xq = Q[0 + j * 3];
			yq = Q[1 + j * 3];
			zq = Q[2 + j * 3];
			d[j + i * m] = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
		}

		for (j = j; j < m; j++)
		{
			xq = Q[0 + j * 3];
			yq = Q[1 + j * 3];
			zq = Q[2 + j * 3];
			d[j + i * m] = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
		}

		float* arr = d + i * m;
		int* r = idx + i * k;
		for (j = 0; j < k; j++)
		{
			r[j] = minimum(arr, m);//store since the second value
			arr[r[j]] = 10000.0;//make the current distance invalid
		}
	}
}

__global__
void Normals(float* q, int* neighbors, int n, int m, int k, float* bar, float* A_total, float* normals)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
	{
		int j = 0, stride = 0;
		float xq, yq, zq;

		//step 1: find the centroid of the k nearest neighbors
		for (j = 1; j < k + 1; j++)
		{
			stride = neighbors[j + i * (k + 1)];//neighbors are stored row-major
			xq = q[0 + stride * 3];
			yq = q[1 + stride * 3];
			zq = q[2 + stride * 3];
			bar[0 + 3 * i] += (xq / (float)k);//q is stored colum-major (x1y1z1 ...)
			bar[1 + 3 * i] += (yq / (float)k);
			bar[2 + 3 * i] += (zq / (float)k);
		}

		//step 2: find the covariance matrix A (stored row major)
		for (j = 1; j < k + 1; j++)
		{
			stride = neighbors[j + i * (k + 1)];
			xq = q[0 + stride * 3];
			yq = q[1 + stride * 3];
			zq = q[2 + stride * 3];
			//place the values of the upper triangular matrix A only
			A_total[0 + 9 * i] += (xq - bar[0 + 3 * i]) * (xq - bar[0 + 3 * i]);
			A_total[1 + 9 * i] += (xq - bar[0 + 3 * i]) * (yq - bar[1 + 3 * i]);
			A_total[2 + 9 * i] += (xq - bar[0 + 3 * i]) * (zq - bar[2 + 3 * i]);
			A_total[4 + 9 * i] += (yq - bar[1 + 3 * i]) * (yq - bar[1 + 3 * i]);
			A_total[5 + 9 * i] += (yq - bar[1 + 3 * i]) * (zq - bar[2 + 3 * i]);
			A_total[8 + 9 * i] += (zq - bar[2 + 3 * i]) * (zq - bar[2 + 3 * i]);
		}

		float* A = A_total + i * 9;
		//step 3: compute the eigenvectors of A
		float p1 = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
		float qi = 0.0f, p2 = 0.0f, p = 0.0f, r = 0.0f, phi = 0.0f;
		float eigen[3] = {};

		qi = (A[0] + A[4] + A[8]) / 3.0f;//trace(A)
		p2 = (A[0] - qi) * (A[0] - qi) +
			(A[4] - qi) * (A[4] - qi) +
			(A[8] - qi) * (A[8] - qi) + 2 * p1;
		p = (float)sqrt(p2 / 6.0f);
		r = ((float)1 / (2 * p * p * p)) *
			((A[0] - qi) * ((A[4] - qi) * (A[8] - qi) - A[5] * A[5])
				- A[1] * (A[1] * (A[8] - qi) - A[2] * A[5])
				+ A[2] * (A[1] * A[5] - A[2] * (A[4] - qi)));
		if (r <= -1) phi = (float)M_PI / 3.0f;
		else if (r >= 1) phi = 0.0f;
		else  phi = (float)acos(r) / 3.0f;

		//the eigenvalues satisfy eig3 <= eig2 <= eig1
		//eigen[0] = qi + 2 * p * (float)cos(phi);//eigenvalue 1
		eigen[2] = qi + 2 * p * (float)cos(phi + (2 * M_PI / 3));//eigenvalue 3
		//eigen[1] = 3 * qi - eigen[0] - eigen[2];//eigenvalue 2

		float aux, modulo;
		float eigenvector[3] = { 1.0f,1.0f,1.0f };
		//A[3] = A[1];
		//A[6] = A[2];
		//A[7] = A[5];
		//A[0] -= eigen[2];//rx.x
		//if (fabs(A[0]) !=0)
		//{
		//	A[4] -= eigen[2];//ry.y
		//	aux = A[3] / A[0];
		//	A[3] -= A[0] * aux;
		//	A[4] -= A[1] * aux;
		//	A[5] -= A[2] * aux;

		//	eigenvector[1] = -A[5] / A[4];
		//	eigenvector[0] = -(A[1] * eigenvector[1] + A[2] * eigenvector[2]) / A[0];
		//}
		//else
		//{
		//	/*A[8] -= eigen[2];//rz.z
		//	A[4] -= eigen[2];//ry.y
		//	aux = A[3] / A[6];
		//	A[3] -= A[6] * aux;
		//	A[4] -= A[7] * aux;
		//	A[5] -= A[8] * aux;

		//	eigenvector[1] = -A[5] / A[4];
		//	eigenvector[0] = -(A[7] * eigenvector[1] + A[8] * eigenvector[2]) / A[6];*/
		//}
		modulo = sqrt(eigenvector[0] * eigenvector[0] + eigenvector[1] * eigenvector[1] + eigenvector[2] * eigenvector[2]);
		normals[0 + i * 3] = eigenvector[0] / modulo;
		normals[1 + i * 3] = eigenvector[1] / modulo;
		normals[2 + i * 3] = eigenvector[2] / modulo;
	}
}

__global__
void Matching(int n, float* P, float* Q, int q_points, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		float min = 100000;
		float d;

		float xp = P[0 + i * 3];
		float yp = P[1 + i * 3];
		float zp = P[2 + i * 3];

		float xq, yq, zq;
		int j;
		for (j = 0; j < q_points / 2; j++)
		{
			xq = Q[0 + j * 3];
			yq = Q[1 + j * 3];
			zq = Q[2 + j * 3];
			d = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
			if (d < min)
			{
				min = d;
				idx[i] = j;
			}
		}

		for (j = j; j < q_points; j++)
		{
			xq = Q[0 + j * 3];
			yq = Q[1 + j * 3];
			zq = Q[2 + j * 3];
			d = (xp - xq) * (xp - xq) + (yp - yq) * (yp - yq) + (zp - zq) * (zp - zq);
			if (d < min)
			{
				min = d;
				idx[i] = j;
			}
		}
	}
}

__global__
void Q_index(int n, float* Q, int* idx, float* Q_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		Q_idx[0 + i * 3] = Q[0 + idx[i] * 3];
		Q_idx[1 + i * 3] = Q[1 + idx[i] * 3];
		Q_idx[2 + i * 3] = Q[2 + idx[i] * 3];
	}
}

//C has to be stored in column-major order
__global__
void Cxb(int n, float* p, float* q, int* idx, float* normals, float* C_total, float* b_total)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		float nix = normals[0 + idx[i] * 3];//nix
		float niy = normals[1 + idx[i] * 3];//niy
		float niz = normals[2 + idx[i] * 3];//niz

		float xp = p[0 + i * 3];
		float yp = p[1 + i * 3];
		float zp = p[2 + i * 3];

		float xq = q[0 + i * 3];
		float yq = q[1 + i * 3];
		float zq = q[2 + i * 3];

		float cix = yp * niz - zp * niy;//cix
		float ciy = zp * nix - xp * niz;//ciy
		float ciz = xp * niy - yp * nix;//ciz

		C_total[0 + i * 36] = cix * cix; C_total[6 + i * 36] = cix * ciy; C_total[12 + i * 36] = cix * ciz;
		C_total[18 + i * 36] = cix * nix; C_total[24 + i * 36] = cix * niy; C_total[30 + i * 36] = cix * niz;

		C_total[7 + i * 36] = ciy * ciy; C_total[13 + i * 36] = ciy * ciz; C_total[19 + i * 36] = ciy * nix;
		C_total[25 + i * 36] = ciy * niy; C_total[31 + i * 36] = ciy * niz;

		C_total[14 + i * 36] = ciz * ciz; C_total[20 + i * 36] = ciz * nix; C_total[26 + i * 36] = ciz * niy;
		C_total[32 + i * 36] = ciz * niz;

		C_total[21 + i * 36] = nix * nix; C_total[27 + i * 36] = nix * niy; C_total[33 + i * 36] = nix * niz;

		C_total[28 + i * 36] = niy * niy; C_total[34 + i * 36] = niy * niz;

		C_total[35 + i * 36] = niz * niz;

		float aux = (xp - xq) * nix + (yp - yq) * niy + (zp - zq) * niz;

		b_total[0 + i * 6] = -cix * aux; b_total[1 + i * 6] = -ciy * aux; b_total[2 + i * 6] = -ciz * aux;
		b_total[3 + i * 6] = -nix * aux; b_total[4 + i * 6] = -niy * aux; b_total[5 + i * 6] = -niz * aux;
	}
}

__global__
void RyT(int n, float* R, float* T, float* P, float* Q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] + R[0 + 2 * 3] * P[2 + i * 3] + T[0];
		Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] + R[1 + 2 * 3] * P[2 + i * 3] + T[1];
		Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] + R[2 + 2 * 3] * P[2 + i * 3] + T[2];
	}
}

int Read_data(float* d_P, float* d_Q, int num_points);
void printScloud(float* cloud, int num_points, int points2show);
void printSarray(float* array, int points2show);
void printIarray(int* array, int points2show);

int main(void)
{
	int i = 0, j = 0;
	int num_points = 16384;
	int d_points = num_points;
	int m_points = num_points;

	size_t bytesD = (size_t)d_points * (size_t)3 * sizeof(float);
	size_t bytesM = (size_t)m_points * (size_t)3 * sizeof(float);

	int p_points = d_points;
	int q_points = m_points;
	float* d_p, * d_q;
	cudaMalloc(&d_p, bytesD);//p point cloud
	cudaMalloc(&d_q, bytesM);//q point cloud
	cudaError_t err = cudaSuccess;//for checking errors in kernels
	//cuBLAS handle
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	cublasStatus_t cublas_error;
	//cuSolver handle
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	cusolverStatus_t cusolver_error;
	float alpha = 0, beta = 0;//for cublas routines

	/*Block 1: Open, read the file with 64 LiDAR data packets,
	make the conversion from polar to Cartesian coordinates
	and compute the rotation for the model cloud*/
	int valid = Read_data(d_p, d_q, num_points);
	if (valid != 0)
	{
		printf("Error when reading LiDAR data\n");
		return -1;
	}
	/*End of block 1*/

	float* h_D = (float*)malloc(bytesD);
	float* h_M = (float*)malloc(bytesM);
	cudaMemcpy(h_D, d_p, bytesD, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_M, d_q, bytesM, cudaMemcpyDeviceToHost);

	//convert from millimetes to meters
	alpha = 1.0 / 1000.0;
	cublasSscal(cublasH, 3 * d_points, &alpha, d_p, 1);
	cublasSscal(cublasH, 3 * m_points, &alpha, d_q, 1);

	//////////////////////////////////////////2nd:ICP Algorithm//////////////////////////////////

	float cx, cy, cz;
	float sx, sy, sz;

	/////////2nd: Normals estimation/////////
	int GridSize = 60;
	int BlockSize = q_points / GridSize + 1;
	printf("For normals:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	//K-Neighbors
	int k = 4;//number of nearest neighbors
	size_t neighbors_size = (size_t)(k + 1) * (size_t)q_points * sizeof(int);
	int* h_NeighborIds = (int*)malloc(neighbors_size);
	int* d_NeighborIds = NULL;
	float* d_dist = NULL;
	cudaMalloc(&d_NeighborIds, neighbors_size);
	cudaMalloc(&d_dist, (size_t)p_points * (size_t)q_points * sizeof(float));

	//Variables for PCA
	float* h_A = (float*)malloc(9 * q_points * sizeof(float));
	float* d_bar, * d_A;
	cudaMalloc(&d_bar, bytesM);
	cudaMalloc(&d_A, 9 * q_points * sizeof(float));
	cudaMemset(d_bar, 0, bytesM);
	cudaMemset(d_A, 0, 9 * q_points * sizeof(float));

	//Normals
	float* h_normals = (float*)malloc(bytesM);
	float* d_normals = NULL;
	cudaMalloc(&d_normals, bytesM);

	double start_mkl, stop_mkl;

	//start normals estimation
	start_mkl = dsecnd();

	knn << < GridSize, BlockSize >> > (d_q, q_points, d_q, q_points, d_NeighborIds, k + 1, d_dist);
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in knn kernel: %s\n", cudaGetErrorString(err));
	/*cudaMemcpy(h_NeighborIds, d_NeighborIds, neighbors_size, cudaMemcpyDeviceToHost);
	printf("Neighbor IDs:\n");
	for (i = 0; i < p_points; i++)
	{
		printf("%d: ", i + 1);
		for (j = 0; j < k + 1; j++) printf("%d ", h_NeighborIds[j + i * (k + 1)] + 1);
		printf("\n");
	}
	printf("\n");*/
	Normals << < GridSize, BlockSize >> > (d_q, d_NeighborIds, p_points, q_points, k, d_bar, d_A, d_normals);
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in normals kernel: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();

	stop_mkl = dsecnd();

	double normals_time = stop_mkl - start_mkl;
	printf("Normals were calculated in %f ms\n\n", normals_time * 1000);

	float w[3] = {};
	int idx_min = 0;
	cudaMemcpy(h_A, d_A, 9 * q_points * sizeof(float), cudaMemcpyDeviceToHost);
	for (i = 0; i < q_points; i++)
	{
		float* A = h_A + i * 9;
		LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', 3, A, 3, w);
		idx_min = (int)cblas_isamin(3, w, 1);//choose the smallest eigenvalue
		for (j = 0; j < 3; j++) h_normals[j + i * 3] = A[j * 3 + idx_min];
	}
	cudaMemcpy(d_normals, h_normals, bytesM, cudaMemcpyHostToDevice);
	/*printf("Normals:\n");
	for (i = 0; i < q_points; i++)
	{
		printf("%d: ", i + 1);
		for (j = 0; j < 3; j++) printf("%.4f ", h_normals[j + i * 3]);
		printf("\n");
	}*/
	/////////End of 2nd/////////

	free(h_NeighborIds); cudaFree(d_NeighborIds); cudaFree(d_dist);
	free(h_A), cudaFree(d_A), cudaFree(d_bar);
	free(h_normals);

	/////////3rd: ICP algorithm/////////

	//Index vector (used for correspondence)
	int* h_idx = (int*)malloc(p_points * sizeof(int));
	int* d_idx = NULL;//index vector (used for correspondence)
	cudaMalloc(&d_idx, (size_t)p_points * sizeof(int));

	//Q index cloud
	float* h_q_idx = (float*)malloc(bytesD);
	float* d_q_idx = NULL;
	cudaMalloc(&d_q_idx, bytesD);

	//C and b of the system of linear equations
	float* h_C = (float*)malloc(36 * sizeof(float));
	float* h_b = (float*)malloc(6 * sizeof(float));
	float* h_C_total = (float*)malloc(36 * p_points * sizeof(float));
	if (h_C_total != NULL) for (i = 0; i < 36 * p_points; i++) h_C_total[i] = 0;
	float* d_C = NULL, * d_b = NULL;//for the system of linear equations (minimization) C * x = b
	float* d_C_total, * d_b_total;
	cudaMalloc(&d_C, 36 * sizeof(float));
	cudaMalloc(&d_b, 6 * sizeof(float));
	cudaMalloc(&d_C_total, 36 * p_points * sizeof(float));
	cudaMalloc(&d_b_total, 6 * p_points * sizeof(float));
	cudaMemcpy(d_C_total, h_C_total, 36 * p_points * sizeof(float), cudaMemcpyHostToDevice);

	//vector with 1s for suming up coordinates trough matrix-vector multiplications
	float* h_unit = (float*)malloc(p_points * sizeof(float));
	if (h_unit != NULL) for (i = 0; i < p_points; i++) h_unit[i] = 1.0f;
	float* d_unit = NULL;
	cudaMalloc(&d_unit, p_points * sizeof(float));
	cudaMemcpy(d_unit, h_unit, p_points * sizeof(float), cudaMemcpyHostToDevice);

	//For the system of linear (Driver routine)
	int Lwork = 0;
	float* d_work = NULL;
	int* devInfo = NULL;
	cudaMalloc(&devInfo, sizeof(int));

	//Rotation matrix and translation vector
	float* h_temp_r = (float*)malloc(sizeof(float) * 9);
	float* h_temp_T = (float*)malloc(sizeof(float) * 3);
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

	GridSize = 60;
	BlockSize = p_points / GridSize + 1;
	printf("For ICP loop:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	double match_time = 0, minimization_time = 0, transf_time = 0, error_time = 0;

	double ini_time, end_time;

	int iteration = 0;
	//cudaEventRecord(start);
	ini_time = dsecnd();
	while (iteration < MAX_ITER)
	{
		//////////////////Matching step/////////////////
		start_mkl = dsecnd();
		Matching << < GridSize, BlockSize >> > (p_points, d_p, d_q, q_points, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_idx, d_idx, p_points * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index  values[%d]:\n", iteration + 1);
		printIarray(h_idx, p_points);*/
		stop_mkl = dsecnd();
		match_time += stop_mkl - start_mkl;
		/////////////////end of Matching/////////////////

		/////////////////Minimization step (point-to-plane)/////////////////
		start_mkl = dsecnd();

		//get the Q indexed cloud
		Q_index << < GridSize, BlockSize >> > (p_points, d_q, d_idx, d_q_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in Q index kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		Cxb << < GridSize, BlockSize >> > (p_points, d_p, d_q_idx, d_idx, d_normals, d_C_total, d_b_total);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in Cxb kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		//sum up d_C_total and store the result in d_C
		alpha = 1; beta = 0;
		cublas_error = cublasSgemv(cublasH, CUBLAS_OP_N, 36, p_points, &alpha, d_C_total, 36,
			d_unit, 1, &beta, d_C, 1);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - C calculation\n");
			return (-1);
		}

		//sum up d_b_total and store the result in d_b
		cublas_error = cublasSgemv(cublasH, CUBLAS_OP_N, 6, p_points, &alpha, d_b_total, 6,
			d_unit, 1, &beta, d_b, 1);
		if (cublas_error != CUBLAS_STATUS_SUCCESS)
		{
			printf("Error cublas operation - b calculation\n");
			return (-1);
		}

		cudaMemcpy(h_C, d_C, 36 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_b, d_b, 6 * sizeof(float), cudaMemcpyDeviceToHost);

		/*printf("C[%d]:\n", iteration + 1);
		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++) printf("%.3f ", h_C[i + j * 6]);
			printf("\n");
		}
		printf("b[%d]:\n", iteration + 1);
		for (i = 0; i < 6; i++) printf("%.3f\n", h_b[i]);*/

		//Allocate the buffer
		cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, d_C, 6, &Lwork);
		cudaMalloc(&d_work, sizeof(float) * Lwork);//allocate memory for the buffer
		//Find the triangular Cholesky factor
		cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, d_C, 6, d_work, Lwork, devInfo);
		//solve the system of linear equations
		cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, 1, d_C, 6, d_b, 6, devInfo);//d_b holds the answer
		cudaMemcpy(h_b, d_b, 6 * sizeof(float), cudaMemcpyDeviceToHost);//move b to the CPU

		//rotation matrix
		cx = (float)cos(h_b[0]); cy = (float)cos(h_b[1]); cz = (float)cos(h_b[2]);
		sx = (float)sin(h_b[0]); sy = (float)sin(h_b[1]); sz = (float)sin(h_b[2]);
		h_temp_r[0] = cy * cz; h_temp_r[3] = cz * sx * sy - cx * sz;  h_temp_r[6] = cx * cz * sy + sx * sz;
		h_temp_r[1] = cy * sz; h_temp_r[4] = cx * cz + sx * sy * sz; h_temp_r[7] = cx * sy * sz - cz * sx;
		h_temp_r[2] = -sy; h_temp_r[5] = cy * sx; h_temp_r[8] = cx * cy;
		//translation vector
		h_temp_T[0] = h_b[3];
		h_temp_T[1] = h_b[4];
		h_temp_T[2] = h_b[5];

		/*printf("R:\n");
		printScloud(h_temp_r, 3, 3);
		printf("T:\n");
		printSarray(h_temp_T, 3);*/

		cudaMemcpy(d_temp_r, h_temp_r, 9 * sizeof(float), cudaMemcpyHostToDevice);//move temp_r to the GPU
		cudaMemcpy(d_temp_T, h_temp_T, 3 * sizeof(float), cudaMemcpyHostToDevice);//move temp_T to the GPU
		stop_mkl = dsecnd();
		minimization_time += stop_mkl - start_mkl;
		/////////////////end of Minimization/////////////////

		/////////////////Transformation step/////////////////
		start_mkl = dsecnd();
		//D = R * D + T
		RyT << <GridSize, BlockSize >> > (p_points, d_temp_r, d_temp_T, d_p, d_aux);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in RyT kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		cublasScopy(cublasH, 3 * p_points, d_aux, 1, d_p, 1);
		stop_mkl = dsecnd();
		transf_time += stop_mkl - start_mkl;
		/////////////////end of Transformation/////////////////

		/////////////////Error estimation/////////////////
		start_mkl = dsecnd();
		alpha = -1;
		cublasScopy(cublasH, 3 * p_points, d_p, 1, d_aux, 1);
		cublasSaxpy(cublasH, 3 * p_points, &alpha, d_q_idx, 1, d_aux, 1);
		cublasSnrm2(cublasH, 3 * p_points, d_aux, 1, &partial_error);
		h_error[iteration + 1] = partial_error / (float)sqrt(p_points);
		//printf("Current error (%d): %.4f\n", iteration + 1, h_error[iteration + 1]);
		stop_mkl = dsecnd();
		error_time += stop_mkl - start_mkl;
		/////////////////end of Error estimation/////////////////

		if ((h_error[iteration + 1] < 0.000001) ||
			((float)fabs((double)h_error[iteration + 1] - (double)h_error[iteration]) < 0.000001)) break;

		iteration++;
	}
	end_time = dsecnd();

	printf("Error:\n");
	printSarray(h_error, iteration + 1);

	double seconds = end_time - ini_time;
	printf("\nThe ICP algorithm was computed in %.4f ms with %d iterations\n\n",
		1000.0 * (seconds), iteration);

	printf("The matching step represents the %.4f%% of the total time with %.4f ms\n\n",
		match_time * 100.0 / seconds, 1000.0 * match_time);

	printf("The minimization step represents the %.4f%% of the total time with %.4f ms\n\n",
		minimization_time * 100.0 / seconds, 1000.0 * minimization_time);

	printf("The transformation step represents the %.4f%% of the total time with %.4f ms\n\n",
		transf_time * 100.0 / seconds, 1000.0 * transf_time);

	printf("The error estimation step represents the %.4f%% of the total time with %.4f ms\n\n",
		error_time * 100.0 / seconds, 1000.0 * error_time);

	//Destroy handles
	cublasDestroy(cublasH);
	//cusolverDnDestroy(cusolverH);

	//Free memory
	cudaFree(d_normals);

	free(h_D), free(h_M);
	cudaFree(d_p), cudaFree(d_q);

	free(h_idx);
	cudaFree(d_idx);

	free(h_q_idx);
	cudaFree(d_q_idx);

	free(h_C), free(h_b);
	cudaFree(d_C), cudaFree(d_b), cudaFree(d_C_total), cudaFree(d_b_total);

	cudaFree(d_work), cudaFree(devInfo);

	free(h_unit);
	cudaFree(d_unit);

	free(h_temp_r), free(h_temp_T);
	cudaFree(d_temp_r), cudaFree(d_temp_T);

	free(h_error);
	cudaFree(d_aux);

	return 0;
}

int Read_data(float* d_P, float* d_Q, int num_points)
{
	const int N_LINE = 128;
	char line[N_LINE];

	FILE* document;
	document = fopen("Donut_1024x16.csv", "r");
	if (!document) {
		perror("File opening failed");
		return (-1);
	}

	float* h_r = NULL;//radios
	size_t bytes_r = num_points * sizeof(float);
	h_r = (float*)malloc(bytes_r);
	unsigned long int h_encoder_count = 0;//initial encoder counter (then grows with 88 ticks)

	int offset = 0;
	unsigned long int word = 0;

	int channel = 2;
	int azimuth_block = 0;
	int lidar_packet = 0;
	int idx_line;//indice palabra a leer
	int j = 1;//numero de linea
	while (fgets(line, N_LINE, document) != NULL)
	{
		//get the first values of the encoder counter
		if (j == 13) h_encoder_count = atoi(line);
		if (j == 14) h_encoder_count = atoi(line) << 8 | h_encoder_count;

		//read the ranges
		idx_line = 17 + 12 * channel + 788 * azimuth_block + 12608 * lidar_packet;
		if (j == idx_line) word = (unsigned long int) atoi(line);
		if (j == idx_line + 1) word = (unsigned long int) (atoi(line) << 8) | word;
		if (j == idx_line + 2) word = (unsigned long int) ((atoi(line) & 0x0000000F) << 16) | word;

		if (j > (idx_line + 2))//go to next channel
		{
			h_r[offset] = (float)word;
			offset++;
			channel += 4;
		}
		if (channel >= 64)//go to next azimuth block
		{
			channel = 2;
			azimuth_block++;
		}
		if (azimuth_block >= 16)//go to next lidar packet
		{
			azimuth_block = 0;
			lidar_packet++;
		}
		if (lidar_packet >= 64) break;//done
		j++;
	}
	fclose(document);

	document = fopen("beam_intrinsics.csv", "r");
	if (!document) {
		perror("File opening failed");
		return (-1);
	}

	float* h_altitude = NULL;
	float* h_azimuth = NULL;
	size_t bytes_angles = 16 * sizeof(float);//16 channels
	h_altitude = (float*)malloc(bytes_angles);
	h_azimuth = (float*)malloc(bytes_angles);

	j = 1;
	while (fgets(line, N_LINE, document) != NULL)
	{
		//leer altitute angles
		if (j == 2) offset = 0;
		if (j >= 2 && j <= 65)
		{
			if (j % 4 == 0)
			{
				h_altitude[offset] = (float)atof(line);
				offset++;
			}
		}

		//leer azimuth angles
		if (j == 68) offset = 0;
		if (j >= 68 && j <= 131)
		{
			if ((j - 66) % 4 == 0)
			{
				h_azimuth[offset] = (float)atof(line);
				offset++;
			}
		}
		j++;
	}
	fclose(document);

	///////End of Block 1///////

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds1 = 0;//for "Conversion"
	float milliseconds2 = 0;//for "RyT"

	int GridSize = 32;
	int BlockSize = num_points / GridSize;

	cudaError_t err = cudaSuccess;//for checking errors in kernels

	///////Block 2: Conversion to Cartesian coordinates///////

	float* d_r = NULL;
	float* d_azimuth = NULL;
	float* d_altitude = NULL;
	unsigned long int* d_encoder_count;
	cudaMalloc(&d_r, bytes_r);
	cudaMalloc(&d_azimuth, bytes_angles);
	cudaMalloc(&d_altitude, bytes_angles);
	cudaMalloc(&d_encoder_count, sizeof(unsigned long int));

	//move data to GPU
	cudaMemcpy(d_r, h_r, bytes_r, cudaMemcpyHostToDevice);
	cudaMemcpy(d_azimuth, h_azimuth, bytes_angles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_altitude, h_altitude, bytes_angles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_encoder_count, &h_encoder_count, sizeof(unsigned long int), cudaMemcpyHostToDevice);

	//Launch "Conversion" kernel
	cudaEventRecord(start);

	Conversion << <GridSize, BlockSize >> > (d_r, d_encoder_count, d_altitude, d_azimuth, d_P);
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in Conversion kernel: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds1, start, stop);
	printf("Conversion kernel's elapsed time: %.3f ms\n", milliseconds1);

	///////End of Block 2///////

	///////Block 3: Compute the rotation and translation///////

	//rotation matrix and translation vector
	float* h_R = (float*)malloc(9 * sizeof(float));
	float* h_T = (float*)malloc(3 * sizeof(float));
	float* d_R, * d_T;
	cudaMalloc(&d_R, 9 * sizeof(float));
	cudaMalloc(&d_T, 3 * sizeof(float));

	//Translation values
	h_T[0] = 0.001f;//x
	h_T[1] = -0.0202f;//y
	h_T[2] = 0.02f;//z

	//Rotation values (rad)
	float rx = 0.01f;//axis x
	float ry = -0.003f;//axis y
	float rz = 0.05f;//axis z

	float cx = (float)cos(rx); float cy = (float)cos(ry); float cz = (float)cos(rz);
	float sx = (float)sin(rx); float sy = (float)sin(ry); float sz = (float)sin(rz);
	h_R[0] = cy * cz; h_R[1] = (cz * sx * sy) + (cx * sz); h_R[2] = -(cx * cz * sy) + (sx * sz);
	h_R[3] = -cy * sz; h_R[4] = (cx * cz) - (sx * sy * sz); h_R[5] = (cx * sy * sz) + (cz * sx);
	h_R[6] = sy; h_R[7] = -cy * sx; h_R[8] = cx * cy;

	//move data to GPU
	cudaMemcpy(d_R, h_R, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, h_T, 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	RyT << <GridSize, BlockSize >> > (num_points, d_R, d_T, d_P, d_Q);
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in RyT kernel: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds2, start, stop);
	//printf("RyT kernel's elapsed time: %.3f ms\n", milliseconds2);

	//Free variables
	free(h_T), free(h_R);
	cudaFree(d_T), cudaFree(d_R);
	free(h_r), free(h_altitude), free(h_azimuth);
	cudaFree(d_r), cudaFree(d_altitude), cudaFree(d_azimuth), cudaFree(d_encoder_count);

	return 0;
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