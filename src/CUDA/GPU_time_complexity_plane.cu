//For debugging:
//nvcc ICP_standard.cu -lcublas -lcurand -lcusolver -o ICP_cuda

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include "mkl.h"
#include "mkl_lapacke.h"
//#include <device_functions.h>

//constants
//#define WIDTH 128
//#define XY_min -2.0
//#define XY_max 2.0
#define MAX_ITER 1

void SmatrixMul(float* A, float* B, float* C, int m, int n, int k);
void printScloud(float* cloud, int num_points, int points2show);
void printSarray(float* array, int points2show);
void printIarray(int* array, int points2show);
void initialize_array(float* array, int n);

__global__
void Matching(int n, float* P, float* Q, int q_points, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		float min = 100000;
		float d;

		for (int j = 0; j < q_points; j++)
		{
			d = (P[0 + i * 3] - Q[0 + j * 3]) * (P[0 + i * 3] - Q[0 + j * 3]) +
				(P[1 + i * 3] - Q[1 + j * 3]) * (P[1 + i * 3] - Q[1 + j * 3]) +
				(P[2 + i * 3] - Q[2 + j * 3]) * (P[2 + i * 3] - Q[2 + j * 3]);
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
void Cxb(int n, float* p, float* q, int* idx, float* normals, float* cn, float* C_total, float* b_total)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		float cix, ciy, ciz;
		float nix, niy, niz;

		cix = p[1 + i * 3] * normals[2 + idx[i] * 3] -
			p[2 + i * 3] * normals[1 + idx[i] * 3];//cix
		ciy = p[2 + i * 3] * normals[0 + idx[i] * 3] -
			p[0 + i * 3] * normals[2 + idx[i] * 3];//ciy
		ciz = p[0 + i * 3] * normals[1 + idx[i] * 3] -
			p[1 + i * 3] * normals[0 + idx[i] * 3];//ciz

		nix = normals[0 + idx[i] * 3];//nix
		niy = normals[1 + idx[i] * 3];//niy
		niz = normals[2 + idx[i] * 3];//niz

		C_total[0 + i * 36] = cix * cix; C_total[6 + i * 36] = cix * ciy; C_total[12 + i * 36] = cix * ciz;
		C_total[18 + i * 36] = cix * nix; C_total[24 + i * 36] = cix * niy; C_total[30 + i * 36] = cix * niz;

		C_total[7 + i * 36] = ciy * ciy; C_total[13 + i * 36] = ciy * ciz; C_total[19 + i * 36] = ciy * nix;
		C_total[25 + i * 36] = ciy * niy; C_total[31 + i * 36] = ciy * niz;

		C_total[14 + i * 36] = ciz * ciz; C_total[20 + i * 36] = ciz * nix; C_total[26 + i * 36] = ciz * niy;
		C_total[32 + i * 36] = ciz * niz;

		C_total[21 + i * 36] = nix * nix; C_total[27 + i * 36] = nix * niy; C_total[33 + i * 36] = nix * niz;

		C_total[28 + i * 36] = niy * niy; C_total[34 + i * 36] = niy * niz;

		C_total[35 + i * 36] = niz * niz;

		float aux = (p[0 + i * 3] - q[0 + i * 3]) * nix + (p[1 + i * 3] - q[1 + i * 3]) * niy + (p[2 + i * 3] - q[2 + i * 3]) * niz;

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

int main()
{
	int WIDTH;
	float XY_max = 2.0f, XY_min = -2.0f;
	FILE* document;
	fopen_s(&document, "GPU_ICP_point_to_plane_TimeComp.csv", "w");
	fprintf(document, "NUM_POINTS,TIME\n");
	for (WIDTH = 4; WIDTH <= 128; WIDTH += 1)
	{
		int num_points, i, j, k;
		float ti[3], ri[3];
		float* lin_space = (float*)malloc(WIDTH * sizeof(float));
		float lenght;

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

		/////////End of 1st/////////

		//since this lines 
		//p assumes the value of D
		//q assumes the value of M
		//number of p and q points
		int p_points = d_points;
		int q_points = m_points;
		float* d_p, * d_q;
		cudaMalloc(&d_p, bytesD);//p points cloud
		cudaMalloc(&d_q, bytesM);//p points cloud//q point cloud
		//transfer data from D and M to p and q
		cudaMemcpy(d_p, h_D, bytesD, cudaMemcpyHostToDevice);//copy data cloud to p
		cudaMemcpy(d_q, h_M, bytesM, cudaMemcpyHostToDevice);//copy model cloud to q
		cudaError_t err = cudaSuccess;//for checking errors in kernels
		//for measuring time
		float milliseconds = 0.0f;
		//cuBLAS handle
		cublasHandle_t cublasH;
		cublasCreate(&cublasH);
		cublasStatus_t cublas_error;
		//cuSolver handle
		cusolverDnHandle_t cusolverH;
		cusolverDnCreate(&cusolverH);
		cusolverStatus_t cusolver_error;

		/////////2nd: Normals estimation/////////
		int GridSize = 16;
		int BlockSize = 1024;
		//printf("For normals:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

		int count = 0, idx_min = 0;
		float* q1 = (float*)malloc(bytesM);
		float* dist = (float*)malloc(q_points * sizeof(float));
		k = 4;//using 4 nearest neighbors
		int* neighborIds = (int*)malloc((size_t)k * (size_t)q_points * sizeof(int));//for each point of q there are 4 points here

		float* normals = (float*)malloc(bytesM);//3 coordinates per normal
		float* d_normals = NULL;
		cudaMalloc(&d_normals, bytesM);
		float* bar = (float*)malloc(3 * sizeof(float));
		float A[9] = {};//covariance matrix (row-major)
		int stride = 0;
		float a = 0.0f;
		MKL_INT info;
		float w[3] = {};
		float xi = 0.0f, yi = 0.0f, zi = 0.0f;

		double start = 0.0, end = 0.0;
		start = dsecnd();
		//Find k nearest neighbors of each Q point
		//printf("distances:\n");
		for (i = 0; i < q_points; i++)
		{
			//bruteforce
			for (count = 0; count < 3; count++)
			{
				cblas_scopy(q_points, &h_M[i + q_points * count], 0, &q1[q_points * count], 1);//copy vector in p
			}
			vsSub(3 * q_points, h_M, q1, q1);
			vsSqr(3 * q_points, q1, q1);//q1=q1^2
			//sum of all the calculated distances
			vsAdd(q_points, q1, &q1[1 * q_points], dist);
			vsAdd(q_points, dist, &q1[2 * q_points], dist);
			for (j = 0; j < k + 1; j++)//up to k+1 because neighbors are only useful since the 2nd point
			{
				idx_min = (int)cblas_isamin(q_points, dist, 1);
				if (j > 0) neighborIds[(j - 1) + i * 4] = idx_min;//store since the second value
				dist[idx_min] = 10000.0;//make the current distance invalid
			}
		}
		/*printf("Neighbor IDs:\n");
		for (i = 0; i < q_points; i++)
		{
			printf("%d: ", i + 1);
			for (j = 0; j < k; j++) printf("%d ", neighborIds[j + i * k]+1);
			printf("\n");
		}*/

		//Find normals using PCA
		//printf("bar:\n");
		for (i = 0; i < q_points; i++)
		{
			//error may be in here
			//get the centroid
			bar[0] = 0.0f; bar[1] = 0.0f; bar[2] = 0.0f;
			for (j = 0; j < k; j++)
			{
				stride = neighborIds[j + i * k];
				bar[0] += h_M[stride + 0 * q_points];
				bar[1] += h_M[stride + 1 * q_points];
				bar[2] += h_M[stride + 2 * q_points];
			}
			a = 1 / (float)k;
			cblas_sscal(3, a, bar, 1);
			/*printf("%d: ", i + 1);
			for (j = 0; j < 3; j++) printf("%.4f ", bar[j]);
			printf("\n");*/

			//get the covariance matrix
			initialize_array(A, 9);
			for (j = 0; j < k; j++)
			{
				stride = neighborIds[j + i * k];
				xi = h_M[stride + 0 * q_points];
				yi = h_M[stride + 1 * q_points];
				zi = h_M[stride + 2 * q_points];
				//place the values of the upper triangular matrix A only
				A[0] += (xi - bar[0]) * (xi - bar[0]);
				A[1] += (xi - bar[0]) * (yi - bar[1]);
				A[2] += (xi - bar[0]) * (zi - bar[2]);
				A[4] += (yi - bar[1]) * (yi - bar[1]);
				A[5] += (yi - bar[1]) * (zi - bar[2]);
				A[8] += (zi - bar[2]) * (zi - bar[2]);
			}
			//a = 1 / (float)q_points;
			//cblas_sscal(9, a, A, 1);

			/*printf("\nA(%d):\n", i + 1);
			for (int r = 0; r < 3; r++)
			{
				for (j = 0; j < 3; j++) printf("%.4f ", A[j + r * 3]);
				printf("\n");
			}*/

			//get eigenvalues
			info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', 3, A, 3, w);
			if (info > 0) {
				printf("The algorithm failed to compute eigenvalues.\n");
				exit(1);
			}
			/*printf("\nEigenvector(%d):\n", i + 1);
			for (int r = 0; r < 3; r++)
			{
				for (j = 0; j < 3; j++) printf("%.4f ", A[j + r * 3]);
				printf("\n");
			}*/
			//since 'V' was passed as an argument
			//eigenvectors are calculated and stored in A
			//store only the third eigenvector of A (the smallest one)
			idx_min = (int)cblas_isamin(3, w, 1);//choose the smallest eigenvalue
			//printf("Eigen %d: %d\n", i + 1, idx_min);
			for (j = 0; j < 3; j++) normals[i + j * q_points] = A[j * 3 + idx_min];//store values in normals (using the first type of grouping)
		}
		end = dsecnd();
		//printf("Normals were calculated in %f ms\n\n", 1000.0 * (end - start));
		cudaMemcpy(d_normals, normals, bytesM, cudaMemcpyHostToDevice);
		/*printf("Normals:\n");
		for (i = 0; i < q_points; i++)
		{
			printf("%d: ", i + 1);
			for (j = 0; j < 3; j++) printf("%.4f ", h_normals[j + i * 3]);
			printf("\n");
		}*/
		/////////End of 2nd/////////

		free(neighborIds);
		free(bar);
		free(normals);

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
		float* d_cn, * d_C_total, * d_b_total;
		cudaMalloc(&d_C, 36 * sizeof(float));
		cudaMalloc(&d_b, 6 * sizeof(float));
		cudaMalloc(&d_cn, 6 * p_points * sizeof(float));
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

		float alpha = 0, beta = 0;//for cublas routines

		GridSize = 16;
		BlockSize = 1024;
		//printf("For ICP loop:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

		int iteration = 0;
		double ini_time, end_time;
		ini_time = dsecnd();
		while (iteration < MAX_ITER)
		{
			//////////////////Matching step/////////////////

			Matching << < GridSize, BlockSize >> > (num_points, d_p, d_q, q_points, d_idx);
			err = cudaGetLastError();
			if (err != cudaSuccess) printf("Error in matching kernel[%d]: %s\n",num_points,cudaGetErrorString(err));
			cudaDeviceSynchronize();
			/*cudaMemcpy(h_idx, d_idx, p_points * sizeof(int), cudaMemcpyDeviceToHost);
			printf("Index  values[%d]:\n", iteration + 1);
			printIarray(h_idx, p_points);*/

			/////////////////end of Matching/////////////////

			//get the Q indexed cloud
			Q_index << < GridSize, BlockSize >> > (num_points, d_q, d_idx, d_q_idx);
			err = cudaGetLastError();
			if (err != cudaSuccess) printf("Error in Q index kernel: %s\n", cudaGetErrorString(err));
			cudaDeviceSynchronize();

			/////////////////Minimization step (point-to-plane)/////////////////

			Cxb << < GridSize, BlockSize >> > (num_points, d_p, d_q_idx, d_idx, d_normals, d_cn, d_C_total, d_b_total);
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

			/////////////////end of Minimization/////////////////

			/////////////////Transformation step/////////////////

			//D = R * D + T
			RyT << <GridSize, BlockSize >> > (num_points, d_temp_r, d_temp_T, d_p, d_aux);
			err = cudaGetLastError();
			if (err != cudaSuccess) printf("Error in RyT kernel: %s\n", cudaGetErrorString(err));
			cudaDeviceSynchronize();
			cublasScopy(cublasH, 3 * p_points, d_aux, 1, d_p, 1);

			/////////////////end of Transformation/////////////////

			/////////////////Error estimation/////////////////

			alpha = -1;
			cublasScopy(cublasH, 3 * p_points, d_p, 1, d_aux, 1);
			cublasSaxpy(cublasH, 3 * p_points, &alpha, d_q_idx, 1, d_aux, 1);
			cublasSnrm2(cublasH, 3 * p_points, d_aux, 1, &partial_error);
			h_error[iteration + 1] = partial_error / (float)sqrt(p_points);
			//printf("Current error (%d): %.4f\n", iteration + 1, h_error[iteration + 1]);

			/////////////////end of Error estimation/////////////////

			if ((h_error[iteration + 1] < 0.000001) ||
				((float)fabs((double)h_error[iteration + 1] - (double)h_error[iteration]) < 0.000001)) break;

			iteration++;
		}
		end_time = dsecnd();
		double seconds = end_time - ini_time;
		fprintf(document, "%d,%.4f\n", num_points, 1000 * seconds);

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
		cudaFree(d_C), cudaFree(d_b), cudaFree(d_C_total), cudaFree(d_b_total), cudaFree(d_cn);

		cudaFree(d_work), cudaFree(devInfo);

		free(h_unit);
		cudaFree(d_unit);

		free(h_temp_r), free(h_temp_T);
		cudaFree(d_temp_r), cudaFree(d_temp_T);

		free(h_error);
		cudaFree(d_aux);
	}
	fclose(document);
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
	for (int i = 0; i < points2show; i++)
	{
		printf("%d: %d\n", i + 1, array[i]);
	}
	printf("\n");
}

void initialize_array(float* array, int n)
{
	for (int i = 0; i < n; i++) array[i] = 0.0f;
}
