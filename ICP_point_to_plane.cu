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
#define WIDTH 32
#define XY_min -2.0
#define XY_max 2.0
#define MAX_ITER 1

void SmatrixMul(float* A, float* B, float* C, int m, int n, int k);
void printScloud(float* cloud, int num_points, int points2show);
void printSarray(float* array, int points2show);
void printIarray(int* array, int points2show);

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

//idx has to allocate mxn values
//d has to allocate mxn values
__global__
void knn(float* P, int n, float* Q, int m, int* idx, int k, float* d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;

	for (j = 0; j < m; j++)
		d[j + i * m] = (float)sqrt((P[0 + i * 3] - Q[0 + j * 3]) * (P[0 + i * 3] - Q[0 + j * 3]) +
									(P[1 + i * 3] - Q[1 + j * 3]) * (P[1 + i * 3] - Q[1 + j * 3]) +
									(P[2 + i * 3] - Q[2 + j * 3]) * (P[2 + i * 3] - Q[2 + j * 3]));

	__syncthreads();

	float* arr = d + i * m;
	int* r = idx + i * k;

	//instead lets just make comparisons
	for (j = 0; j < k; j++)//up to k+1 because neighbors are only useful since the 2nd point
	{
		r[j] = minimum(arr, m);//store since the second value
		arr[r[j]] = 10000.0;//make the current distance invalid
	}
}

__global__
void Normals(float* q, int* neighbors, int n, int m, int k, float* bar, float* A_total, float* normals)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = 0, stride = 0;

	//printf("%d\n", i);
	//step 1: find the centroid of the k nearest neighbors
	for (j = 1; j < k + 1; j++)
	{
		stride = neighbors[j + i * (k + 1)];//neighbors are stored row-major
		bar[0 + 3 * i] += (q[0 + stride * 3] / (float)k);//q is stored colum-major (x1y1z1 ...)
		bar[1 + 3 * i] += (q[1 + stride * 3] / (float)k);
		bar[2 + 3 * i] += (q[2 + stride * 3] / (float)k);
	}
	//for (j = 0; j < 3; j++) printf("bar%d[%d]: %.4f\n", j, i, bar[j + i * 3]);
	__syncthreads();

	//step 2: find the covariance matrix A (stored row major)
	for (j = 1; j < k + 1; j++)
	{
		stride = neighbors[j + i * (k + 1)];
		//place the values of the upper triangular matrix A only
		A_total[0 + 9 * i] += (q[0 + stride * 3] - bar[0 + 3 * i]) * (q[0 + stride * 3] - bar[0 + 3 * i]);
		A_total[1 + 9 * i] += (q[0 + stride * 3] - bar[0 + 3 * i]) * (q[1 + stride * 3] - bar[1 + 3 * i]);
		A_total[2 + 9 * i] += (q[0 + stride * 3] - bar[0 + 3 * i]) * (q[2 + stride * 3] - bar[2 + 3 * i]);
		A_total[4 + 9 * i] += (q[1 + stride * 3] - bar[1 + 3 * i]) * (q[1 + stride * 3] - bar[1 + 3 * i]);
		A_total[5 + 9 * i] += (q[1 + stride * 3] - bar[1 + 3 * i]) * (q[2 + stride * 3] - bar[2 + 3 * i]);
		A_total[8 + 9 * i] += (q[2 + stride * 3] - bar[2 + 3 * i]) * (q[2 + stride * 3] - bar[2 + 3 * i]);
	}
	__syncthreads();

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

//C has to be stored in column-major order
__global__
void Cxb(float* p, float* q, int* idx, float* normals, float* cn, float* C_total, float* b_total)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = idx[i];
	cn[0 + i * 6] = p[1 + i * 3] * normals[2 + idx[i] * 3] -
		p[2 + i * 3] * normals[1 + idx[i] * 3];//cix
	cn[1 + i * 6] = p[2 + i * 3] * normals[0 + idx[i] * 3] -
		p[0 + i * 3] * normals[2 + idx[i] * 3];//ciy
	cn[2 + i * 6] = p[0 + i * 3] * normals[1 + idx[i] * 3] -
		p[1 + i * 3] * normals[0 + idx[i] * 3];//ciz
	cn[3 + i * 6] = normals[0 + idx[i] * 3];//nix
	cn[4 + i * 6] = normals[1 + idx[i] * 3];//niy
	cn[5 + i * 6] = normals[2 + idx[i] * 3];//niz

	C_total[0 + i * 36] = cn[0 + i * 6] * cn[0 + i * 6]; C_total[6 + i * 36] = cn[0 + i * 6] * cn[1 + i * 6]; C_total[12 + i * 36] = cn[0 + i * 6] * cn[2 + i * 6];
	C_total[18 + i * 36] = cn[0 + i * 6] * cn[3 + i * 6]; C_total[24 + i * 36] = cn[0 + i * 6] * cn[4 + i * 6]; C_total[30 + i * 36] = cn[0 + i * 6] * cn[5 + i * 6];

	C_total[7 + i * 36] = cn[1 + i * 6] * cn[1 + i * 6]; C_total[13 + i * 36] = cn[1 + i * 6] * cn[2 + i * 6]; C_total[19 + i * 36] = cn[1 + i * 6] * cn[3 + i * 6];
	C_total[25 + i * 36] = cn[1 + i * 6] * cn[4 + i * 6]; C_total[31 + i * 36] = cn[1 + i * 6] * cn[5 + i * 6];

	C_total[14 + i * 36] = cn[2 + i * 6] * cn[2 + i * 6]; C_total[20 + i * 36] = cn[2 + i * 6] * cn[3 + i * 6]; C_total[26 + i * 36] = cn[2 + i * 6] * cn[4 + i * 6];
	C_total[32 + i * 36] = cn[2 + i * 6] * cn[5 + i * 6];

	C_total[21 + i * 36] = cn[3 + i * 6] * cn[3 + i * 6]; C_total[27 + i * 36] = cn[3 + i * 6] * cn[4 + i * 6]; C_total[33 + i * 36] = cn[3 + i * 6] * cn[5 + i * 6];

	C_total[28 + i * 36] = cn[4 + i * 6] * cn[4 + i * 6]; C_total[34 + i * 36] = cn[4 + i * 6] * cn[5 + i * 6];

	C_total[35 + i * 36] = cn[5 + i * 6] * cn[5 + i * 6];

	float aux = (p[0 + i * 3] - q[0 + i * 3]) * cn[3 + i * 6] +
		(p[1 + i * 3] - q[1 + i * 3]) * cn[4 + i * 6] +
		(p[2 + i * 3] - q[2 + i * 3]) * cn[5 + i * 6];

	b_total[0 + i * 6] = -cn[0 + i * 6] * aux; b_total[1 + i * 6] = -cn[1 + i * 6] * aux; b_total[2 + i * 6] = -cn[2 + i * 6] * aux;
	b_total[3 + i * 6] = -cn[3 + i * 6] * aux; b_total[4 + i * 6] = -cn[4 + i * 6] * aux; b_total[5 + i * 6] = -cn[5 + i * 6] * aux;
}

__global__
void RyT(float* R, float* T, float* P, float* Q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] + R[0 + 2 * 3] * P[2 + i * 3] + T[0];
	Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] + R[1 + 2 * 3] * P[2 + i * 3] + T[1];
	Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] + R[2 + 2 * 3] * P[2 + i * 3] + T[2];
}

int main(void)
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
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
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
	int GridSize = 8;
	int BlockSize = q_points / GridSize;
	printf("For normals:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	//K-Neighbors
	k = 4;//number of nearest neighbors
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

	cudaEventRecord(start);//start normals estimation
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

	printf("\n");
	cudaEventRecord(stop);//end normals estimation
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Normals were calculated in %f ms\n\n", milliseconds);

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
	printf("Normals:\n");
	for (i = 0; i < q_points; i++)
	{
		printf("%d: ", i + 1);
		for (j = 0; j < 3; j++) printf("%.4f ", h_normals[j + i * 3]);
		printf("\n");
	}
	/////////End of 2nd/////////

	free(h_NeighborIds), cudaFree(d_NeighborIds), cudaFree(d_dist);
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

	GridSize = 8;
	BlockSize = p_points / GridSize;
	printf("For ICP loop:\nGrid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	int iteration = 0;
	cudaEventRecord(start);
	while (iteration < MAX_ITER)
	{
		//////////////////Matching step/////////////////

		Matching << < GridSize, BlockSize >> > (d_p, d_q, q_points, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		/*cudaMemcpy(h_idx, d_idx, p_points * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index  values[%d]:\n", iteration + 1);
		printIarray(h_idx, p_points);*/

		/////////////////end of Matching/////////////////

		//get the Q indexed cloud
		Q_index << < GridSize, BlockSize >> > (d_q, d_idx, d_q_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in Q index kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

		/////////////////Minimization step (point-to-plane)/////////////////

		Cxb << < GridSize, BlockSize >> > (d_p, d_q_idx, d_idx, d_normals, d_cn, d_C_total, d_b_total);
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

		printf("C[%d]:\n", iteration + 1);
		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++) printf("%.3f ", h_C[i + j * 6]);
			printf("\n");
		}
		printf("b[%d]:\n", iteration + 1);
		for (i = 0; i < 6; i++) printf("%.3f\n", h_b[i]);

		//Allocate the buffer
		cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, d_C, 6, &Lwork);
		cudaMalloc(&d_work, sizeof(float) * Lwork);//allocate memory for the buffer
		//Find the triangular Cholesky factor
		cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, d_C, 6, d_work, Lwork, devInfo);
		//solve the system of linear equations
		cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_UPPER, 6, 1, d_C, 6, d_b, 1, devInfo);//d_b holds the answer
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

		cudaMemcpy(d_temp_r, h_temp_r, 9 * sizeof(float), cudaMemcpyHostToDevice);//move temp_r to the GPU
		cudaMemcpy(d_temp_T, h_temp_T, 3 * sizeof(float), cudaMemcpyHostToDevice);//move temp_T to the GPU

		/////////////////end of Minimization/////////////////

		/////////////////Transformation step/////////////////

		//D = R * D + T
		RyT << <GridSize, BlockSize >> > (d_temp_r, d_temp_T, d_p, d_aux);
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
		printf("Current error (%d): %.4f\n", iteration + 1, h_error[iteration + 1]);

		/////////////////end of Error estimation/////////////////

		if ((h_error[iteration + 1] < 0.000001) ||
			((float)fabs((double)h_error[iteration + 1] - (double)h_error[iteration]) < 0.000001)) break;

		iteration++;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Error:\n");
	printSarray(h_error, iteration + 1);

	printf("ICP converged successfully!\n\n");

	printf("Elapsed time: %f ms\n", milliseconds);

	//Destroy handles
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	//Free memory
	free(h_normals), cudaFree(d_normals);

	free(h_D), free(h_M);
	cudaFree(d_p), cudaFree(d_q);

	free(h_idx), cudaFree(d_idx);

	free(h_q_idx), cudaFree(d_q_idx);

	free(h_C), free(h_b);
	cudaFree(d_C), cudaFree(d_b), cudaFree(d_C_total), cudaFree(d_b_total), cudaFree(d_cn);

	cudaFree(d_work), cudaFree(devInfo);

	free(h_unit), cudaFree(d_unit);

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
