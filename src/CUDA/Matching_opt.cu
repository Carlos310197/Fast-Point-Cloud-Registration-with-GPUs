#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "mkl.h"

#define N_MIN 3
#define N_MAX 128

void SmatrixMul(float* A, float* B, float* C, int m, int n, int k);

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

int main()
{
	int WIDTH;
	float XY_max = 2.0f, XY_min = -2.0f;
	float match_time = 0;
	FILE* document;
	fopen_s(&document, "Matching_loop_optimized.csv", "w");
	fprintf(document, "#POINTS,TIME\n");
	printf("#POINTS\tTIME\n");
	for (WIDTH = N_MIN; WIDTH <= N_MAX; WIDTH += 1)
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
		//float miliseconds = 0.0f;

		//////////////////////////////////////////2nd:ICP Algorithm//////////////////////////////////

		//Index vector (used for correspondence)
		int* h_idx = (int*)malloc(p_points * sizeof(int));
		int* d_idx = NULL;
		cudaMalloc(&d_idx, p_points * sizeof(int));

		//Q index cloud
		float* h_q_idx = (float*)malloc(bytesD);
		float* d_q_idx = NULL;
		cudaMalloc(&d_q_idx, bytesD);

		int GridSize = 32;
		int BlockSize = 1024;
		//printf("Grid Size: %d, Block Size: %d\n", GridSize, BlockSize);

		//double start_mkl, stop_mkl;

		/////////////////Matching step/////////////////
		float min_time = 1000.0f;
		for (int i = 0; i < 10; i++)
		{
			cudaEventRecord(start);
			Matching << < GridSize, BlockSize >> > (num_points, d_p, d_q, q_points, d_idx);
			err = cudaGetLastError();
			if (err != cudaSuccess) printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
			cudaDeviceSynchronize();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			//match_time = (stop_mkl - start_mkl) * 1000;//time in ms
			cudaEventElapsedTime(&match_time, start, stop);
			if (match_time < min_time) min_time = match_time;
		}
		/////////////////End of Matching/////////////////

		fprintf(document, "%d,%f\n", num_points, min_time);
		printf("%d\t%f\n", num_points, min_time);

		//Free memory
		free(h_D), free(h_M);
		cudaFree(d_p), cudaFree(d_q);

		free(h_idx), cudaFree(d_idx);

		free(h_q_idx), cudaFree(d_q_idx);
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