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

__global__
void Matching(int n, float* P, float* Q, int q_points, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		float min = 1000000;
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

__global__
void deviation(int n, float* P, float* Q, float* barP, float* barQ, float* devP, float* devQ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		devP[0 + i * 3] = P[0 + i * 3] - barP[0];
		devP[1 + i * 3] = P[1 + i * 3] - barP[1];
		devP[2 + i * 3] = P[2 + i * 3] - barP[2];

		devQ[0 + i * 3] = Q[0 + i * 3] - barQ[0];
		devQ[1 + i * 3] = Q[1 + i * 3] - barQ[1];
		devQ[2 + i * 3] = Q[2 + i * 3] - barQ[2];
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
	int i = 0;
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
	int GridSize = 60;
	int BlockSize = p_points / GridSize + 1;
	printf("Grid Size: %d, Block Size: %d\n", GridSize, BlockSize);

	//MAIN LOOP
	int iteration = 0;
	double match_time = 0, minimization_time = 0, transf_time = 0, error_time = 0;
	double start_mkl, stop_mkl;

	double ini_time, end_time;
	ini_time = dsecnd();
	while (iteration < MAX_ITER)
	{
		/////////////////Matching step/////////////////
		start_mkl = dsecnd();
		Matching << < GridSize, BlockSize >> > (p_points, d_p, d_q, q_points, d_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in matching kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();
		stop_mkl = dsecnd();
		match_time += stop_mkl - start_mkl;
		/*cudaMemcpy(h_idx, d_idx, p_points * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Index  values[%d]:\n", iteration + 1);
		printIarray(h_idx, p_points);*/
		/////////////////end of matching/////////////////

		/////////////////Minimization step/////////////////
		start_mkl = dsecnd();
		//get the Q indexed cloud
		Q_index << < GridSize, BlockSize >> > (p_points, d_q, d_idx, d_q_idx);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in Q index kernel: %s\n", cudaGetErrorString(err));
		cudaDeviceSynchronize();

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
		deviation << <GridSize, BlockSize >> > (p_points, d_p, d_q_idx, d_barP, d_barQ, d_devP, d_devQ);
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
		stop_mkl = dsecnd();
		minimization_time += stop_mkl - start_mkl;
		/////////////////end of minimization/////////////////

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
		/////////////////end of transformation step/////////////////

		/////////////////Error estimation step/////////////////
		start_mkl = dsecnd();
		alpha = -1;
		cublasScopy(cublasH, 3 * p_points, d_p, 1, d_aux, 1);
		cublasSaxpy(cublasH, 3 * p_points, &alpha, d_q_idx, 1, d_aux, 1);
		cublasSnrm2(cublasH, 3 * p_points, d_aux, 1, &partial_error);
		h_error[iteration + 1] = partial_error / (float)sqrt(p_points);
		//printf("Current error (%d): %.4f\n", iteration + 1, h_error[iteration + 1]);
		stop_mkl = dsecnd();
		error_time += stop_mkl - start_mkl;
		/////////////////end of error estimation step/////////////////

		if ((h_error[iteration + 1] < 0.000001) ||
			((float)fabs((double)h_error[iteration + 1] - (double)h_error[iteration]) < 0.000001)) break;
		iteration++;
	}
	end_time = dsecnd();

	printf("Error:\n");
	printSarray(h_error, iteration + 1);

	double seconds = end_time - ini_time;
	printf("\nThe ICP algorithm was computed in %.4f ms with %d iterations\n\n",
		1000.0 * (seconds), iteration + 1);

	printf("The matching step represents the %.4f%% of the total time with %.4f ms\n\n",
		match_time * 100.0 / seconds, 1000.0 * match_time);

	printf("The minimization step represents the %.4f%% of the total time with %.4f ms\n\n",
		minimization_time * 100.0 / seconds, 1000.0 * minimization_time);

	printf("The transformation step represents the %.4f%% of the total time with %.4f ms\n\n",
		transf_time * 100.0 / seconds, 1000.0 * transf_time);

	printf("The error estimation step represents the %.4f%% of the total time with %.4f ms\n\n",
		error_time * 100.0 / seconds, 1000.0 * error_time);

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
	printf("RyT kernel's elapsed time: %.3f ms\n", milliseconds2);

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