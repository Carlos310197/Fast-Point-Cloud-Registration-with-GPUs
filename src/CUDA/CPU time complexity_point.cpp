//ICP algorithm using the point-to-point error metric and synthetic data
//CPU version using MKL library
//By: Carlos Huapaya

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mkl.h"
#include "mkl_lapacke.h"

//constants
//#define WIDTH 128//width of grid
//#define XY_min -2.0f
//#define XY_max 2.0f
#define MAX_ITER 1

void repmat(float t[3], int repeat, float* matrix);
void print_cloud(float* cloud, int num_points, int points_show);
void print_array(float* array, int size);
void initialize_array(float* array, int n);
void centroid_deviation(int n, float* p, float* q, int* q_idx, float* p_bar, float* p_dev, float* q_bar, float* q_dev);

int main(void)
{
	int WIDTH;
	float XY_max = 2.0f, XY_min = -2.0f;
	FILE* document;
	fopen_s(&document, "CPU_Parallel_ICP_point_to_point_TimeComp.csv", "w");
	fprintf(document, "NUM_POINTS,TIME\n");
	for (WIDTH = 3; WIDTH <= 128; WIDTH += 1)
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
			lin_space[i] = (float)XY_min + ((float)i * (float)lenght) / ((float)n - 1.0f);
		}

		//create the meshgrid
		float* mesh_x = (float*)malloc(num_points * sizeof(float));
		float* mesh_y = (float*)malloc(num_points * sizeof(float));

		//Data is stored using the first type of grouping
		//xxxxxxxxxxx yyyyyyyyyyyy zzzzzzzzzzz
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
		else return 1;

		//Create the function z = f(x,y) = x^2-y^2
		float* z = (float*)malloc(num_points * sizeof(float));
		for (i = 0; i < num_points; i++) z[i] = (float)pow(mesh_x[i], 2) - (float)pow(mesh_y[i], 2);

		//Create data point cloud matrix
		size_t bytesD = (size_t)d_points * (size_t)3 * sizeof(float);
		float* D = (float*)malloc(bytesD);
		initialize_array(D, 3 * d_points);//Initialize the data point cloud D

		j = 0;
		for (i = 0; i < 3; i++)
		{
			k = 0;
			while (k < d_points)
			{
				if (i == 0) D[j] = mesh_x[k];
				if (i == 1) D[j] = mesh_y[k];
				if (i == 2) D[j] = z[k];
				j++;
				k++;
			}
		}

		//printf("Data point cloud\n");
		//print_cloud(D, d_points, 10);

		//Translation values
		ti[0] = 0.8f;//x
		ti[1] = -0.3f;//y
		ti[2] = 0.2f;//z

		//Rotation values
		ri[0] = 0.2f;//axis x
		ri[1] = -0.2f;//axis y
		ri[2] = 0.05f;//axis z

		float rx[3][3] = {
			{ 1.0f, 0.0f, 0.0f },
			{ 0.0f, (float)cos(ri[0]), -(float)sin(ri[0]) },
			{ 0.0f, (float)sin(ri[0]), (float)cos(ri[0]) },
		};

		float ry[3][3] = {
			{ (float)cos(ri[1]), 0.0f, (float)sin(ri[1]) },
			{ 0.0f, 1.0f, 0.0f },
			{ -(float)sin(ri[1]), 0.0f, (float)cos(ri[1]) },
		};

		float rz[3][3] = {
			{ (float)cos(ri[2]), -(float)sin(ri[2]),0.0f },
			{ (float)sin(ri[2]), (float)cos(ri[2]), 0.0f },
			{ 0.0f, 0.0f, 1.0f },
		};

		//rotation matrix
		float r[3][3] = {};

		//matrix multiplication
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, *rx, 3, *ry, 3, 0, *r, 3);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, *r, 3, *rz, 3, 0, *r, 3);

		size_t bytesM = (size_t)m_points * (size_t)3 * sizeof(float);
		float* M = (float*)malloc(bytesM);
		initialize_array(M, 3 * m_points);//Initialize the model point cloud M

		repmat(ti, m_points, M);
		//Construct the model point cloud M = r * D + M
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			3, num_points, 3, 1, *r, 3, D, num_points,
			1, M, num_points);
		//To add noise to M cloud
		/*srand(time(0));
		for (i = 0; i < 3 * m_points; i++) M[i] = M[i] * 0.001 * rand();*/

		//printf("Model point cloud\n");
		//print_cloud(M, num_points, 10);

		////////////////End of 1st//////////////

		////////////////2nd:ICP algorithm//////////////

		int num_iterations = 0, iteration = 0;
		int count, idx_dmin;
		//MKL_INT info;
		int p_points = d_points;
		int q_points = m_points;

		float* p = (float*)malloc(bytesD);//p points cloud
		float* q = (float*)malloc(bytesM);//q point cloud

		//transfer data from D and M to p and q
		cblas_scopy(3 * d_points, D, 1, p, 1);
		cblas_scopy(3 * m_points, M, 1, q, 1);

		float* q_bar = (float*)malloc(3 * sizeof(float));
		float* p_bar = (float*)malloc(3 * sizeof(float));
		float* q_dev = (float*)malloc(bytesD);
		float* p_dev = (float*)malloc(bytesD);
		float* N = (float*)malloc(3 * 3 * sizeof(float));
		float* F = (float*)malloc(bytesD);
		float* G = (float*)malloc(3 * sizeof(float));
		float* match = (float*)malloc(p_points * sizeof(float));//indexing vector
		float* min_dist = (float*)malloc(p_points * sizeof(float));//minimum distance vector
		int* q_idx = (int*)malloc(p_points * sizeof(int));
		float* E = (float*)malloc(size_t(MAX_ITER + 1) * sizeof(float));//ERROR
		float* superb = (float*)malloc(2 * sizeof(float));
		float* U = (float*)malloc(3 * 3 * sizeof(float));
		float* Vt = (float*)malloc(3 * 3 * sizeof(float));
		float* S = (float*)malloc(3 * 3 * sizeof(float));
		float* p1 = (float*)malloc(bytesD);
		float* dist = (float*)malloc(p_points * sizeof(float));
		float* temp_t = (float*)malloc(3 * sizeof(float));
		float* temp_r = (float*)malloc(9 * sizeof(float));
		initialize_array(E, int(MAX_ITER) + 1);

		double ini_time, end_time;
		double seconds=0;
		double min_time = 100000.0;
		for (int repeat = 0; repeat < 10; repeat++)
		{
			//ICP MAIN LOOP
			iteration = 0;
			ini_time = dsecnd();
			while (iteration < MAX_ITER)
			{
				//Matching step (brute force)
				for (j = 0; j < p_points; j++)
				{
					//bruteforce
					for (count = 0; count < 3; count++)
					{
						cblas_scopy(p_points, &p[j + p_points * count], 0, &p1[p_points * count], 1);//copy vector in p
					}
					vsSub(3 * q_points, q, p1, p1);
					vsSqr(3 * q_points, p1, p1);//p1=p1^2
					//sum of all the calculated distances
					vsAdd(q_points, p1, &p1[1 * q_points], dist);
					vsAdd(q_points, dist, &p1[2 * q_points], dist);
					idx_dmin = (int)cblas_isamin(q_points, dist, 1);
					q_idx[j] = idx_dmin;
				}

				//Minimization step (point-to-point)
				centroid_deviation(p_points, p, q, q_idx, p_bar, p_dev, q_bar, q_dev);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					3, 3, p_points, 1, q_dev, p_points, p_dev, p_points, 0, N, 3);
				LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', 3, 3, N, 3, S, U, 3, Vt, 3, superb);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					3, 3, 3, 1, U, 3, Vt, 3, 0, temp_r, 3);//temp_r = U * Vt
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					3, 1, 3, 1, temp_r, 3, p_bar, 3, 0, G, 1);//G = temp_r*p_bar
				vsSub(3, q_bar, G, temp_t);//temp_t = q_bar - G

				//Transformation step
				repmat(temp_t, p_points, F);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					3, p_points, 3, 1, temp_r, 3, p, p_points, 1, F, p_points);//F = temp_r * p + F
				cblas_scopy(3 * p_points, F, 1, p, 1);

				//Error
				for (i = 0; i < p_points; i++)
				{
					F[i + 0 * p_points] = q[q_idx[i] + 0 * p_points];
					F[i + 1 * p_points] = q[q_idx[i] + 1 * p_points];
					F[i + 2 * p_points] = q[q_idx[i] + 2 * p_points];
				}
				vsSub(3 * p_points, p, F, F);//F = p - q_idx
				E[iteration + 1] = cblas_snrm2(3 * p_points, F, 1) / (float)pow(p_points, 0.5);//get the rms error

				if ((E[iteration + 1] < 0.000001) ||
					((float)fabs((double)E[iteration + 1] - (double)E[iteration]) < 0.000001)) break;

				iteration++;
			}
			end_time = dsecnd();
			seconds = end_time - ini_time;
			//printf("repeat %d: %.4f\n", repeat, seconds);
			if (seconds < min_time) min_time = seconds;
		}

		////////////////End of 2nd//////////////
		fprintf(document, "%d,%.4f\n", num_points, 1000.0* min_time);
		printf("%d,%.4f\n", num_points, 1000.0 * min_time);

		free(mesh_x), free(mesh_y), free(z), free(M), free(D);
		free(p), free(q), free(q_bar), free(p_bar), free(q_dev), free(p_dev), free(N), free(F), free(G);
		free(match), free(min_dist), free(q_idx), free(E), free(superb), free(U), free(Vt);
		free(S), free(p1), free(dist); free(temp_t), free(temp_r);
	}
	fclose(document);
	return 0;
}

//this function converts the vector t into a matrix by
//repeating t a number of times "repeat"
void repmat(float t[3], int repeat, float* matrix)
{
	int j, i;
	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < repeat; i++)
		{
			matrix[i + j * repeat] = t[j];
		}
	}
}

void print_cloud(float* cloud, int num_points, int points_show)
{
	int i, j;
	if (points_show <= num_points)
	{
		printf("x\ty\tz\n");
		for (i = 0; i < points_show; i++)
		{
			for (j = 0; j < 3; j++)
			{
				if (j % 3 != 2) printf("%.4f, ", cloud[i + j * num_points]);
				else printf("%.4f\n", cloud[i + j * num_points]);
			}
		}
	}
	else printf("The cloud can't be printed\n\n");
}

void print_array(float* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d: ", i + 1);
		if (i < size - 1) printf("%.4f\n", array[i]);
		else printf("%.4f\n", array[i]);
	}
}

void initialize_array(float* array, int n)
{
	for (int i = 0; i < n; i++) array[i] = 0.0f;
}

void centroid_deviation(int n, float* p, float* q, int* q_idx, float* p_bar, float* p_dev, float* q_bar, float* q_dev)
{
	int i, j;

	float xp = 0.0f, yp = 0.0f, zp = 0.0f;
	float xq = 0.0f, yq = 0.0f, zq = 0.0f;
	for (i = 0; i < n; i++)
	{
		xp += p[i + 0 * n];
		yp += p[i + 1 * n];
		zp += p[i + 2 * n];

		xq += q[q_idx[i] + 0 * n];
		yq += q[q_idx[i] + 1 * n];
		zq += q[q_idx[i] + 2 * n];
	}

	//centroid
	p_bar[0] = xp / (float)n;
	p_bar[1] = yp / (float)n;
	p_bar[2] = zp / (float)n;
	//printf("P bar:\n");
	//for (i = 0; i < 3; i++) printf("%.3f\n", p_bar[i]);

	q_bar[0] = xq / (float)n;
	q_bar[1] = yq / (float)n;
	q_bar[2] = zq / (float)n;
	//printf("Q bar:\n");
	//for (i = 0; i < 3; i++) printf("%.3f\n", q_bar[i]);

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 3; j++)
		{
			p_dev[i + j * n] = p[i + j * n] - p_bar[j];
			q_dev[i + j * n] = q[q_idx[i] + j * n] - q_bar[j];
		}
	}
}