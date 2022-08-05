//ICP algorithm using the point-to-plane error metric and synthetic data
//By: Carlos Huapaya

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mkl.h"
#include "mkl_lapacke.h"

//constants
#define WIDTH 128//width of grid
#define XY_min -2.0f
#define XY_max 2.0f
#define MAX_ITER 1

void repmat(float t[3], int repeat, float* matrix);
void print_cloud(float* cloud, int num_points, int points_show);
void print_array(float* array, int size);
void initialize_array(float* array, int n);

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

	//fopen_s(&document, "Test_icp", "w");

	//create an array with all points equally separated
	int n = WIDTH;
	for (i = 0; i < WIDTH; i++)
	{
		lin_space[i] = (float)XY_min + ((float)i * (float)lenght) / (float(n) - 1.0f);
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

	//print data cloud
	/*printf("Data point cloud\n");
	print_cloud(D, d_points, 10);*/

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
	for (i = 0; i < 3 * m_points; i++) M[i] = M[i] * 0.01 * rand();*/

	//print model cloud
	/*printf("Model point cloud\n");
	print_cloud(M, num_points, 10);*/

	////////////////End of 1st//////////////

	//since this lines 
	//p assumes the value of D
	//q assumes the value of M
	//number of p and q points
	int p_points = d_points;
	int q_points = m_points;
	float* p = (float*)malloc(bytesD);//p points cloud
	float* q = (float*)malloc(bytesM);//q point cloud
	//transfer data from D and M to p and q
	cblas_scopy(3 * d_points, D, 1, p, 1);
	cblas_scopy(3 * m_points, M, 1, q, 1);

	////////////////2nd: Normals estimation//////////////

	int count = 0, idx_min = 0;
	float* q1 = (float*)malloc(bytesM);
	float* dist = (float*)malloc(q_points * sizeof(float));
	k = 4;//using 4 nearest neighbors
	int* neighborIds = (int*)malloc((size_t)k * (size_t)q_points * sizeof(int));//for each point of q there are 4 points here

	float* normals = (float*)malloc(bytesM);//3 coordinates per normal
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
			cblas_scopy(q_points, &q[i + q_points * count], 0, &q1[q_points * count], 1);//copy vector in p
		}
		vsSub(3 * q_points, q, q1, q1);
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
			bar[0] += q[stride + 0 * q_points];
			bar[1] += q[stride + 1 * q_points];
			bar[2] += q[stride + 2 * q_points];
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
			xi = q[stride + 0 * q_points];
			yi = q[stride + 1 * q_points];
			zi = q[stride + 2 * q_points];
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
	printf("Normals were calculated in %f ms\n\n", 1000.0 * (end - start));
	/*printf("Normals:\n");
	print_cloud(normals, q_points, q_points);*/

	////////////////End of 2nd//////////////

	////////////////3rd:ICP algorithm//////////////
	int num_iterations = 0, iteration = 0;
	float bi = 0.0f;
	lapack_int ipiv[6] = {};
	float cx, cy, cz, sx, sy, sz;

	float* F = (float*)malloc(bytesD);
	float* match = (float*)malloc(p_points * sizeof(float));//indexing vector
	float* min_dist = (float*)malloc(p_points * sizeof(float));//minimum distance vector
	int* q_idx = (int*)malloc(p_points * sizeof(int));//correspondence between p and q
	float* p1 = (float*)malloc(bytesM);//auxiliar for matching
	float* C = (float*)malloc(36 * sizeof(float));//6x6 matrix for the system of linear equations
	float* x = (float*)malloc(6 * sizeof(float));//vector with the 6 DOF
	float* b = (float*)malloc(6 * sizeof(float));//right-hand vector
	float* cn = (float*)malloc(6 * sizeof(float));
	float* temp_t = (float*)malloc(3 * sizeof(float));//temporary translation vector
	float* temp_r = (float*)malloc(9 * sizeof(float));//temporary rotation matrix
	float* E = (float*)malloc(size_t(MAX_ITER + 1) * sizeof(float));//ERROR
	initialize_array(E, int(MAX_ITER) + 1);

	double match_time = 0, minimization_time = 0, transf_time = 0, error_time = 0;
	//double start = 0, end = 0;

	double ini_time, end_time;
	//ICP MAIN LOOP
	ini_time = dsecnd();
	while (iteration < MAX_ITER)
	{
		/////////////////matching step/////////////////
		start = dsecnd();
		for (j = 0; j < p_points; j++)
		{
			//bruteforce
			for (count = 0; count < 3; count++)
			{
				cblas_scopy(q_points, &p[j + p_points * count], 0, &p1[q_points * count], 1);//copy vector in p
			}
			vsSub(3 * q_points, q, p1, p1);
			vsSqr(3 * q_points, p1, p1);//p1=p1^2
			//sum of all the calculated distances
			vsAdd(q_points, p1, &p1[1 * q_points], dist);
			vsAdd(q_points, dist, &p1[2 * q_points], dist);
			idx_min = (int)cblas_isamin(q_points, dist, 1);
			q_idx[j] = idx_min;
		}
		end = dsecnd();
		match_time += (end - start);
		/*printf("q_idx:\n");
		for (i = 0; i < p_points; i++) printf("%d: %d\n",i+1, q_idx[i] + 1);*/

		/////////////////Minimization step (point-to-plane)/////////////////
		start = dsecnd();
		initialize_array(C, 36);//initialize C
		initialize_array(b, 6);//initialize b
		//printf("cn:\n");
		for (i = 0; i < p_points; i++)
		{
			stride = q_idx[i];//correspondence
			cn[0] = p[i + 1 * p_points] * normals[stride + 2 * q_points] -
				p[i + 2 * p_points] * normals[stride + 1 * q_points];//cix
			cn[1] = p[i + 2 * p_points] * normals[stride + 0 * q_points] -
				p[i + 0 * p_points] * normals[stride + 2 * q_points];//ciy
			cn[2] = p[i + 0 * p_points] * normals[stride + 1 * q_points] -
				p[i + 1 * p_points] * normals[stride + 0 * q_points];//ciz
			cn[3] = normals[stride + 0 * q_points];//nix
			cn[4] = normals[stride + 1 * q_points];//niy
			cn[5] = normals[stride + 2 * q_points];//niz
			/*printf("%d: ",i+1);
			print_array(cn, 6);*/

			//Find C
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				6, 6, 1, 1, cn, 1, cn, 6, 1, C, 6);//C = cn*cn(T) + C

			//Find b
			bi = (p[i + 0 * p_points] - q[stride + 0 * q_points]) * cn[3] +
				(p[i + 1 * p_points] - q[stride + 1 * q_points]) * cn[4] +
				(p[i + 2 * p_points] - q[stride + 2 * q_points]) * cn[5];
			for (j = 0; j < 6; j++) b[j] += (-1) * cn[j] * bi;
		}
		/*printf("C[%d]:\n", i + 1);
		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++) printf("%.3f ", C[j + i * 6]);
			printf("\n");
		}*/

		//solve the system of linear equations
		info = LAPACKE_ssysv(LAPACK_ROW_MAJOR, 'U', 6, 1, C, 6, ipiv, b, 1);//b holds the answer x
		if (info != 0) {
			printf("Info: %d\n", info);
			printf("The algorithm failed to solve the system of linear equations.\n");
			exit(1);
		}

		//rotation matrix
		cx = (float)cos(b[0]); cy = (float)cos(b[1]); cz = (float)cos(b[2]);
		sx = (float)sin(b[0]); sy = (float)sin(b[1]); sz = (float)sin(b[2]);
		temp_r[0] = cy * cz; temp_r[1] = cz * sx * sy - cx * sz;  temp_r[2] = cx * cz * sy + sx * sz;
		temp_r[3] = cy * sz; temp_r[4] = cx * cz + sx * sy * sz; temp_r[5] = cx * sy * sz - cz * sx;
		temp_r[6] = -sy; temp_r[7] = cy * sx; temp_r[8] = cx * cy;
		//translation vector
		temp_t[0] = b[3];
		temp_t[1] = b[4];
		temp_t[2] = b[5];

		end = dsecnd();
		minimization_time += (end - start);
		/*printf("R:\n");
		print_cloud(temp_r, 3, 3);
		printf("T:\n");
		print_array(temp_t, 3);*/

		/////////////////Transformation step/////////////////
		start = dsecnd();
		repmat(temp_t, p_points, F);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			3, p_points, 3, 1, temp_r, 3, p, p_points, 1, F, p_points);//F = temp_r * p + F
		cblas_scopy(3 * p_points, F, 1, p, 1);
		end = dsecnd();
		transf_time += (end - start);

		/////////////////Error estimation/////////////////
		start = dsecnd();
		for (i = 0; i < p_points; i++)
		{
			F[i + 0 * p_points] = q[q_idx[i] + 0 * p_points];
			F[i + 1 * p_points] = q[q_idx[i] + 1 * p_points];
			F[i + 2 * p_points] = q[q_idx[i] + 2 * p_points];
		}
		vsSub(3 * p_points, p, F, F);//F = p - q_idx
		E[iteration + 1] = cblas_snrm2(3 * p_points, F, 1) / (float)pow(p_points, 0.5);//get the rms error
		end = dsecnd();
		error_time += (end - start);

		if ((E[iteration + 1] < 0.000001) ||
			((float)fabs((double)E[iteration + 1] - (double)E[iteration]) < 0.000001)) break;

		if (iteration == 0)
		{
			end = dsecnd();
			printf("The fisrt iteration was compute in %.4f ms\n\n", (end - ini_time) * 1000.0);
		}

		iteration++;
	}
	end_time = dsecnd();
	////////////////End of 3rd//////////////

	printf("Error\n");
	print_array(E, iteration+1);

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

	free(mesh_x), free(mesh_y), free(z), free(M), free(D);
	free(p), free(q), free(F);
	free(q1), free(neighborIds), free(bar), free(normals);
	free(match), free(min_dist), free(q_idx), free(p1), free(dist);
	free(C), free(x), free(b), free(cn);
	free(temp_t), free(temp_r);
	free(E);

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
		//printf("x\ty\tz\n");
		for (i = 0; i < points_show; i++)
		{
			printf("%d: ", i + 1);
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