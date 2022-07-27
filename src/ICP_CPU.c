#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>

//constants
#define WIDTH 100
//width of grid
#define NUM_POINTS WIDTH*WIDTH
#define XY_min -2.0
#define XY_max 2.0
#define MAX_ITER 200

FILE* document;

void repmat(double t[3], int repeat, double* answer);
void print_cloud(double* cloud, int num_points, int points_show);
void copy_cloud(double* p, double* q, int num_points);
void copy_vector(double* array1, double* array2, int size);
void initialize_vector(double* vector, int n);
int minimum_value(int a, int b);
void print_array(double* array, int size);
void error(double* q, int q_size, double* q_idx, double* p, int p_size, double* p_idx, double* E, int pos);
void print_all(double* D, double* M, double** pt_total, double* E, int num_points, int num_iterations);
void centroid_deviation(double* cloud, int cloud_size, int* index, double index_size, double* cloud_bar, double* cloud_mark);

int main(void)
{
	////////////////////////////////////1st:Declaration of variables///////////////////////////////////////

	//declarate variables
	int num_points, i, j, k;
	double ti[3], ri[3];
	double lin_space[WIDTH], lenght;
	double* mesh_x, * mesh_y, * z, * D, * M, * C;

	num_points = WIDTH * WIDTH;//number of points
	lenght = XY_max - XY_min;

	/////////////////////////////////////////////End of 1st////////////////////////////////////////////////

	////////////////2nd:Creation of the synthetic dataset and model and data point clouds variables////////

	//fopen_s(&document, "Test_icp", "w");

	//create an array with all points equally separated
	int n = WIDTH;
	for (i = 0; i < WIDTH; i++)
	{
		lin_space[i] = XY_min + double(i) * (lenght) / (double(n) - 1.0);
	}

	//create the meshgrid
	mesh_x = (double*)malloc(num_points * sizeof(double));
	mesh_y = (double*)malloc(num_points * sizeof(double));

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
	z = (double*)malloc(num_points * sizeof(double));
	for (i = 0; i < num_points; i++) z[i] = pow(mesh_x[i], 2) - pow(mesh_y[i], 2);

	//Create data point cloud matrix
	D = (double*)malloc(num_points * 3.0 * sizeof(double));

	j = 0;
	for (i = 0; i < 3; i++)
	{
		k = 0;
		while (k < num_points)
		{
			if (i == 0) D[j] = mesh_x[k];
			if (i == 1) D[j] = mesh_y[k];
			if (i == 2) D[j] = z[k];
			j++;
			k++;
		}
	}

	/*printf("Data point cloud\n");
	print_cloud(D, num_points, 10);*/

	//Translation values
	ti[0] = 1.0;//x
	ti[1] = -0.3;//y
	ti[2] = 0.2;//z

	//Rotation values (rad)
	ri[0] = 1;//axis x
	ri[1] = -0.5;//axis y
	ri[2] = 0.05;//axis z

	double rx[3][3] = {
		{ 1, 0, 0 },
		{ 0, cos(ri[0]), sin(ri[0]) },
		{ 0, -sin(ri[0]), cos(ri[0]) },
	};

	double ry[3][3] = {
		{ cos(ri[1]), 0, -sin(ri[1]) },
		{ 0, 1, 0 },
		{ sin(ri[1]), 0, cos(ri[1]) },
	};

	double rz[3][3] = {
		{ cos(ri[2]), sin(ri[2]),0 },
		{ -sin(ri[2]), cos(ri[2]), 0 },
		{ 0, 0, 1 },
	};

	//rotation matrix
	double r[3][3] = {};

	//matrix multiplication
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, *rx, 3, *ry, 3, 0, *r, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, *r, 3, *rz, 3, 0, *r, 3);

	M = (double*)malloc(3.0 * num_points * sizeof(double));

	//Initialize the model point cloud M
	for (i = 0; i < 3 * num_points; i++) M[i] = 0;

	//Construct the model point cloud M
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, num_points, 3, 1, *r, 3, D, num_points, 0, M, num_points);//M = r*D

	//C = repmat(t,1,num_points)
	C = (double*)malloc(3.0 * num_points * sizeof(double));
	repmat(ti, num_points, C);

	//srand(time(0));
	//M = M + C
	for (i = 0; i < 3 * num_points; i++) M[i] += C[i];//(C[i] * 0.00001*rand());

	/*printf("Model point cloud\n");
	print_cloud(M, num_points, 10);*/

	/////////////////////////////////////////////End of 2nd////////////////////////////////////////////////

	//////////////////////////////////////////3rd:ICP Algorithm////////////////////////////////////////////

	int num_iterations = 0, p_size, q_size, count, idx_dmin,* p_idx, * q_idx;
	double* p, * q, * pt, * match, * min_dist, * d,
		* q_bar, * q_mark, * p_bar, * p_mark, * N, * E,
		* U, * Vt, * S, * superb, * G, * F, * p1, * x, * dist, * zero_vector;
	double** pt_total;
	double time1;

	MKL_INT info;
	p_size = num_points;
	q_size = num_points;

	p = (double*)malloc(3.0 * num_points * sizeof(double));
	pt = (double*)malloc(3.0 * num_points * sizeof(double));
	q = (double*)malloc(3.0 * num_points * sizeof(double));
	q_bar = (double*)malloc(3 * sizeof(double));
	p_bar = (double*)malloc(3 * sizeof(double));
	q_mark = (double*)malloc(3.0 * num_points * sizeof(double));
	p_mark = (double*)malloc(3.0 * num_points * sizeof(double));
	N = (double*)malloc(3 * 3 * sizeof(double));
	F = (double*)malloc(3.0 * p_size * sizeof(double));
	G = (double*)malloc(3 * sizeof(double));
	match = (double*)malloc(p_size * sizeof(double));//indexing vector
	min_dist = (double*)malloc(p_size * sizeof(double));//minimum distance vector
	d = (double*)malloc(q_size * sizeof(double));//distance vector
	p_idx = (int*)malloc(p_size * sizeof(int));
	q_idx = (int*)malloc(q_size * sizeof(int));
	E = (double*)malloc((MAX_ITER+1.0) * sizeof(double));//maximum iterations: 100
	superb = (double*)malloc((minimum_value(3, 3) - 1.0) * sizeof(double));
	U = (double*)malloc(3 * 3 * sizeof(double));
	Vt = (double*)malloc(3 * 3 * sizeof(double));
	S = (double*)malloc(3 * 3 * sizeof(double));
	p1 = (double*)malloc(3.0 * p_size * sizeof(double));
	x = (double*)malloc(3.0 * p_size * sizeof(double));
	dist = (double*)malloc(p_size * sizeof(double));
	zero_vector = (double*)malloc(3.0 * num_points * sizeof(double));

	initialize_vector(zero_vector, 3 * num_points);

	//pointer "pt_total" is used for the registration of all temporary transformed data point clouds
	pt_total = (double**)malloc(MAX_ITER * sizeof(double*));
	for (count = 0; count < MAX_ITER; count++)
	{
		*(pt_total + count) = (double*)malloc(MAX_ITER * num_points * sizeof(double));
	}

	cblas_dcopy(3 * num_points, D, 1, p, 1);
	cblas_dcopy(3 * num_points, M, 1, q, 1);
	cblas_dcopy(3 * num_points, p, 1, pt, 1);

	//ICP main loop
	double temp_t[3] = { 0,0,0 };
	double temp_r[3][3] = { {1,0,0},{0,1,0},{0,0,1} };
	initialize_vector(E, 101);
	double ini_time, end_time;

	for (k = 0; k < p_size; k++) p_idx[k] = k;//index of p

	ini_time = dsecnd();
	i = 0;
	while(1)
	{
		//Matching step (brute force)
		for (j = 0; j < p_size; j++)
		{
			//bruteforce
			for (count = 0; count < 3; count++)
			{
				cblas_dcopy(p_size, (pt + j + p_size * count), 0, (p1 + (p_size * count)), 1);//copy vector in p
			}
			vdSub(3 * q_size, q, p1, p1);
			vdSqr(3 * q_size, p1, p1);//p1=p1^2
			//sum of all the calculated distances
			vdAdd(q_size, p1, (p1 + 1 * q_size), dist);
			vdAdd(q_size, dist, (p1 + 2 * q_size), dist);
			idx_dmin = cblas_idamin(q_size, dist, 1);
			q_idx[j] = idx_dmin;
		}

		//Minimization step (point-to-point)
		centroid_deviation(q, q_size, q_idx, p_size, q_bar, q_mark);//model cloud's centroid and deviation 
		centroid_deviation(pt, p_size, p_idx, p_size, p_bar, p_mark);//data cloud's centroid and deviation 
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, q_size, 1, q_mark, q_size, p_mark, q_size, 0, N, 3);
		info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', 3, 3, N, 3, S, U, 3, Vt, 3, superb);
		if (info > 0)
		{
			printf("The algorithm computing SVD failed to converge.\n");
			return 1;
		}
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, U, 3, Vt, 3, 0, *temp_r, 3);//temp_r = U * Vt
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 1, 3, 1, *temp_r, 3, p_bar, 3, 0, G, 1);//G = temp_r*p_bar
		vdSub(3, q_bar, G, temp_t);//temp_t = q_bar - G

		//transformation step
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, p_size, 3, 1, *temp_r, 3, pt, p_size, 0, C, p_size);//C = temp_r*pt
		repmat(temp_t, p_size, F);//F = repmat(temp_T,1,p_size)
		vdAdd(3 * p_size, C, F, pt);//apply transformation to pt (pt = C + F)
		cblas_dcopy(3 * p_size, pt, 1, *(pt_total + i), 1);//save current pt in pt_total

		//Error
		for (j = 0; j < q_size; j++)//copy vector q in C using q_idx
		{
			for (k = 0; k < 3; k++)
			{
				*(C + j + k * q_size) = *(q + q_idx[j] + k * q_size);
			}
		}
		vdSub(3 * q_size, C, pt, C);//C = C - pt
		//sum of all the calculated distances
		E[i + 1] = cblas_dnrm2(3 * q_size, C, 1) / pow(num_points, 0.5);//get the rms error
		if ((E[i + 1] < 0.00001) || (fabs((E[i + 1] - E[i])) < 0.00001)) break;
		i++;
		if (i > MAX_ITER-1) break;
		if (i == 10) time1 = dsecnd();
	}
	end_time = dsecnd();

	num_iterations = i;

	printf("Error\n");
	print_array(E, num_iterations + 1);

	printf("Time for one iteration: %f ms\n\n", (time1-ini_time)*1000);
	printf("The ICP algorithm was computed in %f ms with %d iterations\n\n", 1000.0 * (end_time - ini_time), num_iterations);

	/////////////////////////////////////////////End of 3rd////////////////////////////////////////////////
	free(mesh_x); free(mesh_y); free(z); free(M); free(D); free(C); free(p); free(q); free(pt);
	free(match); free(min_dist); free(d); free(q_bar); free(p_bar); free(q_mark); free(p_mark);
	free(p_idx); free(q_idx); free(N); free(F); free(G); free(superb); free(U); free(Vt);
	free(S); free(E); free(p1); free(x); free(dist);
	for (count = 0; count < num_iterations; count++) free(*(pt_total + count));
	free(pt_total);

	return 0;
	system("PAUSE");
}

//repmat converts a 3d vector in a matrix (3xrepeat) by repeating the vector 'repeat' times
void repmat(double t[3], int repeat, double* answer)
{
	int j, i;
	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < repeat; i++)
		{
			answer[i + j * repeat] = t[j];
		}
	}
}
void print_cloud(double* cloud, int num_points, int points_show)
{
	int i, j;
	if (points_show <= num_points)
	{
		for (j = 0; j < 3; j++)
		{
			for (i = 0; i < points_show; i++) printf("%.5f, ", cloud[i + j * num_points]);
			printf("\n");
		}
		printf("\n\n");
	}
	else printf("The cloud can't be printed\n\n");
}

void copy_cloud(double* p, double* q, int num_points)
{
	for (int i = 0; i < 3 * num_points; i++)
	{
		p[i] = q[i];
	}
}

void copy_vector(double* array1, double* array2, int size)
{
	for (int i = 0; i < size; i++)
	{
		array1[i] = array2[i];
	}
}

void initialize_vector(double* vector, int n)
{
	for (int i = 0; i < n; i++) vector[i] = 0;
}

void centroid_deviation(double* cloud, int cloud_size, int* index, double index_size, double* cloud_bar, double* cloud_mark)
{
	int i, j;

	double x = 0, y = 0, z = 0;
	for (i = 0; i < cloud_size; i++)
	{
		x += *(cloud + int(index[i] + (0.0 * cloud_size)));
		y += *(cloud + int(index[i] + (1.0 * cloud_size)));
		z += *(cloud + int(index[i] + (2.0 * cloud_size)));
	}

	//centroid
	cloud_bar[0] = x * (1.0 / double(cloud_size));
	cloud_bar[1] = y * (1.0 / double(cloud_size));
	cloud_bar[2] = z * (1.0 / double(cloud_size));

	for (i = 0; i < cloud_size; i++)
	{
		for (j = 0; j < 3; j++)
		{
			cloud_mark[i + (cloud_size * j)] = *(cloud + int(index[i] + (j * cloud_size))) - cloud_bar[j];
		}
	}
}

int minimum_value(int a, int b)
{
	if (a > b) return b;
	else return a;
}

void print_array(double* array, int size)
{
	for (int i = 0; i < size; i++) printf("%.5f, ", array[i]);
	printf("\n\n");
}

void error(double* q, int q_size, double* q_idx, double* p, int p_size, double* p_idx, double* E, int pos)
{
	double* pot, * dsq;
	double total = 0;
	pot = (double*)malloc(3.0 * p_size * sizeof(double));
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < p_size; j++)
		{
			pot[j + i * p_size] = pow(q[int(q_idx[j]) + i * q_size] - p[int(p_idx[j]) + i * p_size], 2);
		}
	}

	dsq = (double*)malloc(p_size * sizeof(double));

	for (j = 0; j < p_size; j++)
	{
		dsq[j] = pot[j + 0 * p_size] + pot[j + 1 * p_size] + pot[j + 2 * p_size];
		total += dsq[j];
	}

	E[pos + 1] = pow(total / double(p_size), 0.5);

	free(pot);
	free(dsq);
}

void print_all(double* D, double* M, double** pt_total, double* E, int num_points, int num_iterations)
{
	int i, j, x, y, z;
	//fprintf(document,"Data point cloud|Model point cloud");
	//for (i = 0; i < 3; i++) fprintf(document,"|Transformed data point cloud (iteration: %d)", i + 1);
	//fprintf(document,"\n");

	fprintf(document, "x_data|y_data|z_data|x_model|y_model|z_model");
	for (i = 0; i < num_iterations; i++) fprintf(document, "|TDx_%d|TDy_%d|TDz_%d", i + 1, i + 1, i + 1);
	fprintf(document, "\n");

	x = 0;
	y = 1;
	z = 2;
	for (i = 0; i < num_points; i++)
	{
		fprintf(document, "%- 7.3f| ", D[i + x * num_points]);
		fprintf(document, "%- 7.3f| ", D[i + y * num_points]);
		fprintf(document, "%- 7.3f| ", D[i + z * num_points]);

		fprintf(document, "%- 7.3f| ", M[i + x * num_points]);
		fprintf(document, "%- 7.3f| ", M[i + y * num_points]);
		fprintf(document, "%- 7.3f| ", M[i + z * num_points]);

		for (j = 0; j < num_iterations; j++)
		{
			fprintf(document, "%- 7.3f| ", pt_total[j][i + x * num_points]);
			fprintf(document, "%- 7.3f| ", pt_total[j][i + y * num_points]);
			fprintf(document, "%- 7.3f| ", pt_total[j][i + z * num_points]);
		}
		fprintf(document, "\n");
	}

	fprintf(document, "\nError|");
	for (i = 0; i < num_iterations; i++)
	{
		fprintf(document, "%.3f|", E[i]);
	}
	fprintf(document, "\n");
}