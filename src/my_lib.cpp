#include <stdio.h>

//all of these functions operate with colum-major order matrices

//float matrix multiplication
void fmatrixMul(float *A,float *B, float *C, int m, int n, int k)
{
    int i,j,q;
    float temp;
    for (i = 0; i < n; i++) 
    { 
        for (j = 0; j < m; j++) 
        { 
            temp = 0; 
            for (q = 0; q < k; q++) temp += A[j + q * m] *  B[q + i * k]; 
            C[j + i * m] = temp;
        }
    }
}

//double matrix multiplication
void dmatrixMul(double *A,double *B, double *C, int m, int n, int k)
{
    int i,j,q;
    double temp;
    for (i = 0; i < n; i++) 
    { 
        for (j = 0; j < m; j++) 
        { 
            temp = 0.0f; 
            for (q = 0; q < k; q++) temp += A[j + q * m] *  B[q + i * k]; 
            C[j + i * m] = temp;
        }
    }
}

//print matrix
void print_cloud(double* cloud, int num_points, int points2show)//
{
	int i, j,offset;
	printf("x\ty\tz\n");
	if (points2show <= num_points)
	{
		for(i = 0; i < points2show; i++)
		{
			for(j = 0; j < 3; j++)
			{
				offset = j + i * 3;
				printf("%.4f\t",cloud[offset]);
				if(j%3==2) printf("\n");
			}
		}
	}
	else printf("The cloud can't be printed\n\n");
}

//print vector with double values
void print_darray(double* array, int points2show)
{
    int i;
    for(i = 0; i < points2show; i++)
    {
        printf("%.3f ", array[i]);
    }
    printf("\n");
}

//print vector with integer values
void print_iarray(int* array, int points2show)
{
    int i;
    for(i = 0; i < points2show; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
