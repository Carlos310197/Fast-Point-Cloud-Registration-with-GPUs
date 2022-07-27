#ifndef _MYLIB
#define _MYLIB

void fmatrixMul(float *A,float *B, float *C, int m, int n, int k);
void dmatrixMul(double *A,double *B, double *C, int m, int n, int k);
void print_cloud(double* cloud, int num_points, int points2show);
void print_darray(double* array, int points2show);
void print_iarray(int* array, int points2show);

#include  "my_lib.cpp"
#endif