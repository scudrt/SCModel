#pragma once

#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//show the information of all devices on this computer
void showDevice();

template<class T>
T** makeNewMatrix(int row, int col) {
	T** ret = new T*[row];
	ret[0] = new T[row * col];
	for (int i = 1; i < row; ++i) {
		ret[i] = ret[i - 1] + col;
	}
	return ret;
}

template <class T>
void CPUMatrixMul(T **a, T **b, T **c, int n, int m, int k) {
	for (int kk = 0; kk < k; ++kk) {
		for (int i = 0; i < n; ++i) {
			T temp = 0;
			for (int j = 0; j < m; ++j) {
				temp += a[i][j] * b[j][kk];
			}
			c[i][kk] = temp;
		}
	}
}

extern void GPUMatrixMul(double **a, double **b, double **c, int n, int m, int k);
extern void GPUMatrixMul(float **a, float **b, float **c, int n, int m, int k);
extern void GPUMatrixMul(int **a, int **b, int **c, int n, int m, int k);
