#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int BLOCK_NUM = 4;
const int THREAD_PER_BLOCK = 512;
const int THREAD_NUM = BLOCK_NUM * THREAD_PER_BLOCK;

//kerbel function of matrix multiplication
template<class T>
__global__ void gpuMatMultKernel(T *a, T *b, T *result, const int N, const int M, const int K) {
	//thread id
	int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x
		+ blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < N * K) {
		int row = tid / N;
		int col = tid % K;
		T temp = 0;
		for (int i = 0; i < M; ++i) {
			temp += a[row * M + i] * b[i * K + col];
		}
		result[tid] = temp;
		tid += THREAD_NUM;
	}
}

//show the information of all devices on this computer
void showDevice() {
	cudaDeviceProp deviceProp;
	int deviceCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("没有检测到设备");
		return;
	}
	for (int i = 0; i < deviceCount; i++)
	{
		cudaError = cudaGetDeviceProperties(&deviceProp, i);

		printf("设备 %d 的主要属性:\n", i);
		printf("设备显卡型号： %s\n", deviceProp.name);
		printf("设备全局内存总量(MB)： %d\n", deviceProp.totalGlobalMem / 1024 / 1024);
		printf("设备上一个线程块（Block）中可用的最大共享内存(KB)： %d\n", deviceProp.sharedMemPerBlock / 1024);
		printf("设备上一个线程块（Block）种可用的32位寄存器数量： %d\n", deviceProp.regsPerBlock);
		printf("设备上一个线程块（Block）可包含的最大线程数量：%d\n", deviceProp.maxThreadsPerBlock);
		printf("设备的计算功能集（Compute Capability）的版本号：%d.%d\n", deviceProp.major, deviceProp.minor);
		printf("设备上多处理器的数量：%d\n", deviceProp.multiProcessorCount);
	}
}

//Matrix Multiplication
//a[n][m] * b[m][k] = c[n][k]
extern void matrixMultOnGPU(int **a, int **b, int **c, int n, int m, int k) {
	//printf("starting CUDA matrix multiplication (int)\n");

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("Error: No device");
		return;
	}

	//parameters on device
	int *dev_a = NULL;
	int *dev_b = NULL;
	int *dev_c = NULL;

	//multi-device is not supported
	cudaSetDevice(0);

	cudaError_t error;

	//allocate memory for calculations on device
	cudaMalloc((void**)&dev_a, sizeof(int) * n * m);
	cudaMalloc((void**)&dev_b, sizeof(int) * m * k);
	cudaMalloc((void**)&dev_c, sizeof(int) * n * k);
	error = cudaGetLastError();
	printf("CUDA malloc: %s\n", cudaGetErrorString(error));

	//copy memory to device
	cudaMemcpy(dev_a, a[0], sizeof(int) * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b[0], sizeof(int) * m * k, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	printf("CUDA host to device: %s\n", cudaGetErrorString(error));

	time_t t = clock();
	//kernel function <<<block_number, thread_per_block>>>
	gpuMatMultKernel<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(dev_a, dev_b, dev_c, n, m, k);
	cudaDeviceSynchronize(); //wait for multiplication over
	printf("%dms\n", clock() - t);
	error = cudaGetLastError();
	printf("CUDA multi: %s\n", cudaGetErrorString(error));

	//get memory from device
	cudaMemcpy(&c[0][0], dev_c, n * k * sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	printf("CUDA device to host: %s\n", cudaGetErrorString(error));

	//free
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	error = cudaGetLastError();
	printf("CUDA mem free: %s\n", cudaGetErrorString(error));

	//printf("CUDA matrix multiplication over\n");
}

//float version of matrix multiplication
extern void matrixMultOnGPU(float **a, float **b, float **c, int n, int m, int k) {
	//printf("starting CUDA matrix multiplication (float)\n");

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("Error: No device");
		return;
	}

	//parameters on device
	float *dev_a = NULL;
	float *dev_b = NULL;
	float *dev_c = NULL;

	//multi-device is not supported
	cudaSetDevice(0);

	cudaError_t error;

	//allocate memory for calculations on device
	cudaMalloc((void**)&dev_a, sizeof(float) * n * m);
	cudaMalloc((void**)&dev_b, sizeof(float) * m * k);
	cudaMalloc((void**)&dev_c, sizeof(float) * n * k);
	error = cudaGetLastError();
	printf("CUDA malloc: %s\n", cudaGetErrorString(error));

	//copy memory to device
	cudaMemcpy(dev_a, a[0], sizeof(float) * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b[0], sizeof(float) * m * k, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	printf("CUDA host to device: %s\n", cudaGetErrorString(error));

	//kernel function <<<block_number, thread_per_block>>>
	time_t t = clock();
	gpuMatMultKernel << <BLOCK_NUM, THREAD_PER_BLOCK >> > (dev_a, dev_b, dev_c, n, m, k);
	cudaDeviceSynchronize(); //wait for multiplication over
	printf("%dms\n", clock() - t);
	error = cudaGetLastError();
	printf("CUDA multi: %s\n", cudaGetErrorString(error));

	//get memory from device
	cudaMemcpy(c[0], dev_c, n * k * sizeof(float), cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	printf("CUDA device to host: %s\n", cudaGetErrorString(error));

	//free
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	error = cudaGetLastError();
	printf("CUDA mem free: %s\n", cudaGetErrorString(error));

	//printf("CUDA matrix multiplication over\n");
}
