#include "kernel.h"

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
	cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("û�м�⵽�豸");
		return;
	}
	for (int i = 0; i < deviceCount; i++){
		cudaError = cudaGetDeviceProperties(&deviceProp, i);

		printf("�豸 %d ����Ҫ����:\n", i);
		printf("�豸�Կ��ͺţ� %s\n", deviceProp.name);
		printf("�豸ȫ���ڴ�����(MB)�� %d\n", deviceProp.totalGlobalMem / 1024 / 1024);
		printf("�豸��һ���߳̿飨Block���п��õ�������ڴ�(KB)�� %d\n", deviceProp.sharedMemPerBlock / 1024);
		printf("�豸��һ���߳̿飨Block���ֿ��õ�32λ�Ĵ��������� %d\n", deviceProp.regsPerBlock);
		printf("�豸��һ���߳̿飨Block���ɰ���������߳�������%d\n", deviceProp.maxThreadsPerBlock);
		printf("�豸�ļ��㹦�ܼ���Compute Capability���İ汾�ţ�%d.%d\n", deviceProp.major, deviceProp.minor);
		printf("�豸�϶ദ������������%d\n", deviceProp.multiProcessorCount);
	}
}

extern void GPUMatrixMul(float **a, float **b, float **c, int n, int m, int k) {
	//detect available device(s)
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount <= 0) {
		printf("GPUMatrixMul(): Error, no device.\n");
		return;
	}

	//multi-device is not supported
	cudaSetDevice(0);

	//parameters on device
	float *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;
	const int TYPE_SIZE = sizeof(float);

	//allocate memory for calculations on device
	cudaMalloc((void**)&dev_a, TYPE_SIZE * n * m);
	cudaMalloc((void**)&dev_b, TYPE_SIZE * m * k);
	cudaMalloc((void**)&dev_c, TYPE_SIZE * n * k);

	//copy memory to device
	cudaMemcpy(dev_a, a[0], TYPE_SIZE * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b[0], TYPE_SIZE * m * k, cudaMemcpyHostToDevice);

	//kernel function <<<block_number, thread_per_block>>>
	gpuMatMultKernel<<<BLOCK_NUM, THREAD_PER_BLOCK >>> (dev_a, dev_b, dev_c, n, m, k);
	cudaDeviceSynchronize();

	//get memory from device
	cudaMemcpy(c[0], dev_c, n * k * TYPE_SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//free
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void GPUMatrixMul(double **a, double **b, double **c, int n, int m, int k) {
	//detect available device(s)
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount <= 0) {
		printf("GPUMatrixMul(): Error, no device.\n");
		return;
	}

	//multi-device is not supported
	cudaSetDevice(0);

	//parameters on device
	double *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;
	const int TYPE_SIZE = sizeof(double);

	//allocate memory for calculations on device
	cudaMalloc((void**)&dev_a, TYPE_SIZE * n * m);
	cudaMalloc((void**)&dev_b, TYPE_SIZE * m * k);
	cudaMalloc((void**)&dev_c, TYPE_SIZE * n * k);

	//copy memory to device
	cudaMemcpy(dev_a, a[0], TYPE_SIZE * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b[0], TYPE_SIZE * m * k, cudaMemcpyHostToDevice);

	//kernel function <<<block_number, thread_per_block>>>
	gpuMatMultKernel << <BLOCK_NUM, THREAD_PER_BLOCK >> > (dev_a, dev_b, dev_c, n, m, k);
	cudaDeviceSynchronize();

	//get memory from device
	cudaMemcpy(c[0], dev_c, n * k * TYPE_SIZE, cudaMemcpyDeviceToHost);

	//free
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void GPUMatrixMul(int **a, int **b, int **c, int n, int m, int k) {
	//detect available device(s)
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount <= 0) {
		printf("GPUMatrixMul(): Error, no device.\n");
		return;
	}

	//multi-device is not supported
	cudaSetDevice(0);

	//parameters on device
	int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;
	const int TYPE_SIZE = sizeof(int);

	//allocate memory for calculations on device
	cudaMalloc((void**)&dev_a, TYPE_SIZE * n * m);
	cudaMalloc((void**)&dev_b, TYPE_SIZE * m * k);
	cudaMalloc((void**)&dev_c, TYPE_SIZE * n * k);

	//copy memory to device
	cudaMemcpy(dev_a, a[0], TYPE_SIZE * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b[0], TYPE_SIZE * m * k, cudaMemcpyHostToDevice);

	//kernel function <<<block_number, thread_per_block>>>
	gpuMatMultKernel << <BLOCK_NUM, THREAD_PER_BLOCK >> > (dev_a, dev_b, dev_c, n, m, k);
	cudaDeviceSynchronize();

	//get memory from device
	cudaMemcpy(c[0], dev_c, n * k * TYPE_SIZE, cudaMemcpyDeviceToHost);

	//free
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
