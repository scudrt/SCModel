#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

const int maxn = 3000;
const int mod = 10007;

void showDevice();
extern void matrixMultOnGPU(int**, int**, int**, int, int, int);
extern void matrixMultOnGPU(float**, float**, float**, int, int, int);

template<class T>
inline void matrixMulOnCPU(T **a, T **b, T **c, int n, int m, int k) {
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

int main() {

	//showDevice();

	ios::sync_with_stdio(false);
	cout << "initialising Matrix Multiplication test..." << endl;

	float** a = new float*[maxn];
	float** b = new float*[maxn];
	float** c = new float*[maxn];
	float** d = new float*[maxn];

	a[0] = new float[maxn * maxn];
	b[0] = new float[maxn * maxn];
	c[0] = new float[maxn * maxn];
	d[0] = new float[maxn * maxn];
	for (int i = 1; i < maxn; ++i) {
		a[i] = a[0] + i * maxn;
		b[i] = b[0] + i * maxn;
		c[i] = c[0] + i * maxn;
		d[i] = d[0] + i * maxn;
	}

	srand(time(0));
	for (int i = 0; i < maxn * maxn; ++i) {
		a[0][i] = (float)(rand() % mod / 100.0f);
		b[0][i] = (float)(rand() % mod / 100.0f);
		c[0][i] = d[0][i] = 0.0f;
	}

	time_t t;

	/*
	//CPU
	cout << "start runing on cpu:" << '\n';
	t = clock();
	matrixMulOnCPU(a, b, d, maxn, maxn, maxn);
	cout << "cpu time: " << clock() - t << "ms\n" << endl;
	*/
	
	//GPU
	cout << "start running on gpu:\n";
	t = clock();
	matrixMultOnGPU(a, b, c, maxn, maxn, maxn);
	cout << "gpu time: " << clock() - t << "ms" << endl;

	/*
	printf("showing differences:\n");
	int diffCount = 0;
	for (int i = 0; i < maxn; ++i) {
		for (int j = 0; j < maxn; ++j) {
			if (abs((c[i][j] - d[i][j]) / d[i][j]) >= 1e-4) {
				++diffCount;
				cout << i << ' ' << j << ' ' << c[i][j] << ' ' << d[i][j] << endl;
			}
		}
	}
	cout << "total: " << maxn * maxn << endl;
	cout << "differences: " << diffCount << endl;
	*/

	delete[] a;delete[] b;
	delete[] c;delete[] d;

	system("pause");
	return 0;
}
