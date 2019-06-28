#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#include "kernel.h"

using namespace std;

const int maxn = 500;
const int mod = 10007;

int main() {
	showDevice();

	ios::sync_with_stdio(false);
	cout << "initialising Matrix Multiplication test..." << endl;

	srand(time(0));

	float **a, **b, **c, **d;
	time_t t;

	for (int limit = 1500; limit < 1501; limit += 10) {
		a = makeNewMatrix<float>(limit, limit);
		b = makeNewMatrix<float>(limit, limit);
		c = makeNewMatrix<float>(limit, limit);
		d = makeNewMatrix<float>(limit, limit);
		for (int i = 0; i < limit; ++i) {
			for (int j = 0; j < limit; ++j) {
				a[i][j] = rand() % mod / 100.0f;
				b[i][j] = rand() % mod / 100.0f;
				c[i][j] = d[i][j] = 0.0f;
			}
		}
		cout << "testing :" << limit << 'x' << limit << endl;
		
		cout << "start running on cpu:\n";
		t = clock();
		CPUMatrixMul(a, b, d, limit, limit, limit);
		cout << "cpu time: " << clock() - t << "ms" << endl;
		
		//run on GPU
		cout << "start running on gpu:\n";
		t = clock();
		GPUMatrixMul(a, b, c, limit, limit, limit);
		cout << "gpu time: " << clock() - t << "ms" << endl;
		
		int diff = 0;
		for (int i = 0; i < limit; ++i) {
			for (int j = 0; j < limit; ++j) {
				if (abs((c[i][j] - d[i][j]) / d[i][j]) >= 1e-6) {
					cout << fixed << setprecision(4) << c[i][j] << ' ' << d[i][j] << endl;
					++diff;
				}
			}
		}
		cout << "final diff = " << diff << endl;
		cout << '\n' << endl;
		delete[] a; delete[] b;
		delete[] c; delete[] d;
	}
	system("pause");
	return 0;
}
