#include <chrono>
#include <iostream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

#define vvi vector<vector<int>>
#define vi vector<int>

vvi multiply(vvi &a, vvi &b) {
  int n = a.size(), m = b[0].size(), p = b.size();
  vvi result(n, vi(m, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < p; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

vvi multiplyBlocks(const vvi &a, const vvi &b, int blockSize) {
  int n = a.size(), m = b[0].size(), p = b.size();
  vvi result(n, vi(m, 0));

  for (int ii = 0; ii < n; ii += blockSize)
    for (int jj = 0; jj < m; jj += blockSize)
      for (int kk = 0; kk < p; kk += blockSize)

        for (int i = ii; i < std::min(ii + blockSize, n); i++)
          for (int j = jj; j < std::min(jj + blockSize, m); j++)
            for (int k = kk; k < std::min(kk + blockSize, p); k++)
              result[i][j] += a[i][k] * b[k][j];
  return result;
}

void printMatrix(const vvi &mat) {
  for (auto &row : mat) {
    for (auto val : row) {
      cout << val << " ";
    }
    cout << endl;
  }
}

int main() {
    vvi A10(10, vi(10, 1));
    vvi B10(10, vi(10, 2));

    auto t1 = std::chrono::high_resolution_clock::now();
    vvi C10 = multiply(A10, B10);
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << "clásica 10x10: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;

    t1 = std::chrono::high_resolution_clock::now();
    vvi C10b = multiplyBlocks(A10, B10, 2);
    t2 = std::chrono::high_resolution_clock::now();
    cout << "bloques 10x10: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;


    vvi A100(100, vi(100, 1));
    vvi B100(100, vi(100, 2));

    t1 = std::chrono::high_resolution_clock::now();
    vvi C100 = multiply(A100, B100);
    t2 = std::chrono::high_resolution_clock::now();
    cout << "clásica 100x100: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;

    t1 = std::chrono::high_resolution_clock::now();
    vvi C100b = multiplyBlocks(A100, B100, 10);
    t2 = std::chrono::high_resolution_clock::now();
    cout << "bloques 100x100: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;


    vvi A1000(1000, vi(1000, 1));
    vvi B1000(1000, vi(1000, 2));

    t1 = std::chrono::high_resolution_clock::now();
    vvi C1000 = multiply(A1000, B1000);
    t2 = std::chrono::high_resolution_clock::now();
    cout << "clásica 1000x1000: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;

    t1 = std::chrono::high_resolution_clock::now();
    vvi C1000b = multiplyBlocks(A1000, B1000, 50);
    t2 = std::chrono::high_resolution_clock::now();
    cout << "bloques 1000x1000: "
         << std::chrono::duration<double>(t2 - t1).count() << " segundos" << endl;

    return 0;
}
