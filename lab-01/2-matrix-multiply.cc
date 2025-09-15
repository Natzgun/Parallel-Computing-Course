#include <iostream>
#include <vector>
using std::vector;
using std::cout;
using std::endl;

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
  srand(time(0));

  vvi A10(10, vi(10, 1));
  vvi B10(10, vi(10, 2));

  cout << "Multiplicación clásica 10x10:" << endl;
  vvi C10 = multiply(A10, B10);
  printMatrix(C10);

  cout << "\nMultiplicación por bloques 10x10 (blockSize=2):" << endl;
  vvi C10b = multiplyBlocks(A10, B10, 2);
  printMatrix(C10b);


  vvi A100(100, vi(100, 2));
  vvi B100(100, vi(100, 2));

  cout << "\nMultiplicación clásica 100x100 realizada." << endl;
  vvi C100 = multiply(A100, B100);

  cout << "Multiplicación por bloques 100x100 (blockSize=10) realizada." << endl;
  vvi C100b = multiplyBlocks(A100, B100, 10);

  return 0;
}
