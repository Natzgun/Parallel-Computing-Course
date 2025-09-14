#include <iostream>
#include <vector>
using std::vector;

#define vvi vector<vector<int>>
#define vi vector<int>

vvi multiply(vvi &a, vvi &b) {
  int n = a.size(), m = b[0].size();
  vvi result(n, vi(m, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < b.size(); k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

int main(int argc, char *argv[]) { return 0; }
