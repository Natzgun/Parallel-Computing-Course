#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

#define vvi vector<vector<int>>
#define vi vector<int>

#define MAX 20000

int main (int argc, char *argv[]) {

  // Initializae A and x, assign y = 0
  vvi A(MAX, vi(MAX, 1));
  vi x(MAX, 1);
  vi y(MAX, 0);

  // First pair of loops
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < MAX; i++)
    for (int j = 0; j < MAX; j++)
      y[i] += A[i][j] * x[j];
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "First matrix: " << std::chrono::duration<double>(t2 - t1).count() << endl;

  // Assign y = 0
  std::fill(y.begin(), y.end(), 0);
  
  t1 = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < MAX; j++)
    for (int i = 0; i < MAX; i++)
      y[i] += A[i][j] * x[j];
  t2 = std::chrono::high_resolution_clock::now();
  cout << "Second matrix: " << std::chrono::duration<double>(t2 - t1).count() << endl;

  return 0;
}
