#include <iostream>

#define MAX 1000

int main (int argc, char *argv[]) {

  double A[MAX][MAX], x[MAX], y[MAX];

  // Initializae A and x, assign y = 0

  // First pair of loops
  for (int i = 0; i < MAX; i++)
    for (int j = 0; j < MAX; j++)
      y[i] += A[i][j] * x[j];

  // Assign y = 0
  for (int j = 0; j < MAX; j++)
    for (int i = 0; i < MAX; i++)
      y[i] += A[i][j] * x[j];

  return 0;
}
