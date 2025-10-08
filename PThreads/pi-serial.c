#include <bits/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  long long n, i;
  double factor = 1.0;
  double sum = 0.0;
  double pi;
  struct timespec start, end;

  n = strtol(argv[1], NULL, 10);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < n; i++, factor = -factor) {
    sum += factor / (2 * i + 1);
  }
  clock_gettime(CLOCK_MONOTONIC, &end); 
  double elapsed_time = (end.tv_sec - start.tv_sec) + 
    (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  printf("(Wall Time): %.9f sec\n", elapsed_time);

  pi = 4.0 * sum;

  printf("Aprox of pi with %lld terms: %.15f\n", n, pi);

  return 0;
}
