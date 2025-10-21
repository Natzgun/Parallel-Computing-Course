#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 8000
#define N 8000

double matrix[M][N];
double vector[N];
double result[M];

int num_threads;

typedef struct {
  int start_row;
  int end_row;
} ThreadData;

void *multiply_partial(void *arg) {
  ThreadData *data = (ThreadData *)arg;

  for (int i = data->start_row; i < data->end_row; i++) {
    result[i] = 0.0;
    for (int j = 0; j < N; j++) {
      result[i] += matrix[i][j] * vector[j];
    }
  }

  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Uso: %s <num_threads>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  num_threads = atoi(argv[1]);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i][j] = i;
    }
  }
  for (int j = 0; j < N; j++) {
    vector[j] = j + 1;
  }

  pthread_t threads[num_threads];
  ThreadData thread_data[num_threads];

  int rows_per_thread = M / num_threads;
  int extra = M % num_threads;
  int start = 0;

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  for (int t = 0; t < num_threads; t++) {
    int end = start + rows_per_thread + (t < extra ? 1 : 0);
    thread_data[t].start_row = start;
    thread_data[t].end_row = end;
    pthread_create(&threads[t], NULL, multiply_partial, &thread_data[t]);
    start = end;
  }

  for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);

  double elapsed = (end_time.tv_sec - start_time.tv_sec) +
                   (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  printf("Tiempo total de multiplicación: %.6f segundos\n", elapsed);

  return 0;
}
