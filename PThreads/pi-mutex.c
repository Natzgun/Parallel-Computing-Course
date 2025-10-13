#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

int thread_count;
long long n;
double sum = 0.0;
int flag = 0;
pthread_mutex_t mutex;

void* Thread_sum(void* rank) {
  long my_rank = (long) rank;
  double factor, local_sum;
  long long i;
  long long my_n = n / thread_count;
  long long my_first_i = my_n * my_rank;
  long long my_last_i = my_first_i + my_n;

  if (my_first_i % 2 == 0)
    factor = 1.0;
  else 
    factor = -1.0;
  
  for (i = my_first_i;  i < my_last_i; i++, factor = -factor) {
    local_sum += factor / (2 * i + 1);
  }

  pthread_mutex_lock(&mutex);
  sum += local_sum;
  pthread_mutex_unlock(&mutex);

  return NULL;
}

int main(int argc, char *argv[]) {
  double pi;
  long thread;
  pthread_t *thread_handles;
  struct timespec start, end;
  pthread_mutex_init(&mutex, NULL);

  // n terms of pi
  n = strtol(argv[1], NULL, 10);
  thread_count = strtol(argv[2], NULL, 10);

  thread_handles = malloc(thread_count * sizeof(pthread_t));

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (thread = 0; thread < thread_count; thread++)
    pthread_create(&thread_handles[thread], NULL, Thread_sum, (void *) thread);


  printf("Hello from the main thread: \n");

  for (thread = 0; thread < thread_count; thread++)
    pthread_join(thread_handles[thread], NULL);

  clock_gettime(CLOCK_MONOTONIC, &end); 
  double elapsed_time = (end.tv_sec - start.tv_sec) + 
    (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  printf("(Wall Time): %.9f sec\n", elapsed_time);

  pthread_mutex_destroy(&mutex);

  free(thread_handles);

  pi = 4.0 * sum;

  printf("Aprox of pi with %lld terms: %.15f\n", n, pi);

  return 0;
}
