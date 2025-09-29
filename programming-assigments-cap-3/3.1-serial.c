#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int Find_bin(float value, float bin_maxes[], int bin_count, float min_meas) {
  // Caso especial: bin 0
  if (value < bin_maxes[0] && value >= min_meas) {
    return 0;
  }
  for (int b = 1; b < bin_count; b++) {
    if (value >= bin_maxes[b - 1] && value < bin_maxes[b]) {
      return b;
    }
  }
  // Si cae exactamente en el maximo (caso borde), va al último bin
  return bin_count - 1;
}

int main(int argc, char **argv) {

  int world_size;
  int world_rank;

  // Input
  int data_count = 20;
  float data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,
                  4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};
  float min_meas = 0.0, max_meas = 5.0;
  int bin_count = 5;

  // Output
  float bin_width = (max_meas - min_meas) / bin_count;
  float *bin_maxes = malloc(bin_count * sizeof(float));
  int *bin_counts = calloc(bin_count, sizeof(int));

  for (int b = 0; b < bin_count; b++) {
    bin_maxes[b] = min_meas + bin_width * (b + 1);
  }

  for (int i = 0; i < data_count; i++) {
    int bin = Find_bin(data[i], bin_maxes, bin_count, min_meas);
    bin_counts[bin]++;
  }

  for (int b = 0; b < bin_count; b++) {
    printf("Bin %d (max %.2f): %d\n", b, bin_maxes[b], bin_counts[b]);
  }

  free(bin_maxes);
  free(bin_counts);

  // MPI_Init(NULL, NULL);
  //
  // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  //
  //
  //
  // MPI_Finalize();
}
