#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int Find_bin(float value, float bin_maxes[], int bin_count, float min_meas) {
  if (value < bin_maxes[0] && value >= min_meas) {
    return 0;
  }
  for (int b = 1; b < bin_count; b++) {
    if (value >= bin_maxes[b - 1] && value < bin_maxes[b]) {
      return b;
    }
  }
  return bin_count - 1;
}

int main(int argc, char **argv) {
  int world_size, world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Proceso %d de %d ejecutándose en %s\n", world_rank, world_size,
         processor_name);

  // Input global
  int data_count = 20;
  float data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,
                  4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};
  float min_meas = 0.0, max_meas = 5.0;
  int bin_count = 5;

  float bin_width = (max_meas - min_meas) / bin_count;
  float *bin_maxes = malloc(bin_count * sizeof(float));
  for (int b = 0; b < bin_count; b++) {
    bin_maxes[b] = min_meas + bin_width * (b + 1);
  }

  // Distribuir datos entre procesos
  int local_count = data_count / world_size;
  int remainder = data_count % world_size;

  if (world_rank < remainder) {
    local_count++;
  }

  float *local_data = malloc(local_count * sizeof(float));

  int *sendcounts = NULL;
  int *displs = NULL;
  if (world_rank == 0) {
    sendcounts = malloc(world_size * sizeof(int));
    displs = malloc(world_size * sizeof(int));
    int offset = 0;
    for (int r = 0; r < world_size; r++) {
      sendcounts[r] = data_count / world_size;
      if (r < remainder) {
        sendcounts[r]++;
      }
      displs[r] = offset;
      offset += sendcounts[r];
    }
  }

  MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT, local_data, local_count,
               MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Histograma local
  int *loc_bin_counts = calloc(bin_count, sizeof(int));
  for (int i = 0; i < local_count; i++) {
    int bin = Find_bin(local_data[i], bin_maxes, bin_count, min_meas);
    loc_bin_counts[bin]++;
  }

  // Histograma global (reduce sum)
  int *bin_counts = calloc(bin_count, sizeof(int));
  MPI_Reduce(loc_bin_counts, bin_counts, bin_count, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (int b = 0; b < bin_count; b++) {
      printf("Bin %d [%.2f, %.2f): %d\n", b,
             (b == 0 ? min_meas : bin_maxes[b - 1]), bin_maxes[b],
             bin_counts[b]);
    }
  }

  free(bin_maxes);
  free(loc_bin_counts);
  free(local_data);
  if (world_rank == 0) {
    free(bin_counts);
    free(sendcounts);
    free(displs);
  }

  MPI_Finalize();
  return 0;
}
