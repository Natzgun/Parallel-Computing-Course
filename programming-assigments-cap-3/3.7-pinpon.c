#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int world_rank, world_size;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int cont = 0;
  int partner = (world_rank + 1) % 2;
  while (cont < 4) {
    if (world_rank == cont % 2) {
      cont++;
      MPI_Send(&cont, 1, MPI_INT, partner, 10, MPI_COMM_WORLD);
      printf("Rank %d -- Send --> %d \n", world_rank, cont);

    } else {
      MPI_Recv(&cont, 1, MPI_INT, partner, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Rank %d -- Recive --> %d\n", world_rank, cont);
    }
  
  }


  MPI_Finalize();
  return 0;
}
