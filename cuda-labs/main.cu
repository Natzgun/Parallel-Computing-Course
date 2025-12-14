#include <cstdio>

// Aqui definimos del tamaño del vector
#define N 100000

/* Con  la directiva global nos permite usar tanto host como el device que
 * es la GPU, en  esta funcion le pasamos los vectores para calcular
 * su producto punto
 */
__global__ void dotProductVector(int *out, int *a, int *b, int n) {
  /* Caluilo del indice global del hilo en donde:
   * blockIdx es la popsición del thread block
   * blockDim tamaño del thread block en este caso 256
   * threadIdx posicion del thread dentro del thread block
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Aqui nos aseguramos que si hay threads sobrantes estos no se accedan para
  // no cometer un indebido acceso a la memoria
  if (i < n) {
    out[i] = a[i] * b[i];
  }
}

int main() {
  // printf("Hello World desde la CPU!\n");

  int *a, *b, *out;
  int *d_a, *d_b, *d_out;

  /* Como el kernel solo le vamos a mandar a calcular
   * hacemos primero la reserva de memoria en el host
   * para poder inicializarla y posteriormente mandarla
   * al kernel de cuda
   */
  a = (int*)malloc(sizeof(int) * N);
  b = (int*)malloc(sizeof(int) * N);
  out = (int*)malloc(sizeof(int) * N);

  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  /* AL igual que en el host se reserva memoria en la RAM del host,
   * para cuda necesitamos reservar la memoria en el GPU
   */
  cudaMalloc(&d_a, N * sizeof(int));
  cudaMalloc(&d_b, N * sizeof(int));
  cudaMalloc(&d_out, N * sizeof(int));

  /* Los valores ya inicializados de los vectores, lo copiamos desde
   * el host hacia el device
   */
  cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  /* Para que todas las operaciones usen un threado apropiadamente
  * se calcula la cantidad de thread block que vamos a usar y la cantidad
  * de threads por cada uno de estos, como en este caso vamos a hacer N
  * operaciones para cubrir la totalidad de operaciones usamos el techo
  * para dar garantia de que cada operacion ocupara un thread respectivamente
   */
  // int num_blocks = (N + 256 - 1) / 256;

  // Se hace la llamada al kernel
  // dotProductVector<<<num_blocks, 256>>>(d_out, d_a, d_b, N);

  /* Usando grids y bloques */
  dim3 dimGrid(ceil(N/256.0), 1 ,1);
  dim3 dimBlock(256, 1, 1);

  dotProductVector<<<dimGrid, dimBlock>>>(d_out, d_a, d_b, N);

  /* Como el kernel se ejecuta de forma asincrónica en la GPU,
   * podemos esperar a que la GPU termine su trabajo para obtener
   * el valor del producto punto de los vectores
   */
  cudaDeviceSynchronize();

  // Pasamos los datos calculados en el vector d_out del device al host
  cudaMemcpy(out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

  // Realizamos la suma de los resulados
  int scalar_result = 0;
  for (int i = 0; i < N; i++) {
    scalar_result += out[i];
  }
  printf("Scalar result: %d", scalar_result);

  // Liberamos memoria tanto de la reserva en el device como el host
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  free(a);
  free(b);

  return 0;
}
