#include <cstdio>

#include <cstdio>

// KERNEL B: thread per data
/* * PROS:
 * - Máximo Paralelismo: Ocupación total de la GPU (N*N hilos), cumple con la masividad.
 * - Coalescencia Perfecta: Los hilos vecinos leen direcciones de memoria contiguas.
 * - Rendimiento: Generalmente el más rápido.
 * - Cada oepracion lo hace en O(1)
 * * CONTRAS:
 * - Al lanzar muchos hilos golpe estos se llenan muy facilmente
 */
__global__ void addMatrixB(float *out, float *a, float *b, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < width) {
    int index = row * width + col;
    out[index] = a[index] + b[index];
  }
}

// KERNEL C: thread per row
/* * PROS:
 * - Coalescencia Buena: El acceso dentro del bucle es secuencial (row*width + i) permitiendo cargas eficientes.
 * * CONTRAS:
 * - Bajo Paralelismo: Solo lanza 'width' hilos. Desperdicia la capacidad masiva de la GPU.
 * - Desbalanceo: Si las filas son muy largas, un solo hilo carga con mucho trabajo secuencial.
 * - Cada operacion lo hace en O(n)
 */
__global__ void addMatrixC(float *out, float *a, float *b, int width) {
  if (int row = blockIdx.x * blockDim.x + threadIdx.x; row < width) {
    for (int i = 0; i < width; i++) {
      int index = row * width + i;
      out[index] = a[index] + b[index];
    }
  }
}

// KERNEL D: thread per column
/* * PROS:
 * * CONTRAS:
 * - Pésima Coalescencia debido a que los hilos leen saltando 'width' posiciones en memoria.
 * - Bajo Paralelismo: Igual que el C subutiliza los núcleos de la GPU.
 */
__global__ void addMatrixD(float *out, float *a, float *b, int width) {
  if (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width) {
    for (int i = 0; i < width; i++) {
      int index = i * width + col;
      out[index] = a[index] + b[index];
    }
  }
}

void matAdd_stub (float* out, float *a, float *b , int n) {

  float *d_out, *d_a, *d_b;
  int size_bytes = n * n * sizeof(float);

  cudaMalloc((void**)&d_out, size_bytes);
  cudaMalloc((void**)&d_a, size_bytes);
  cudaMalloc((void**)&d_b, size_bytes);

  cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_bytes, cudaMemcpyHostToDevice);

  // Kernel Call for B. Exercise
  // dim3 dimBlock(16, 16);
  // dim3 dimGrid(ceil(n/dimBlock.x), ceil(n/dimBlock.y), 1);
  // addMatrixB<<<dimGrid, dimBlock>>>(d_out, d_a, d_b, n);

  // Kernel Call for C. Exercise
  // dim3 dimBlock(256);
  // dim3 dimGrid(ceil(n/dimBlock.x));
  // addMatrixC<<<dimGrid, dimBlock>>>(d_out, d_a, d_b, n);

// Kernel Call for D. Exercise
  dim3 dimBlockD(256);
  dim3 dimGridD(ceil(n/dimBlockD.x));
  addMatrixD<<<dimGridD, dimBlockD>>>(d_out, d_a, d_b, n);

  cudaDeviceSynchronize();
  cudaMemcpy(out, d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_a);
  cudaFree(d_b);

}

int main() {

  int N = 1024;
  float *a = (float *)malloc(sizeof(float) * N * N);
  float *b = (float *)malloc(sizeof(float) * N * N);
  float *out = (float *)malloc(sizeof(float) * N * N);

  for (int i = 0; i < N * N; i++) {
    a[i] = 1;
    b[i] = 2;
    out[i] = 0;
  }

  matAdd_stub(out, a, b, N);

  printf("%f \n", out[0]);
}
