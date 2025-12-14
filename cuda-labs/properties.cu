#include <cstdio>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("¡No se encontraron dispositivos CUDA!\n");
    return 1;
  }

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\n--- Información del Dispositivo %d ---\n", dev);
    printf("Nombre:                        %s\n", deviceProp.name);
    printf("Capacidad de Cómputo:          %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Multiprocesadores (SMs):       %d\n", deviceProp.multiProcessorCount);

    printf("\n[Memoria]\n");
    printf("Memoria Global Total:          %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memoria Compartida por Bloque: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registros por Bloque:          %d\n", deviceProp.regsPerBlock);

    printf("\n[Hilos y Bloques]\n");
    printf("Max Hilos por Bloque:          %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Dimensiones de Bloque:     [%d, %d, %d]\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max Dimensiones de Grid:       [%d, %d, %d]\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Warp Size:                     %d\n", deviceProp.warpSize);
  }

  return 0;
}