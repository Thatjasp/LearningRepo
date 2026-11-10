#include <stdio.h>
void printDevProp(cudaDeviceProp devProp) {
  printf("Major revision number:         %d\n", devProp.major);
  printf("Minor revision number:         %d\n", devProp.minor);
  printf("Name:                          %s\n", devProp.name);
  printf("Total global memory:           %u\n", devProp.totalGlobalMem);
  printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", devProp.regsPerBlock);
  printf("Warp size:                     %d\n", devProp.warpSize);
  printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
  printf("Max blocks per SM:  %d\n", devProp.maxBlocksPerMultiProcessor);
  printf("Max threads per SM  %d\n", devProp.maxThreadsPerMultiProcessor);
  printf("Total constant memory:         %u\n", devProp.totalConstMem);
  printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
  return;
}

void printHostCudaProps() {
  int devCount;
  cudaGetDeviceCount(&devCount);

  for (int devIndex = 0; devIndex < devCount; ++devIndex) {
    cudaDeviceProp devProperties;
    cudaGetDeviceProperties(&devProperties, devIndex);
    printDevProp(devProperties);
  }
}
int main() { printHostCudaProps(); }
