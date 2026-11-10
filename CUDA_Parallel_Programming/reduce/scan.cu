// Name: Jose Salinas
// Course: CS 6370
// Professor: Meilin Liu
// Date: 10/28/2026
// Project Name: Project 3 Scan

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <time.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__global__ void simpleKuggeStoneScan(int *input, int *output) {
  int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    int index = (tx + 1) * 2 * stride - 1;
    if (index >= stride) {
      input[index] += input[index - stride];
    }
    __syncthreads();
  }

  if (tx == 0) {
    *output = input[tx];
  }
}

//__global__ void kuggeStoneSharedMemScan(int *input, int *output,
//                                        unsigned int N) {
//  __shared__ int sharedInput[BLOCK_SIZE];
//  int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
//  if (tx < N) {
//    sharedInput[threadIdx.x] = input[tx];
//  } else {
//    sharedInput[threadIdx.x] = 0;
//  }
//
//  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
//    __syncthreads();
//    int index = (tx + 1) * 2 * stride - 1;
//    if (index >= stride) {
//      sharedInput[index] += sharedInput[index - stride];
//    }
//  }
//
//  if (tx == 0) {
//    *output = input[tx];
//  }
//}
__global__ void BrentKungScan(int *input, int *output, unsigned int N) {
  __shared__ int sharedInput[2 * BLOCK_SIZE];

  int start = 2 * (blockDim.x * blockIdx.x);
  int tx = threadIdx.x;

  sharedInput[tx] = input[tx + start];
  sharedInput[tx + blockDim.x] = input[start + tx + blockDim.x];

  __syncthreads();

  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    int curIndex =
        (threadIdx.x + 1) * 2 * stride - 1; // this will get the 0 index
    if (curIndex < 2 * BLOCK_SIZE) {
      sharedInput[curIndex] += sharedInput[curIndex - stride];
    }
    __syncthreads();
  }

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    int curIndex = (threadIdx.x + 1) * 2 * stride - 1;
    if (curIndex + stride < 2 * BLOCK_SIZE) {
      sharedInput[curIndex + stride] += sharedInput[curIndex];
    }
    __syncthreads();
  }
  __syncthreads();
  input[start + tx] = sharedInput[tx];
  input[start + tx + blockDim.x] = sharedInput[tx + blockDim.x];
  if (tx == BLOCK_SIZE - 1) {
    output[blockIdx.x] = sharedInput[2 * BLOCK_SIZE - 1];
  }
}
void simpleCPUScan(int *input, int *output, int N) {
  output[0] = input[0];
  for (int i = 1; i < N; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}
// void simpleCPUScan(int *x, int N) {
//   for (int i = 1; i < N; i++)
//     x[i] = x[i - 1] + x[i];
// }

void print20Elements(int *M, int width) {
  if (width < 20) {
    return;
  }
  for (int i = 0; i < 20; ++i) {
    printf("%d ", M[i]);
  }
  printf("\n");
}

void verify(int *arr1, int *arr2, int width) {
  for (int i = 0; i < width; ++i) {
    if (arr1[i] != arr2[i]) {
      printf("Failed at index %d!\n", i);
      printf("Arr1 val %d, Arr2 val %d!\n", arr1[i], arr2[i]);
      return;
    }
  }
  printf("Test PASSED\n");
}

int *createInitializedList(int N) {
  int *A = (int *)malloc(sizeof(int) * N); // N is the size
  int init = 1325;
  for (int i = 0; i < N; i++) {
    init = 3125 * init % 6553;
    A[i] = (init - 1000) % 87;
  }
  return A;
}
int *allocateAndReturnIntCudaMem(int N) {
  int *intPtr;
  cudaMalloc((void **)&intPtr, sizeof(int) * N);
  return intPtr;
}

int promptListSize() {
  char c = 'x';
  int length;
  while (c != 'y' && c != 'n') {
    std::cout << "Would you like to input factored length (NxM)? [y/n] ";
    std::cin >> c;
  }
  if (c == 'y') {
    int width, height;
    std::cout << "Input width: ";
    std::cin >> width;
    std::cout << "Input height: ";
    std::cin >> height;
    length = width * height;
  } else if (c == 'n') {
    std::cout << "Input length: ";
    std::cin >> length;
  }
  std::cout << "Length will be " << length << "\n";
  return length;
}

struct CudaTimedEvents {
  cudaEvent_t start_d;
  cudaEvent_t stop_d;
  float milliseconds = 0;
  CudaTimedEvents() {
    cudaEventCreate(&start_d);
    cudaEventCreate(&stop_d);
  }
  void startTimer() { cudaEventRecord(start_d, 0); }
  void stopTimer() {
    cudaEventRecord(stop_d, 0);
    cudaEventSynchronize(stop_d);
    float appendMilliseconds = 0;
    cudaEventElapsedTime(&appendMilliseconds, start_d, stop_d);
    milliseconds += appendMilliseconds;
  }
  float time() { return milliseconds; }
  void printTime() {
    float milliseconds = time();
    printf("GPU time taken %f milliseconds\n", milliseconds);
  }
};
struct HostTimedEvents {
  clock_t start;
  clock_t diff;
  void startTimer() { start = clock(); }
  void stopTimer() { diff = clock() - start; }
  float time() { return static_cast<float>(diff) * 1000 / CLOCKS_PER_SEC; }
  void printTime() {
    float milliseconds = time();
    printf("CPU time taken %f milliseconds\n", milliseconds);
  }
};

__global__ void vecAddNum(int *input_d, int *output, int N) {
  // We want to start at index 1
  int start = BLOCK_SIZE * 2;
  int inputIdx = (blockIdx.x * BLOCK_SIZE * 2) + start + threadIdx.x;

  if (inputIdx < N) {
    input_d[inputIdx] += output[blockIdx.x];
  }
}
void printAndTimeGPU(int *input_d, int *output_d, int gridSize, int N) {
  int *gridOutput = allocateAndReturnIntCudaMem(gridSize);
  CudaTimedEvents cudaTimedEvent;
  // Start Timing on execution
  cudaTimedEvent.startTimer();
  BrentKungScan<<<gridSize, BLOCK_SIZE>>>(input_d, output_d, N);
  cudaDeviceSynchronize();

  int outputGridSize = static_cast<double>(gridSize) / (2 * BLOCK_SIZE);

  if (gridSize > 1) {
    BrentKungScan<<<outputGridSize, BLOCK_SIZE>>>(output_d, gridOutput,
                                                  gridSize);

    vecAddNum<<<gridSize - 1, BLOCK_SIZE>>>(input_d, gridOutput, N);
  }

  cudaTimedEvent.stopTimer();
  cudaTimedEvent.printTime();
}
void printAndTimeCPU(int *input, int *output, int N) {
  HostTimedEvents cpuTimedEvent;
  cpuTimedEvent.startTimer();
  simpleCPUScan(input, output, N);
  cpuTimedEvent.stopTimer();
  cpuTimedEvent.printTime();
}

int main() {
  int N = promptListSize();
  int gridSize = std::ceil(static_cast<double>(N) / (2 * BLOCK_SIZE));
  int *cuda_output = (int *)malloc(sizeof(int) * N);

  int *input = createInitializedList(N);
  int *output = (int *)malloc(sizeof(int) * N);
  print20Elements(input, N);
  int *input_d = allocateAndReturnIntCudaMem(N);
  int *output_d = allocateAndReturnIntCudaMem(gridSize);
  cudaMemcpy(input_d, input, sizeof(int) * N, cudaMemcpyHostToDevice);

  printAndTimeGPU(input_d, output_d, gridSize, N);

  printAndTimeCPU(input, output, N);

  cudaMemcpy(cuda_output, input_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

  verify(output, cuda_output, N);

  cudaFree(input_d);
  cudaFree(output_d);

  free(input);
}
