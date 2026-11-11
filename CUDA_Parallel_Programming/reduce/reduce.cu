// Name: Jose Salinas
// Course: CS 6370
// Professor: Meilin Liu
// Date: 10/28/2026
// Project Name: Project 3 Reduce

#include <cstdlib>
#include <iostream>
#include <stdio.h>

#ifndef THREAD_BLOCK_SIZE
#define THREAD_BLOCK_SIZE 64
#endif

#define BLOCK_DIM 1024
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

__global__ void simpleReduce(int *input, int *output) {
  int tx = 2 * threadIdx.x;
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (tx % stride == 0) {
      input[tx] += input[tx + stride];
    }
    __syncthreads();
  }

  if (tx == 0) {
    *output = input[tx];
  }
}

__global__ void reduceMemCoalesced(int *input, int *output) {
  int tx = threadIdx.x;
  for (int stride = blockDim.x; 1 <= stride; stride /= 2) {
    if (tx < stride) {
      input[tx] += input[tx + stride];
    }
    __syncthreads();
  }

  if (tx == 0) {
    *output = input[tx];
  }
}
__global__ void reduceMemCoalescedAndSharedMem(int *input, int *output, int N) {
  unsigned int tx = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  __shared__ int input_ds[2 * THREAD_BLOCK_SIZE];

  if (start + tx < N) {
    input_ds[tx] = input[start + tx];
  } else {
    input_ds[tx] = 0;
  }
  if (start + blockDim.x + tx < N) {
    input_ds[tx + blockDim.x] = input[start + tx + blockDim.x];
  } else {
    input_ds[tx + blockDim.x] = 0;
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
      input_ds[tx] += input_ds[tx + stride];
    }
  }
  if (tx == 0) {
    output[blockIdx.x] = input_ds[0];
  }
}

int CPUSumReduction(int *x, int N) {
  for (int i = 1; i < N; i++)
    x[0] = x[0] + x[i];
  int overallSum = x[0];
  return overallSum;
}

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
// Make this bring back a unique_ptr later
int *createInitializedList(int N) {
  int *A = (int *)malloc(sizeof(int) * N); // N is the size
  int init = 1325;
  for (int i = 0; i < N; i++) {
    init = 3125 * init % 6553;
    A[i] = (init - 1000) % 87;
  }
  return A;
}
// Make this a return unique_ptr later
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
    std::cout << "Length will be " << length << "\n";
  } else if (c == 'n') {
    std::cout << "Input length: ";
    std::cin >> length;
  }
  return length;
}

void printAndTimeGPU(int **input_d, int **output_d, int gridSize, int N) {
  CudaTimedEvents cudaTimedEvent;

  int *curInput = *input_d;
  int *curOutput = *output_d;
  // Start Timing on execution
  printf("THREAD_BLOCK_SIZE: %d\n", THREAD_BLOCK_SIZE);
  printf("Initial Grid Size: %d\n", gridSize);
  cudaTimedEvent.startTimer();
  reduceMemCoalescedAndSharedMem<<<gridSize, THREAD_BLOCK_SIZE>>>(curInput,
                                                                  curOutput, N);
  cudaDeviceSynchronize();
  cudaTimedEvent.stopTimer();

  int curN = gridSize;
  int outputGridSize = gridSize;

  while (outputGridSize > 1) {
    int *temp = curInput;
    curInput = curOutput;
    curOutput = temp;

    curN = outputGridSize;
    outputGridSize = std::ceil(static_cast<double>(outputGridSize) /
                               (2 * THREAD_BLOCK_SIZE));

    cudaTimedEvent.startTimer();
    reduceMemCoalescedAndSharedMem<<<outputGridSize, THREAD_BLOCK_SIZE>>>(
        curInput, curOutput, curN);
    cudaDeviceSynchronize();
    cudaTimedEvent.stopTimer();
  }
  *input_d = curInput;
  *output_d = curOutput;

  cudaTimedEvent.printTime();
}
int printAndTimeCPU(int *input, int N) {
  HostTimedEvents cpuTimedEvent;
  cpuTimedEvent.startTimer();
  int result = CPUSumReduction(input, N);
  cpuTimedEvent.stopTimer();
  cpuTimedEvent.printTime();
  return result;
}
int main() {
  int N = promptListSize();
  int gridSize = std::ceil(static_cast<double>(N) / (2 * THREAD_BLOCK_SIZE));

  int *cuda_output = (int *)malloc(sizeof(int) * gridSize);
  int *input = createInitializedList(N);
  printf("Current Input\n");
  print20Elements(input, N);

  int *input_d = allocateAndReturnIntCudaMem(N);
  int *output_d = allocateAndReturnIntCudaMem(gridSize);

  cudaMemcpy(input_d, input, sizeof(int) * N, cudaMemcpyHostToDevice);

  printAndTimeGPU(&input_d, &output_d, gridSize, N);
  cudaMemcpy(cuda_output, output_d, sizeof(int) * gridSize,
             cudaMemcpyDeviceToHost);

  int reductionResult = printAndTimeCPU(input, N);

  printf("GPU Output\n");
  print20Elements(cuda_output, gridSize);

  printf("CPU Output\n");
  print20Elements(input, N);

  verify(&reductionResult, cuda_output, 1);
  printf("CPU Output: %d, GPU Output: %d\n", reductionResult, cuda_output[0]);

  cudaFree(input_d);
  cudaFree(output_d);

  free(input);
}
