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
#define MAX_N ((2 * BLOCK_SIZE) * (2 * BLOCK_SIZE))
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
  if (tx + start < N) {
    sharedInput[tx] = input[tx + start];
  } else {
    sharedInput[tx] = 0;
  }
  if (tx + start + blockDim.x < N) {
    sharedInput[tx + blockDim.x] = input[start + tx + blockDim.x];
  } else {
    sharedInput[tx + blockDim.x] = 0;
  }

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

  if (tx + start < N) {
    input[start + tx] = sharedInput[tx];
  }
  if (tx + start + blockDim.x < N) {
    input[start + tx + blockDim.x] = sharedInput[tx + blockDim.x];
  }
  __syncthreads();
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
  void printExecTime() {
    float milliseconds = time();
    printf("GPU time taken %f milliseconds\n", milliseconds);
  }
  void printMemTime() {
    float milliseconds = time();
    printf("GPU Memory time taken %f milliseconds\n", milliseconds);
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

__global__ void addNumberToSection(int *input_d, int *prevNum, int N) {
  // We want to start at index 1
  int inputIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (inputIdx < N) {
    input_d[inputIdx] += prevNum[0];
  }
  if (inputIdx + blockDim.x < N) {
    input_d[inputIdx + blockDim.x] += prevNum[0];
  }
}
__global__ void addScanBlock(int *input_d, int *output_d, int N) {
  // We want to start at index 1
  int start = BLOCK_SIZE * 2;
  int inputIdx = (blockIdx.x * (blockDim.x * 2)) + start + threadIdx.x;

  if (inputIdx < N) {
    input_d[inputIdx] += output_d[blockIdx.x];
  }
  if (inputIdx + blockDim.x < N) {
    input_d[inputIdx + blockDim.x] += output_d[blockIdx.x];
  }
}
__global__ void addNumToInputBlock(int *input_d, int *num_d, int N) {
  int index = (2 * blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    input_d[index] += num_d[0];
  }
  if (index + blockDim.x < N) {
    input_d[index + blockDim.x] += num_d[0];
  }
}
void printAndTimeGPU(int *input_d, int *output_d, int gridSize, int N) {
  CudaTimedEvents cudaTimedEvent;

  // int gridSize = std::ceil(static_cast<double>(N) / (2 * BLOCK_SIZE));
  // what is the max block
  // (BLOCK SIZE * 2) * (BLOCK_SIZE * 2) = MAX_N?
  int num_of_sections = std::ceil(static_cast<double>(N) / MAX_N);
  int i = 0;
  int *gridOutput = allocateAndReturnIntCudaMem(1);
  int *prevLastOutput = allocateAndReturnIntCudaMem(1);
  int defaultVal = 0;
  cudaMemcpy((void **)prevLastOutput, &defaultVal, sizeof(int),
             cudaMemcpyHostToDevice);

  printf("THREAD_BLOCK %d\n", BLOCK_SIZE);

  while (i < num_of_sections) {
    int curNum = MAX_N;
    int curGridSize = 2 * BLOCK_SIZE;
    if (i + 1 == num_of_sections && N % MAX_N != 0) {
      curNum = N % MAX_N;
      curGridSize = std::ceil(static_cast<double>(curNum) / (2 * BLOCK_SIZE));
    }
    int curOffset = i * MAX_N;

    printf("Grid Size for section %d:  %d\n", i, curGridSize);
    cudaTimedEvent.startTimer();
    BrentKungScan<<<curGridSize, BLOCK_SIZE>>>(input_d + curOffset, output_d,
                                               curNum);
    cudaDeviceSynchronize();
    cudaTimedEvent.stopTimer();

    if (curGridSize != 1) {

      cudaTimedEvent.startTimer();
      BrentKungScan<<<1, BLOCK_SIZE>>>(output_d, gridOutput, curGridSize);
      cudaDeviceSynchronize();
      cudaTimedEvent.stopTimer();

      cudaTimedEvent.startTimer();
      addScanBlock<<<curGridSize - 1, BLOCK_SIZE>>>(input_d + curOffset,
                                                    output_d, curNum);
      cudaDeviceSynchronize();
      if (i > 0) {
        addNumToInputBlock<<<curGridSize, BLOCK_SIZE>>>(
            input_d + curOffset, input_d + curOffset - 1, curNum);
        cudaDeviceSynchronize();
      }
      cudaTimedEvent.stopTimer();
    }
    i++;
  }
  cudaTimedEvent.printExecTime();
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
  printf("Current Input\n");
  print20Elements(input, N);

  CudaTimedEvents cudaTimedEvent;
  int *input_d = allocateAndReturnIntCudaMem(N);
  int *output_d = allocateAndReturnIntCudaMem(gridSize);

  cudaTimedEvent.startTimer();
  cudaMemcpy(input_d, input, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaTimedEvent.stopTimer();

  printAndTimeGPU(input_d, output_d, gridSize, N);
  printAndTimeCPU(input, output, N);

  cudaTimedEvent.startTimer();
  cudaMemcpy(cuda_output, input_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaTimedEvent.stopTimer();

  cudaTimedEvent.printMemTime();

  printf("CPU Output\n");
  print20Elements(output, N);

  printf("GPU Output\n");
  print20Elements(output, N);

  verify(output, cuda_output, N);

  cudaFree(input_d);
  cudaFree(output_d);

  free(input);
}
