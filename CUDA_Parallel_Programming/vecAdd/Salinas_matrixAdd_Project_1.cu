// Name: Jose Salinas
// Course: CS 6370
// Professor: Meilin Liu
// Date: 9/15/2026
// Project Name: Project 1 Matrix Add

#include <cstdlib>
#include <iostream>
#include <stdio.h>

void verify(int *arr1, int *arr2, int width) {
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = i * width + j;
      if (arr1[index] != arr2[index]) {
        printf("Failed at index %d, %d!\n", i, j);
        return;
      }
    }
  }
  printf("Test PASSED\n");
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

void printWholeMatrix(int *M, int width) {
  int index = 0;
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      index = i * width + j;
      printf("%d ", M[index]);
    }
    printf("\n");
  }
}

void add_matrix_cpu(int *a, int *b, int *c, int width) {
  int i, j, index;
  for (i = 0; i < width; i++) {
    for (j = 0; j < width; j++) {
      index = i * width + j;
      c[index] = a[index] + b[index];
    }
  }
}

__global__ void add_matrix_gpu(int *a_d, int *b_d, int *c_d, int width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int index = row * width + col;

  if (row < width && col < width) {
    c_d[index] = a_d[index] + b_d[index];
  }
}

int main() {
  int threadBlockSize = 16;
  int N = 1024;

  std::cout << "Enter Thread Block Size: ";
  std::cin >> threadBlockSize;

  std::cout << "Enter Matrix Size: ";
  std::cin >> N;

  int *a, *b, *c, *d;

  a = (int *)malloc(sizeof(int) * N * N);
  b = (int *)malloc(sizeof(int) * N * N);
  c = (int *)malloc(sizeof(int) * N * N);

  d = (int *)malloc(sizeof(int) * N * N);

  int init = 1325;
  int index;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      init = 3125 * init % 65536;
      index = (i * N) + j;
      a[index] = (init - 32768) / 6553;
      b[index] = (init % 1000);
    }
  }
  int *a_d, *b_d, *c_d;
  cudaMalloc((void **)&a_d, sizeof(int) * N * N);
  cudaMalloc((void **)&b_d, sizeof(int) * N * N);
  cudaMalloc((void **)&c_d, sizeof(int) * N * N);

  cudaMemcpy(a_d, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  printf("THREAD_BLOCK_SIZE: %d x %d\n", threadBlockSize, threadBlockSize);
  printf("MATRIX_SIZE: %d x %d\n", N, N);
  // how many threads in a block
  dim3 dimBlock(threadBlockSize, threadBlockSize, 1);

  dim3 dimGrid(ceil(N / double(dimBlock.x)), ceil(N / double(dimBlock.y)), 1);
  printf("GRID_SIZE: x: %d y: %d\n", dimGrid.x, dimGrid.y);

  // clock_t diff;
  // clock_t start = clock();
  add_matrix_gpu<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N);
  cudaDeviceSynchronize();
  // diff = clock() - start;
  // int msec = diff * 1000 / CLOCKS_PER_SEC;

  // printf("GPU time taken %d seconds %d milliseconds\n", msec / 1000,
  //         msec % 1000);
  cudaMemcpy(d, c_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  // start = clock();
  add_matrix_cpu(a, b, c, N);
  // diff = clock() - start;
  // msec = diff * 1000 / CLOCKS_PER_SEC;
  //   printf("CPU time taken %d seconds %d milliseconds\n", msec / 1000,
  //         msec % 1000);
  if (N <= 8) {
    printf("All Matrix Elements in GPU Matrix\n");
    printWholeMatrix(c, N);
    printf("All Matrix Elements in CPU Matrix\n");
    printWholeMatrix(d, N);

  } else {
    printf("First 20 elements in GPU Matrix\n");
    print20Elements(d, N * N);
    printf("First 20 elements in CPU Matrix\n");
    print20Elements(c, N * N);
  }
  verify(c, d, N);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  free(a);
  free(b);
  free(c);
  free(d);
}
