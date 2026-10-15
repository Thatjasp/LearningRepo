// Name: Jose Salinas
// Course: CS 6370
// Professor: Meilin Liu
// Date: 9/15/2026
// Project Name: Project 1 Matrix Mult

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <time.h>

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

void MatrixMulOnHost(int *M, int *N, int *P, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      int sum = 0;
      for (int k = 0; k < width; ++k) {
        int a = M[i * width + k];
        int b = N[k * width + j];
        sum += a * b;
      }
      P[i * width + j] = sum;
    }
  }
}

__global__ void MatrixMulOnDevice(int *M_d, int *N_d, int *P_d, int width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int index = row * width + col;

  if (row < width && col < width) {
    int sum = 0;
    for (int k = 0; k < width; ++k) {
      int a = M_d[row * width + k];
      int b = N_d[k * width + col];
      sum += a * b;
    }
    P_d[index] = sum;
  }
}

int main() {
  int threadBlockSize = 16;
  int matrixWidth = 1024;
  std::cout << "Enter Thread Block Size: ";
  std::cin >> threadBlockSize;

  std::cout << "Enter Matrix Size: ";
  std::cin >> matrixWidth;
  int matrixHeight = matrixWidth;
  int *a, *b, *c, *d;

  a = (int *)malloc(sizeof(int) * matrixHeight * matrixWidth);
  b = (int *)malloc(sizeof(int) * matrixHeight * matrixWidth);
  c = (int *)malloc(sizeof(int) * matrixHeight * matrixWidth);

  d = (int *)malloc(sizeof(int) * matrixHeight * matrixWidth);

  int init = 1325;
  int index;
  for (int i = 0; i < matrixHeight; ++i) {
    for (int j = 0; j < matrixWidth; ++j) {
      init = 3125 * init % 65536;
      index = (i * matrixWidth) + j;
      a[index] = (init - 32768) / 6553;
      b[index] = (init % 1000);
    }
  }
  int *a_d, *b_d, *c_d;
  cudaMalloc((void **)&a_d, sizeof(int) * matrixHeight * matrixWidth);
  cudaMalloc((void **)&b_d, sizeof(int) * matrixHeight * matrixWidth);
  cudaMalloc((void **)&c_d, sizeof(int) * matrixHeight * matrixWidth);

  cudaMemcpy(a_d, a, sizeof(int) * matrixHeight * matrixWidth,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(int) * matrixHeight * matrixWidth,
             cudaMemcpyHostToDevice);

  printf("THREAD_BLOCK_SIZE: %d x %d\n", threadBlockSize, threadBlockSize);
  printf("MATRIX_SIZE: %d x %d\n", matrixHeight, matrixWidth);
  // how many threads in a block
  dim3 dimBlock(threadBlockSize, threadBlockSize, 1);

  dim3 dimGrid(ceil(matrixWidth / double(dimBlock.x)),
               ceil(matrixHeight / double(dimBlock.y)), 1);
  printf("GRID_SIZE: x: %d y: %d\n", dimGrid.x, dimGrid.y);

  clock_t diff;
  clock_t start = clock();
  MatrixMulOnDevice<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, matrixWidth);
  cudaDeviceSynchronize();
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;

  printf("GPU time taken %d seconds %d milliseconds\n", msec / 1000,
         msec % 1000);
  cudaMemcpy(d, c_d, matrixHeight * matrixWidth * sizeof(int),
             cudaMemcpyDeviceToHost);

  start = clock();
  MatrixMulOnHost(a, b, c, matrixWidth);
  diff = clock() - start;
  msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("CPU time taken %d seconds %d milliseconds\n", msec / 1000,
         msec % 1000);

  if (matrixHeight <= 8) {
    printf("All Matrix Elements in GPU Matrix\n");
    printWholeMatrix(c, matrixWidth);
    printf("All Matrix Elements in CPU Matrix\n");
    printWholeMatrix(d, matrixWidth);

  } else {
    printf("First 20 elements in GPU Matrix\n");
    print20Elements(d, matrixHeight * matrixWidth);
    printf("First 20 elements in CPU Matrix\n");
    print20Elements(c, matrixHeight * matrixWidth);
  }

  verify(c, d, matrixWidth);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  free(a);
  free(b);
  free(c);
  free(d);
}
