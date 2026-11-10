// Name: Jose Salinas
// Course: CS 6370
// Professor: Meilin Liu
// Date: 10/5/2026
// Project Name: Project 2 Tiled Matrix Mult

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <time.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

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
  __shared__ int Ns_d[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Ms_d[TILE_WIDTH][TILE_WIDTH];
  int output_row = TILE_WIDTH * blockIdx.y + threadIdx.y;
  int output_col = TILE_WIDTH * blockIdx.x + threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tempVal = 0;
  for (int phase = 0; phase < width / TILE_WIDTH; ++phase) {
    // Right Matrix
    int right_col = output_col;
    int right_row = (phase * TILE_WIDTH) + ty;
    // Left Matrix
    int left_row = output_row;
    int left_col = (phase * TILE_WIDTH) + tx;

    Ms_d[ty][tx] = M_d[left_row * width + left_col];
    Ns_d[ty][tx] = N_d[right_row * width + right_col];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      tempVal += Ms_d[ty][k] * Ns_d[k][tx];
    }
    __syncthreads();
  }
  P_d[output_row * width + output_col] = tempVal;
}

int main() {
  int threadBlockSize = TILE_WIDTH;
  int matrixWidth = 0;

  std::cout << "Enter Matrix Size: ";
  std::cin >> matrixWidth;
  int matrixHeight = matrixWidth;

  if (matrixWidth < 1) {
    return -1;
  }
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
      a[index] = (init - 100) % 6553;
      b[index] = (init % 251);
    }
  }

  printf("THREAD_BLOCK_SIZE: %d x %d\n", threadBlockSize, threadBlockSize);
  printf("MATRIX_SIZE: %d x %d\n", matrixHeight, matrixWidth);

  dim3 dimBlock(threadBlockSize, threadBlockSize, 1);

  dim3 dimGrid(ceil(matrixWidth / double(dimBlock.x)),
               ceil(matrixHeight / double(dimBlock.y)), 1);
  printf("GRID_SIZE: x: %d y: %d\n", dimGrid.x, dimGrid.y);

  cudaEvent_t start_d, stop_d;
  cudaEventCreate(&start_d);
  cudaEventCreate(&stop_d);

  int *a_d, *b_d, *c_d;

  cudaMalloc((void **)&a_d, sizeof(int) * matrixHeight * matrixWidth);
  cudaMalloc((void **)&b_d, sizeof(int) * matrixHeight * matrixWidth);
  cudaMalloc((void **)&c_d, sizeof(int) * matrixHeight * matrixWidth);

  cudaEventRecord(start_d, 0);

  cudaMemcpy(a_d, a, sizeof(int) * matrixHeight * matrixWidth,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(int) * matrixHeight * matrixWidth,
             cudaMemcpyHostToDevice);

  MatrixMulOnDevice<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, matrixWidth);
  cudaDeviceSynchronize();
  cudaMemcpy(d, c_d, matrixHeight * matrixWidth * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_d, 0);
  cudaEventSynchronize(stop_d);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_d, stop_d);
  printf("GPU time taken %f milliseconds\n", milliseconds);

  clock_t start = clock();
  clock_t diff;
  MatrixMulOnHost(a, b, c, matrixWidth);
  diff = clock() - start;
  float msec = static_cast<float>(diff) * 1000 / CLOCKS_PER_SEC;
  printf("CPU time taken %d milliseconds\n", msec);

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
