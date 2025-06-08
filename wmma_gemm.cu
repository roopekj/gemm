#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Input matrix dimensions
#define M_DIM 16384
#define N_DIM 16384
#define K_DIM 16384

// Each WMMA operation works on a 16x16 tile with a K-dimension of 16.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmmaKernel(const half *A, const half *B, float *C) {
  const int lda = K_DIM; // row-major
  const int ldb = K_DIM; // column-major
  const int ldc = N_DIM; // row-major

  // The grid is organized in warp tiles, where each warp covers one 16x16 tile
  int warpM = blockIdx.y;
  int warpN = blockIdx.x;

  // Boundary check (here M_DIM and N_DIM are multiples of 16, so this doesn't
  // really matter)
  if (warpM * WMMA_M >= M_DIM || warpN * WMMA_N >= N_DIM)
    return;

  // Declare fragments for a 16x16 tile
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      aFrag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      bFrag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;

  // Initialize the output to zero
  wmma::fill_fragment(accFrag, 0.0f);

  // Loop over the K dimension, biting off a WMMA_K sized chunk each time
  for (int k = 0; k < K_DIM / WMMA_K; ++k) {
    // This gets a bit confusing so let's clarify with comments...

    // A is row-major, so:
    // * A[i][j] is stored at A[i * lda + j]
    // * Starting "row" index (i) is warpM * WMMA_M
    // * Starting "column" index (j) is k * WMMA_K
    const half *tileA = A + warpM * WMMA_M * lda + k * WMMA_K;

    // B is column-major, so everything is flipped:
    // * B[i][j] is stored at B[i + j * ldb]
    // * Starting "row" index (i) is k * WMMA_K
    // * Starting "column" index (j) is warpN * WMMA_N
    const half *tileB = B + k * WMMA_K + warpN * WMMA_N * ldb;

    // Load the tiles into WMMA fragments.
    wmma::load_matrix_sync(aFrag, tileA, lda);
    wmma::load_matrix_sync(bFrag, tileB, ldb);

    // Perform the MMA using tensor cores.
    wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
  }

  // Store the computed 16x16 tile back to C.
  float *tileC = C + warpM * WMMA_M * ldc + warpN * WMMA_N;
  wmma::store_matrix_sync(tileC, accFrag, ldc, wmma::mem_row_major);
}

int main() {
  size_t bytesA = M_DIM * K_DIM * sizeof(half);
  size_t bytesB = K_DIM * N_DIM * sizeof(half);
  size_t bytesC = M_DIM * N_DIM * sizeof(float);

  // Allocate the required memory on host
  half *h_A = (half *)malloc(bytesA);
  half *h_B = (half *)malloc(bytesB);
  float *h_C = (float *)malloc(bytesC);

  // Initialize A and B to all ones for simplicity
  for (int i = 0; i < M_DIM * K_DIM; i++)
    h_A[i] = __float2half(1.0f);
  for (int i = 0; i < K_DIM * N_DIM; i++)
    h_B[i] = __float2half(1.0f);

  // Allocate the required memory on device
  half *d_A, *d_B;
  float *d_C;
  cudaMalloc((void **)&d_A, bytesA);
  cudaMalloc((void **)&d_B, bytesB);
  cudaMalloc((void **)&d_C, bytesC);

  // Copy contents from host matrices A and B to device
  cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

  // Set up grid and block dimensions
  // Grid:
  //    One block per 16x16 tile of C ==> (16384/16) x (16384/16) = (1024, 1024)
  // Block:
  //    One warp per block -> 32 threads
  dim3 gridDim(N_DIM / WMMA_N, M_DIM / WMMA_M);
  dim3 blockDim(32, 1, 1);

  // PC case turbine activator
  for (int i = 0; i < 10; ++i) {
    wmmaKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
  }

  // Copy the computed matrix C back to host
  cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

  // Since A and B are filled with ones, every element in C should be K_DIM
  bool correct = true;
  for (int i = 0; i < M_DIM * N_DIM; ++i) {
    if (h_C[i] != float(K_DIM)) {
      printf("Error at index %d: %f != %d\n", i, h_C[i], K_DIM);
      correct = false;
      break;
    }
  }
  if (correct)
    printf("Result is correct.\n");

  // Free the device and host memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
