#include <cmath>
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

// We split the result matrix into 128x128 tiles per block.
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64

// The grid is 256x256
// Each block computes a 64x64 region of C, spanning the entire 16384x16384
// matrix. Within a block, its 64x64 tile is made of 4x4 WMMA sub-tiles, each
// of size 16x16. There are 4 warps (128 threads) per block. Each warp handles
// one "row" of the WMMA sub-tiles.
__global__ void wmmaKernel(const half *A, const half *B, float *C) {
  const int lda = K_DIM; // row-major
  const int ldb = K_DIM; // column-major
  const int ldc = N_DIM; // row-major

  // The starting coordinates of this block's tile in matrix C
  int block_row = blockIdx.y * BLOCK_TILE_M;
  int block_col = blockIdx.x * BLOCK_TILE_N;

  // Within the block, launch 4 warps -> 128 threads.
  // Each warp computes a 16-row & 64-column strip of the full 64x64 tile, in
  // eight batches of 16x16 sub-tiles
  int warpId = threadIdx.x / 32; // 0 .. 7

  // Global row offset for what is computed by this warp
  int global_row = block_row + warpId * WMMA_M;

  // Each warp computes 4 WMMA tiles along the column direction, so we need
  // an array of accumulator fragments: one for each of the 4 sub-tiles in this
  // warp's row.
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4];

// Barrel roll
#pragma unroll
  for (int i = 0; i < 4; i++) {
    wmma::fill_fragment(acc_frag[i], 0.0f);
  }

  // Loop over the K dimension, biting off a WMMA_K sized chunk each time
  for (int k_tile = 0; k_tile < K_DIM / WMMA_K; k_tile++) {

    // Each warp first loads its fragment in A
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        a_frag;

    // A is row-major, so:
    // * A[i][j] is stored at A[i * lda + j]
    // * Starting "row" index (i) is the global row computed above
    // * Starting "column" index (j) is k_tile * WMMA_K
    const half *tileA = A + global_row * lda + k_tile * WMMA_K;

    wmma::load_matrix_sync(a_frag, tileA, lda);

// Now, loop over the 4 sub-tiles in columns that this warp is responsible for
#pragma unroll
    for (int i = 0; i < 4; i++) {
      // Global column offset for this WMMA tile
      int global_col = block_col + i * WMMA_N;

      // Load the fragment from B
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                     wmma::col_major>
          b_frag;

      // B is column-major, so everything is flipped:
      // * B[i][j] is stored at B[i + j * ldb]
      // * Starting "row" index is k_tile * WMMA_K
      // * Starting "column" index is the global column computed above
      const half *tileB = B + k_tile * WMMA_K + global_col * ldb;

      wmma::load_matrix_sync(b_frag, tileB, ldb);

      // Perform the MMA
      wmma::mma_sync(acc_frag[i], a_frag, b_frag, acc_frag[i]);
    }
  }

// Because the accumulation is done over K, each warp stores its 4 resulting
// 16x16 tiles
#pragma unroll
  for (int i = 0; i < 4; i++) {
    // The starting pointer in matrix C for this 16x16 sub-tile
    int global_col = block_col + i * WMMA_N;
    float *tileC = C + global_row * ldc + global_col;

    wmma::store_matrix_sync(tileC, acc_frag[i], ldc, wmma::mem_row_major);
  }
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
  //    One block per 64x64 tile of C ==> (16384/64) x (16384/64) = (256,
  //    256)
  // Block:
  //    Four warps per block -> 128 threads
  dim3 gridDim(N_DIM / BLOCK_TILE_N, M_DIM / BLOCK_TILE_M);
  dim3 blockDim(128, 1, 1);

  // PC case turbine activator
  for (int i = 0; i < 10; ++i) {
    wmmaKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
  }

  // Copy the computed matrix C back to host
  cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

  // Since A and B are filled with ones, every element in C should be K_DIM
  bool correct = true;
  for (int i = 0; i < M_DIM * N_DIM; i++) {
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

