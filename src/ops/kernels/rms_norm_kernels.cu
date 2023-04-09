/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

RMSNormMeta::RMSNormMeta(FFHandler handler, RMSNorm const *rms)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  alpha = 1.0f;
  beta = 0.0f;

  in_dim = rms->data_dim;
  batch_size = rms->effective_batch_size;
  num_elements = in_dim * batch_size;

  checkCUDA(cudaMalloc(&rms_ptr, batch_size * sizeof(float)));
  checkCUDA(cudaMalloc(&norm_ptr, num_elements * sizeof(float)));
}

namespace Kernels {
namespace RMSNorm {

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value,
                                            unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void
    RowwiseRootMeanSquareKernel(int64_t N, T eps, T const *X, T *rms) {
  __shared__ T v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T sum = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum += static_cast<T>(X[index]) * static_cast<T>(X[index]);
  }
  sum = BlockReduceSum<T>(sum, v_shared); // use BlockReduceSum() to sum X_ij^2
  if (threadIdx.x == 0) {
    rms[i] = 1.0 / (sqrt((sum / static_cast<T>(N)) + static_cast<T>(eps)));
  }
}

template <typename T>
__global__ void NormKernel(int64_t N, T const *X, T const *rstd, T *Y) {
  using T_ACC = T;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    Y[index] = static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(rstd[i]);
  }
}

__global__ void elewise_apply_weights(int64_t batch_size,
                                      int64_t in_dim,
                                      float const *norm,
                                      float const *weights,
                                      float *output) {
  CUDA_KERNEL_LOOP(i, batch_size * in_dim) {
    output[i] = norm[i] * weights[i % 4096];
  }
}

void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  RowwiseRootMeanSquareKernel<float>
      <<<m->batch_size, kCUDABlockReduceNumThreads, 0, stream>>>(
          m->in_dim, m->eps, input.get_float_ptr(), m->rms_ptr);

  NormKernel<float><<<m->batch_size, kCUDANumThreads, 0, stream>>>(
      m->in_dim, input.get_float_ptr(), m->rms_ptr, m->norm_ptr);
  int parallelism = m->batch_size * m->in_dim;
  elewise_apply_weights<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(m->batch_size,
                                    m->in_dim,
                                    m->norm_ptr,
                                    weight.get_float_ptr(),
                                    output.get_float_ptr());
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow