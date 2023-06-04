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

#include "flexflow/ops/sampling.h"
#include "flexflow/utils/cuda_helper.h"
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace FlexFlow {

template <typename DT>
__global__ void mask_value_above_top_p(DT *input_ptr,
                                       DT *cumsum_ptr,
                                       float top_p,
                                       int total_eles) {
  CUDA_KERNEL_LOOP(i, total_eles) {
    if ((cumsum_ptr[i] - input_ptr[i]) > static_cast<DT>(top_p)) {
      input_ptr[i] = 0.0;
    }
  }
}

template <typename DT>
__global__ void re_normalized(DT *input_ptr, DT div, int length) {
  CUDA_KERNEL_LOOP(i, length) {
    input_ptr[i] /= div;
  }
}

template <typename DT>
__global__ void find_idx(DT *cumsum_ptr, DT target, int length, int *indices_ptr, int k) {
  CUDA_KERNEL_LOOP(i, length) {
    if(cumsum_ptr[i] >= target){
      indices_ptr[k] = i;
    }
  }
}

/*static*/
template <typename DT>
void Sampling::forward_kernel(SamplingMeta const *m,
                              DT *input_ptr,
                              int *indices_ptr,
                              float top_p,
                              int length,
                              int batch_size,
                              cudaStream_t stream) {
  // 1. sort
  // 2. cumsum
  // how to do it in parallel?
  
  DT *cumsum_ptr;
  checkCUDA(cudaMalloc(&cumsum_ptr, batch_size * length * sizeof(DT)));

  for (int i = 0; i < batch_size; i++) {
    thrust::sort(thrust::device, input_ptr + i * length, input_ptr + (i + 1) * length, thrust::greater<DT>());
    thrust::inclusive_scan(thrust::device, input_ptr + i * length,
                           input_ptr + (i + 1) * length,
                           cumsum_ptr + i * length);
  }

   
  // 3. mask
  int parallelism = batch_size * length;
  mask_value_above_top_p<DT>
      <<<GET_BLOCKS(parallelism),
         min(CUDA_NUM_THREADS, parallelism),
         0,
         stream>>>(input_ptr, cumsum_ptr, top_p, parallelism);
   
  // 4. sum/div
  for (int i = 0; i < batch_size; i++) {
    DT sum = thrust::reduce(thrust::device, input_ptr + i * length,
                            input_ptr + (i + 1) * length);
    parallelism = length;
    re_normalized<DT><<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(input_ptr + i * length, sum, length);
  }
  print_tensor<float>((float *)input_ptr, 100, "sdsdasd");      
  assert(false);
  // 5.multinominal
  for (int i = 0; i < batch_size; i++) {
    parallelism = length;
    DT random = static_cast<DT>(((float)std::rand()) / RAND_MAX);
    thrust::inclusive_scan(thrust::device, input_ptr + i * length,
                           input_ptr + (i + 1) * length,
                           cumsum_ptr + i * length);

                          
    find_idx<DT><<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(cumsum_ptr + i * length, random, length, indices_ptr, i);                 
  }
}

/*static*/
void Sampling::forward_kernel_wrapper(SamplingMeta const *m,
                                      GenericTensorAccessorW const &input,
                                      GenericTensorAccessorW const &indices) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int batch_size = input.domain.get_volume() / length;

  if (input.data_type == DT_HALF) {
    Sampling::forward_kernel<half>(m,
                                   input.get_half_ptr(),
                                   indices.get_int32_ptr(),
                                   m->top_p,
                                   length,
                                   batch_size,
                                   stream);
  } else if (input.data_type == DT_FLOAT) {
    Sampling::forward_kernel<float>(m,
                                    input.get_float_ptr(),
                                    indices.get_int32_ptr(),
                                    m->top_p,
                                    length,
                                    batch_size,
                                    stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[Sampling] forward time = %.2lfms\n", elapsed);
  }
}

SamplingMeta::SamplingMeta(FFHandler handler, Op const *op)
    : OpMeta(handler, op) {}

}; // namespace FlexFlow
