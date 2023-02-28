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

#include "flexflow/ops/masked_fill.h"
#include "flexflow/ops/kernels/masked_fill_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

MaskedFillMeta::MaskedFillMeta(FFHandler handler, MaskedFill const *masked_fill)
    : OpMeta(handler, masked_fill) {
  filled_value = masked_fill->filled_value;
}

namespace Kernels {
namespace MaskedFill {

void forward_kernel_wrapper(MaskedFillMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &mask,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  float filled_value = m->filled_value;
  if (mask.data_type == DT_INT32) {
     Internal::forward_kernel(input.get_float_ptr(),
                             mask.get_int32_ptr(),
                             output.get_float_ptr(),
                             output.domain.get_volume(),
                             filled_value,
                             stream);
  } 
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [MaskedFill] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(MaskedFillMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &mask,
                             GenericTensorAccessorW const &input_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  float filled_value = m->filled_value;
  if (mask.data_type == DT_INT32) {
    Internal::backward_kernel(output_grad.get_float_ptr(),
                              mask.get_int32_ptr(),
                              input_grad.get_float_ptr(),
                              output_grad.domain.get_volume(),
                              filled_value,
                              stream);
  }
}

namespace Internal {

template <typename IndexType>
__global__ void masked_fill_forward(float const *input,
                               IndexType const *mask,
                               float *output,
                               float value) {
  CUDA_KERNEL_LOOP(i, output_size) {
    output[i] = mask[i] == 0 ? input[i] : value;
  }
}

template <typename IndexType>
void forward_kernel(float const *input_ptr,
                    IndexType const *mask_ptr,
                    float *output_ptr,
                    float filled_value,
                    cudaStream_t stream) {
  assert(input_ptr != nullptr);
  assert(mask_ptr != nullptr);
  assert(output_ptr != nullptr);
  masked_fill_forward<IndexType>
      <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
          input_ptr, mask_ptr, output_ptr, filled_value);
}

template <typename IndexType>
__global__ void masked_fill_backward(float const *output_grad,
                                IndexType const *mask,
                                float *input_grad,
                                float filled_value) {
  CUDA_KERNEL_LOOP(i, output_size) {
    input_grad[i] = mask[i] == 0 ? input_grad[i] : filled_value
  }
}

template <typename IndexType>
void backward_kernel(float const *output_grad_ptr,
                     IndexType const *mask_ptr,
                     float *input_grad_ptr,
                     float filled_value,
                     cudaStream_t stream) {
  assert(output_grad_ptr != nullptr);
  assert(input_grad_ptr != nullptr);
  assert(mask_ptr != nullptr);
  masked_fill_backward<IndexType>
      <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
          output_grad_ptr,
          mask_ptr,
          input_grad_ptr,
          filled_value);
}

} // namespace Internal
} // namespace MaskedFill
} // namespace Kernels

}; // namespace FlexFlow
