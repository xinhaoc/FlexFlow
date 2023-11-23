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
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {
namespace IncMultiHeadAttention {

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int qkv_weight_size,
                                  int oProjSize) {
  CUDA_KERNEL_LOOP(i, num_tokens * oProjSize) {
    int bias_idx = qkv_weight_size + i % oProjSize;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qProjSize,
                                    int kProjSize,
                                    int vProjSize,
                                    int global_num_q_heads,
                                    int num_q_heads,
                                    bool scaling_query,
                                    float scaling_factor,
                                    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size * QKV_WEIGHT_NUM) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    // int qkv_index = i / (num_tokens * qProjSize) % 3;

    int token_idx = i / (hidden_size * QKV_WEIGHT_NUM);
    size_t in_token_idx = i - token_idx * hidden_size * QKV_WEIGHT_NUM;

    int qkv_index = in_token_idx / hidden_size;

    int proj_size = qkv_index == 0 ? qProjSize : kProjSize;

    int head_idx =
        (in_token_idx - qkv_index * num_q_heads * proj_size) / proj_size;
    int global_head_idx = head_idx + shard_id * num_q_heads;

    size_t pre_length =
        qkv_index == 0
            ? 0
            : (qkv_index == 1 ? qProjSize * global_num_q_heads
                              : qProjSize * global_num_q_heads * KV_WEIGHT_NUM);

    size_t bias_idx = pre_length + global_head_idx * proj_size + i % proj_size;

    input_ptr[i] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[i] *= scaling_factor;
    }
  }
}

template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qProjSize,
                                     int num_tokens,
                                     int num_q_heads,
                                     float scaling_factor,
                                     int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    input_ptr[i % hidden_size + token_idx * hidden_size * QKV_WEIGHT_NUM] *=
        scaling_factor;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_native(DT *input_ptr,
                                  cuFloatComplex *complex_input,
                                  BatchConfig::PerTokenInfo const *tokenInfos,
                                  int qProjSize,
                                  int kProjSize,
                                  int num_q_heads,
                                  int num_tokens,
                                  int num_kv_heads,
                                  int q_block_size,
                                  int k_block_size,
                                  int q_array_size) {
  CUDA_KERNEL_LOOP(
      i,
      num_tokens * (qProjSize * num_q_heads + kProjSize * num_kv_heads) / 2) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qProjSize : kProjSize;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int head_idx = real_i / (num_tokens * proj_size / 2);
    int idx = real_i % (num_tokens * proj_size / 2);
    int real_part_index = idx * 2 +
                          head_idx * (q_tensor ? q_block_size : k_block_size) +
                          (q_tensor ? 0 : q_array_size);

    int complex_part_index = real_part_index + 1;

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    int token_idx =
        (real_i - head_idx * (num_tokens * proj_size / 2)) / (proj_size / 2);
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    // float before_real = complex_input[i].x, before_complex =
    // complex_input[i].y;

    int pos_i = real_i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_hf(DT *input_ptr,
                              cuFloatComplex *complex_input,
                              BatchConfig::PerTokenInfo const *tokenInfos,
                              int qProjSize,
                              int kProjSize,
                              int num_tokens,
                              size_t q_array_size,
                              int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qProjSize : kProjSize;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int token_idx = real_i / (hidden_size / 2);
    int idx = real_i % (proj_size / 2);
    int head_idx = (real_i - (token_idx * (hidden_size / 2))) / (proj_size / 2);

    int real_part_index = idx + head_idx * proj_size +
                          token_idx * hidden_size * QKV_WEIGHT_NUM +
                          hidden_size * (q_tensor ? 0 : 1);
    int complex_part_index = real_part_index + (proj_size / 2);

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    // get the freq_cis: shape 1 * (qProjSize/2) = 1 * 64
    // apply a Cartesian coordinate transformation
    // multiple with input & /copy back to q/k

    // get position of token

    // size_t pos = id_map[token_idx].token_position;
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    // float before_real = complex_input[i].x, before_complex =
    int pos_i = real_i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
__global__ void fill_entries_above_diagonal(DT *matrix,
                                            size_t num_rows,
                                            size_t num_cols,
                                            size_t num_q_heads,
                                            size_t entries_above_diagonal,
                                            DT value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_q_heads) {
    size_t head_idx = i / entries_above_diagonal;
    size_t entry_idx = i % entries_above_diagonal;
    size_t y = (-1 + sqrt(8 * (float)entry_idx + 1)) / 2;
    size_t x = entry_idx - y * (y + 1) / 2;
    y += (num_cols - num_rows) + 1;
    matrix[head_idx * num_rows * num_cols + num_cols * y + x] = value;
  }
}

template <typename DT>
__global__ void update_key_value_gradient(DT *devQKVProjArray,
                                          DT *kGradCache,
                                          DT *vGradCache,
                                          int update_tokens,
                                          int hidden_size) {
  CUDA_KERNEL_LOOP(i, update_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    int offset = i % hidden_size;

    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;
    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    // key cache
    kGradCache[i] += kVal;
    vGradCache[i] += vVal;

    // for computation
    devQKVProjArray[val_idx] = kGradCache[i];
    devQKVProjArray[val_idx + hidden_size] = vGradCache[i];
  }
}

template <typename DT>
void compute_qkv_kernel(IncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        int shard_id,
                        DT const *input_ptr,
                        DT const *weight_ptr,
                        DT *output_ptr,
                        DT const *bias_ptr,
                        cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  assert(m->qSize == m->vSize && m->qSize == m->kSize);
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudaDataType_t compute_type = cublas_data_type;
  // #if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  //   cudaDataType_t compute_type = cublas_data_type;
  // #else
  //   // For best performance, set the default cublas compute type to
  //   // CUBLAS_COMPUTE_16F for half precision and to
  //   // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  //   cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  //   if (m->output_type[0] == DT_FLOAT) {
  //     compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  //   }
  // #endif
  // Compute (W^T)x matmul: einsum(ijkl,im->jmkl)
  // Weights: qSize x qProjSize x 3 x num_q_heads
  // Input: qSize x num_tokens
  // Output >>> qProjSize x num_tokens x 3 x num_q_heads
  int m_q = m->qProjSize * m->num_q_heads;
  int m_k = m->kProjSize * m->num_q_heads;
  int m_v = m->vProjSize * m->num_q_heads;
  assert(m_q == m_k && m_k == m_v); // keep things simple for now
  int n = bc->num_active_infr_tokens();
  int k = m->qSize;
  int m_ = m_q * QKV_WEIGHT_NUM;
  int lda = k, ldb = k, ldc = m_;
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         m_,
                         n,
                         k,
                         &alpha,
                         weight_ptr,
                         cublas_data_type,
                         lda,
                         input_ptr,
                         cublas_data_type,
                         ldb,
                         &beta,
                         output_ptr,
                         cublas_data_type,
                         ldc,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // apply rotary emmmbedding for q
  // and k step1 change the k, v to complex tensor
  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;
  size_t q_array_size = m->qProjSize * num_tokens * m->num_q_heads;
  // apply bias for q, k, v
  if (*m->qkv_bias) {
    apply_proj_bias_qkv<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(output_ptr,
                                    bias_ptr,
                                    shard_id,
                                    num_tokens,
                                    m->qProjSize,
                                    m->kProjSize,
                                    m->vProjSize,
                                    m->global_num_q_heads,
                                    m->num_q_heads,
                                    *m->scaling_query,
                                    m->scaling_factor,
                                    m->hidden_size);
  } else if (m->scaling_query) {
    scaling_query_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(output_ptr,
                                     num_tokens,
                                     m->num_q_heads,
                                     m->qProjSize,
                                     m->scaling_factor,
                                     m->hidden_size);
  }
  if (*m->apply_rotary_embedding) {
    /*q&k*/
    parallelism = num_tokens * m->hidden_size;
    apply_rotary_embedding_hf<<<GET_BLOCKS(parallelism),
                                min(CUDA_NUM_THREADS, parallelism),
                                0,
                                stream>>>(output_ptr,
                                          m->complex_input,
                                          m->token_infos,
                                          m->qProjSize,
                                          m->kProjSize,
                                          num_tokens,
                                          q_array_size,
                                          m->hidden_size);
  }
}

template <typename DT>
void update_kv_cache_kernel(IncMultiHeadSelfAttentionMeta const *m,
                            BatchConfig const *bc,
                            cudaStream_t stream) {
  int num_tokens = bc->num_active_infr_tokens();
  if (num_tokens > 0) {
    int parallelism = m->hidden_size * num_tokens;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(static_cast<DT *>(m->devQKVProjArray),
                               static_cast<DT *>(m->keyCache),
                               static_cast<DT *>(m->valueCache),
                               m->token_infos,
                               num_tokens,
                               BatchConfig::max_sequence_length(),
                               m->hidden_size);
  }
}

template <typename DT>
void pre_build_weight_kernel(IncMultiHeadSelfAttentionMeta const *m,
                             GenericTensorAccessorR const weight,
                             DataType data_type,
                             cudaStream_t stream) {
  // additional processing for weight uploading
  // Note that we update weight_ptr and bias_ptr when uploading weight and
  // bias
  if (m->quantization_type != DT_NONE) {
    // copy weight_ptr to quantized_weight_ptr, do compression and store in
    // m->weight_ptr
    cudaMemcpyAsync(m->quantized_weight_ptr,
                    weight.get_byte_ptr(),
                    m->quantized_weightSize,
                    cudaMemcpyHostToDevice,
                    stream);

    if (m->quantization_type == DT_INT4) {
      int parallelism = m->qProjSize * m->qSize * m->num_q_heads / 2;
      decompress_int4_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_q_heads);
    } else {
      assert(m->quantization_type == DT_INT8);
      int parallelism = m->qProjSize * m->qSize * m->num_q_heads;
      decompress_int8_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_q_heads);
    }
  } else {
    if (data_type == DT_FLOAT) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_float_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else if (data_type == DT_HALF) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_half_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else {
      assert(false);
    }
  }
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
  // here because we need position info in inference 1

  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  cudaMemcpyAsync(m->token_infos,
                  &(bc->tokensInfo),
                  bc->num_active_infr_tokens() *
                      sizeof(BatchConfig::PerTokenInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  // phase 1: Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);

  // phase 2: Update key/val cache
  update_kv_cache_kernel<DT>(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  compute_attention_kernel(
      m, bc, shard_id, output_ptr, bias_ptr, weight_ptr, stream);
}

template <typename DT>
void peft_bwd_kernel(IncMultiHeadSelfAttentionMeta const *m,
                     BatchConfig const *bc,
                     int shard_id,
                     DT *input_grad_ptr,
                     DT const *weight_ptr,
                     DT const *output_grad_ptr,
                     DT const *bias_ptr,
                     cudaStream_t stream) {
  assert(!m->offload);
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;
  // #if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  //   cudaDataType_t compute_type = cublas_data_type;
  // #else
  //   // For best performance, set the default cublas compute type to
  //   // CUBLAS_COMPUTE_16F for half precision and to
  //   // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  //   cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  //   if (m->output_type[0] == DT_FLOAT) {
  //     compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  //   }
  // #endif
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    if (!bc->requestsInfo[i].peft_bwd) {
      continue;
    }
    int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int num_total_tokens = bc->requestsInfo[i].peft_fwd_tokens;
    int num_processed_tokens = bc->requestsInfo[i].peft_bwd_tokens;
    // int num_total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
    //                        bc->requestsInfo[i].num_tokens_in_batch;
    // Currently assume we are calculating gradients for all tokens
    // of a request
    // assert(num_tokens == num_total_tokens);
    int kt_block_size = m->kProjSize;
    int kt_req_block_size =
        kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
    int vt_block_size = m->vProjSize;
    int vt_req_block_size =
        vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
    // Step 1: compute gradients before final projection
    {
      int m_ = m->vProjSize * m->num_q_heads;
      int n_ = num_tokens;
      int k_ = m->oProjSize;
      int lda = k_;
      int ldb = k_;
      int ldc = m_;
      float alpha = 1.0f, beta = 0.0f;
      // matrix A: output projection weight
      // matrix A's layout: [num_heads, vProjSize, oProjSize]
      DT const *A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                             m->kProjSize * m->num_q_heads +
                                             m->vProjSize * m->num_q_heads);
      // matrix B: output gradients
      // matrix B's layout: [num_new_tokens, oProjSize]
      // DT const *B =
      //     output_grad_ptr +
      //     bc->requestsInfo[i].first_token_offset_in_batch * m->oProjSize;
      DT const *B =
          output_grad_ptr + bc->requestsInfo[i].peft_bwd_tokens * m->oProjSize;
      // matrix C: attn_heads gradients
      // matrix C's layout: [num_new_tokens, num_heads, vProjSize]
      DT *C = static_cast<DT *>(m->handle.workSpace);
      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             m_,
                             n_,
                             k_,
                             &alpha,
                             A,
                             cublas_data_type,
                             lda,
                             B,
                             cublas_data_type,
                             ldb,
                             &beta,
                             C,
                             cublas_data_type,
                             ldc,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Step 2: compute gradients w.r.t. value
    {
      float alpha = 1.0f, beta = 0.0f;
      // matrix A: attn_heads gradients
      // matrix A's layout: [num_tokens, num_heads, vProjSize]
      DT const *A = static_cast<DT *>(m->handle.workSpace);
      // matrix B: qk_prods_softmax
      // matrix B's layout: [num_heads, num_tokens, num_tokens]
      // DT const *B = static_cast<DT *>(m->qk_prods_softmax);
      DT const *B = static_cast<DT *>(m->softmax_activation_buffer) +
                    (num_total_tokens - num_processed_tokens - num_tokens) *
                        num_total_tokens;
      // matrix C: gradients for value (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, num_heads, qProjsize + kProjSize +
      // vProjSize]
      DT *C =
          static_cast<DT *>(m->devQKVProjArray) + m->qProjSize + m->kProjSize;
      int m_ = m->vProjSize;
      int n_ = num_total_tokens;
      int k_ = num_tokens;
      int lda = m->vProjSize * m->num_q_heads;
      int ldb = num_tokens;
      int ldc = m->num_q_heads * (m->qProjSize + m->kProjSize + m->vProjSize);
      int strideA = m->vProjSize;
      int strideB = num_tokens * num_total_tokens;
      int strideC = m->qProjSize + m->kProjSize + m->vProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           m_,
                                           n_,
                                           k_,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Step 3: compute gradients w.r.t. the qk_prods_softmax tensor
    {
      float alpha = 1.0f, beta = 0.0f;
      int m_ = num_total_tokens;
      int n_ = num_tokens;
      int k_ = m->vProjSize;
      int lda = m->vProjSize * m->num_q_heads;
      int ldb = m->vProjSize * m->num_q_heads;
      int ldc = num_tokens;
      int strideA = m->vProjSize;
      int strideB = m->vProjSize;
      int strideC = num_tokens * num_total_tokens;
      // matrix A: value cache
      // matrix A's layout: [num_req, max_num_tokens, num_heads, vProjSize]
      DT const *A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
      // matrix B: attn_heads gradients
      // matrix B's layout: [num_new_tokens, num_heads, vProjSize]
      DT const *B = static_cast<DT *>(m->handle.workSpace);
      // matrix C: qk_prods_softmax gradients
      // matrix C's layout: [num_heads, num_total_tokens, num_new_tokens]
      DT *C = static_cast<DT *>(m->qk_prods_softmax);
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           m_,
                                           n_,
                                           k_,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Step 4: softmax backpropagation
    {
      float alpha = 1.0f, beta = 0.0f;
      int n_param = m->num_q_heads;
      int c_param = num_total_tokens;
      int h_param = 1;
      int w_param = num_tokens;
      checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                            CUDNN_TENSOR_NCHW,
                                            cudnn_data_type,
                                            n_param,
                                            c_param,
                                            h_param,
                                            w_param));
      checkCUDNN(cudnnSoftmaxBackward(m->handle.dnn,
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      m->qk_tensor,
                                      m->softmax_activation_buffer,
                                      m->qk_tensor,
                                      m->qk_prods_softmax,
                                      &beta,
                                      m->qk_tensor,
                                      m->qk_prods));
      // fill all elements above diagonal to force causal attention
      size_t entries_above_diagonal = num_tokens * (num_total_tokens - 1) / 2;
      if (entries_above_diagonal > 0) {
        size_t parallelism = m->num_q_heads * entries_above_diagonal;
        fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                      min((size_t)CUDA_NUM_THREADS,
                                          parallelism),
                                      0,
                                      stream>>>(static_cast<DT *>(m->qk_prods),
                                                num_tokens,
                                                num_total_tokens,
                                                m->num_q_heads,
                                                entries_above_diagonal,
                                                DT(0.0f));
      }
    }
    // Step 5: compute gradients w.r.t. key
    {
      float alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = 1.0f / sqrt(m->kProjSize);
      }
      // matrix A: query activation (in query_activation_buffer)
      // matrix A's layout: [num_tokens, num_heads, m->qProjSize]
      DT const *A = static_cast<DT *>(m->query_activation_buffer) +
                    (num_total_tokens - num_processed_tokens + num_tokens) *
                        m->hidden_size;
      // matrix B: gradients w.r.t. qk_prods
      // matrix B's layout: [num_heads, num_tokens, num_tokens]
      DT const *B = static_cast<DT *>(m->qk_prods);
      // matrix C: gradients w.r.t. key (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, num_heads, qProjsize + kProjSize +
      // vProjSize]
      DT *C = static_cast<DT *>(m->devQKVProjArray) + m->qProjSize;
      int m_ = m->kProjSize;
      int n_ = num_total_tokens;
      int k_ = num_tokens;
      int lda = m->num_q_heads * m->qProjSize;
      int ldb = num_tokens;
      int ldc = m->num_q_heads * (m->qProjSize + m->kProjSize + m->vProjSize);
      int strideA = m->qProjSize;
      int strideB = num_tokens * num_total_tokens;
      int strideC = m->qProjSize + m->kProjSize + m->vProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_T,
                                           m_,
                                           n_,
                                           k_,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // update the key value gradient cache;
    int update_tokens = num_total_tokens - num_processed_tokens;
    int parallelism = m->hidden_size * num_tokens;
    update_key_value_gradient<<<GET_BLOCKS(parallelism),
                                min(CUDA_NUM_THREADS, parallelism),
                                0,
                                stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                          static_cast<DT *>(m->keyGradCache),
                                          static_cast<DT *>(m->valueGradCache),
                                          update_tokens,
                                          m->hidden_size);

    // Step 6: compute gradients w.r.t query
    {
      float alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = 1.0f / sqrt(m->kProjSize);
      }
      // matrix A: key cache
      // matrix A's layout: [num_tokens, num_heads, m->kProjSize]
      DT const *A = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // matrix B: gradients w.r.t. qk_prods
      // matrix B's layout: [num_heads, num_tokens, num_tokens]
      DT const *B = static_cast<DT *>(m->qk_prods);
      // matrix C: gradients w.r.t. query (saved as part of m->devQKVProjArray)
      // matrix C's layout:
      // [num_tokens, num_heads, qProjsize + kProjSize + vProjSize]
      DT *C = static_cast<DT *>(m->devQKVProjArray);
      int m_ = m->qProjSize;
      int n_ = num_tokens;
      int k_ = num_total_tokens;
      int lda = m->kProjSize * m->num_q_heads;
      int ldb = num_tokens;
      int ldc = m->num_q_heads * (m->qProjSize + m->kProjSize + m->vProjSize);
      int strideA = m->kProjSize;
      int strideB = num_tokens * num_total_tokens;
      int strideC = m->qProjSize + m->kProjSize + m->vProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_T,
                                           m_,
                                           n_,
                                           k_,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Step 7: compute gradients w.r.t. input
    {
      float alpha = 1.0f, beta = 0.0f;
      if (!m->reset_input_grads[0]) {
        beta = 1.0f;
      }
      // matrix A: QKV projection weights
      // matrix A's layout:
      // [(qProjSize + kProjSize + vProjSize) * num_q_heads, qSize]
      DT const *A = weight_ptr;
      // matrix B: gradients w.r.t. QKV (concatenated in devQKVArray)
      // matrix B's layout:
      // [num_tokens, num_heads, qProjsize + kProjSize + vProjSize]
      DT const *B = static_cast<DT *>(m->devQKVProjArray);
      // matrix C: gradients w.r.t. input
      // matrix C's layout: [num_tokens, m->qSize]
      DT *C = input_grad_ptr +
              bc->requestsInfo[i].first_token_offset_in_batch * m->qSize;
      int m_ = m->qSize;
      int n_ = num_tokens;
      int k_ = m->num_q_heads * (m->qProjSize + m->kProjSize + m->vProjSize);
      int lda = m_;
      int ldb = k_;
      int ldc = m_;
      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m_,
                             n_,
                             k_,
                             &alpha,
                             A,
                             cublas_data_type,
                             lda,
                             B,
                             cublas_data_type,
                             ldb,
                             &beta,
                             C,
                             cublas_data_type,
                             ldc,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  }
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

template <typename DT>
__global__ void store_kv_cache(DT const *devQKVProjArray,
                               DT *kCache_ptr,
                               DT *vCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int num_tokens,
                               int max_seq_len,
                               int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    int offset = i % hidden_size;

    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];
    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    // key cache
    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
__global__ void store_query_cache(DT const *devQKVProjArray,
                                  DT *qCache_ptr,
                                  int num_tokens,
                                  int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    int offset = i % hidden_size;

    size_t val_idx = token_idx * QKV_WEIGHT_NUM * hidden_size + offset;

    DT qVal = devQKVProjArray[val_idx];

    // query cache
    qCache_ptr[i] = qVal;
  }
}

template <typename DT>
void compute_attention_kernel(IncMultiHeadSelfAttentionMeta *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              DT const *bias_ptr,
                              DT const *weight_ptr,
                              cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;
  // #if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  //   cudaDataType_t compute_type = cublas_data_type;
  // #else
  //   // For best performance, set the default cublas compute type to
  //   // CUBLAS_COMPUTE_16F for half precision and to
  //   // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  //   cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  //   if (m->output_type[0] == DT_FLOAT) {
  //     compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  //   }
  // #endif
  // int num_requests = bc->num_active_requests();
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int q_block_size = m->qProjSize;
  int kt_block_size = m->kProjSize;
  int kt_req_block_size =
      kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  int vt_block_size = m->vProjSize;
  int vt_req_block_size =
      vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    int start_offset = bc->requestsInfo[i].first_token_depth_in_request;
    assert(tokens_previous_requests ==
           bc->requestsInfo[i].first_token_offset_in_batch);
    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                       bc->requestsInfo[i].num_tokens_in_batch;
    // Copy query to m->query_activation_buffer if we need to compute
    // PEFT backward
    if (bc->requestsInfo[i].peft_bwd) {
      // MemoryAllocator *allocator = m->handle.peft_activation_allocator;
      // m->query_activation_buffer = allocator->allocate_instance_untyped(
      //     sizeof(DT) * total_tokens * m->num_q_heads * m->qProjSize);
      int parallelism = m->hidden_size * num_tokens;
      store_query_cache<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(
          static_cast<DT *>(m->devQKVProjArray),
          static_cast<DT *>(m->query_activation_buffer) +
              start_offset * m->hidden_size,
          num_tokens,
          m->hidden_size);
    }

    // bc->token_last_available_idx[i] + 1;
    // Compute (QK^T/sqrt(d_k))
    // a flag of using this scaling alpha
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
        ldc = m_;
    int strideA = q_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // To get A, skip over Q entries from previous requests (same head)
    DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                  tokens_previous_requests * m->qProjSize * m->num_q_heads *
                      QKV_WEIGHT_NUM;
    // To get B, skip over K entries from previous requests (all heads +
    // padding)
    DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
    // To get C, skip over QK^T products from previous requests
    DT *C = static_cast<DT *>(m->qk_prods);
    checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_q_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // add alibi position bias to qk production
    if (*m->position_bias) {
      size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
      apply_position_bias_qkprd<<<GET_BLOCKS(parallelism),
                                  min((size_t)CUDA_NUM_THREADS, parallelism),
                                  0,
                                  stream>>>(C,
                                            num_new_tokens,
                                            total_tokens,
                                            m->num_q_heads,
                                            m->global_num_q_heads,
                                            shard_id);
    }

    // Fill all elements above diagonal in qk prods with -inf to force
    // causal attention.
    assert(num_new_tokens <= total_tokens);
    size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    stream>>>(C,
                                              num_new_tokens,
                                              total_tokens,
                                              m->num_q_heads,
                                              entries_above_diagonal,
                                              static_cast<DT>(-INFINITY));
    }
    // Compute Softmax(QK^T/sqrt(d_k))
    // Before modifying the parameters below, make sure to read the following
    // description of the CUDNN_TENSOR_NCHW tensor layout, from
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t:
    // This tensor format specifies that the data is laid out in the following
    // order: batch size, feature maps, rows, columns. The strides are
    // implicitly defined in such a way that the data are contiguous in memory
    // with no padding between images, feature maps, rows, and columns; the
    // columns are the inner dimension and the images are the outermost
    // dimension.
    int n_param = m->num_q_heads;
    int c_param = total_tokens;
    int h_param = 1;
    int w_param = num_new_tokens;
    checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          n_param,
                                          c_param,
                                          h_param,
                                          w_param));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
    // The softmax operation below is executed according to the
    // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
    // softmax operation is computed per spatial location (H,W) per image (N)
    // across dimension C.
    checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &softmax_alpha,
                                   m->qk_tensor,
                                   C,
                                   &softmax_beta,
                                   m->qk_tensor,
                                   C_softmax));
    // Copy C_softmax to m->softmax_activation_buffer if we need to compute
    // PEFT backward
    if (bc->requestsInfo[i].peft_bwd) {
      // MemoryAllocator *allocator = m->handle.peft_activation_allocator;
      // m->softmax_activation_buffer = allocator->allocate_instance_untyped(
      //     sizeof(DT) * total_tokens * num_new_tokens * m->num_q_heads);
      DT *softmax_cache = static_cast<DT *>(m->softmax_activation_buffer) +
                          start_offset * bc->requestsInfo[i].peft_bwd_tokens;
      checkCUDA(cudaMemcpyAsync(softmax_cache,
                                C_softmax,
                                sizeof(DT) * total_tokens * num_new_tokens *
                                    m->num_q_heads,
                                cudaMemcpyDeviceToDevice,
                                stream));
    }

    // Matmul softmax(QK^T/sqrt(d_k)) by V
    alpha = 1.0f, beta = 0.0f;
    m_ = m->vProjSize;
    n = num_new_tokens;
    k = total_tokens;
    lda = m_ * m->num_q_heads, ldb = n, ldc = m_ * m->num_q_heads;
    strideA = vt_block_size;
    strideB = num_new_tokens * total_tokens;
    strideC = m->vProjSize;
    // To get A, skip over V^T entries from previous requests (all heads +
    // padding)
    A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
    // To get B, skip over softmax(QK^T/sqrt(d_k)) entries from previous
    // requests (all heads)
    B = C_softmax;
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests
    C = static_cast<DT *>(m->attn_heads) +
        tokens_previous_requests * m->num_q_heads * m->vProjSize;
    checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_T,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_q_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    tokens_previous_requests += num_new_tokens;
  }

  // Project to output, save result directly on output tensor
  DT alpha = 1.0f, beta = 0.0f;
  int m_ = m->oProjSize;
  int k = m->vProjSize * m->num_q_heads;
  int n = bc->num_active_tokens();
  int lda = k, ldb = k, ldc = m_;
  DT const *A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                         m->kProjSize * m->num_q_heads +
                                         m->vProjSize * m->num_q_heads);
  DT const *B = static_cast<DT *>(m->attn_heads);
  DT *C = static_cast<DT *>(output_ptr);

  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         m_,
                         n,
                         k,
                         &alpha,
                         A,
                         cublas_data_type,
                         lda,
                         B,
                         cublas_data_type,
                         ldb,
                         &beta,
                         C,
                         cublas_data_type,
                         ldc,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->oProjSize * num_tokens;
    int qkv_weight_size = m->qProjSize * m->global_num_q_heads +
                          m->kProjSize * m->global_num_q_heads +
                          m->vProjSize * m->global_num_q_heads;

    apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(
        output_ptr, bias_ptr, num_tokens, qkv_weight_size, m->oProjSize);
  }

  assert(tokens_previous_requests == num_tokens);
}

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    if (m->offload) {
      pre_build_weight_kernel<half>(m, weight, input.data_type, stream);
    }
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_half_ptr(),
        m->offload ? static_cast<half *>(m->weight_ptr) : weight.get_half_ptr(),
        output.get_half_ptr(),
        bias_ptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    if (m->offload) {
      pre_build_weight_kernel<float>(m, weight, input.data_type, stream);
    }
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_float_ptr(),
        m->offload ? static_cast<float *>(m->weight_ptr)
                   : weight.get_float_ptr(),
        output.get_float_ptr(),
        bias_ptr,
        stream);
  } else {
    assert(false && "Unspported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);

    // if (input.data_type == DT_HALF) {
    //   print_tensor<half>(input.get_half_ptr(),
    //                      32,
    //                      "[IncMultiHeadSelfAttention:forward:input]");
    //   print_tensor<half>(weight.get_half_ptr(),
    //                      32,
    //                      "[IncMultiHeadSelfAttention:forward:weight]");
    //   print_tensor<half>(output.get_half_ptr(),
    //                      32,
    //                      "[IncMultiHeadSelfAttention:forward:output]");
    //   print_tensor<half>(
    //       bias.get_half_ptr(), 32,
    //       "[IncMultiHeadSelfAttention:forward:bias]");
    // } else {
    //   print_tensor<float>(input.get_float_ptr(),
    //                       32,
    //                       "[IncMultiHeadSelfAttention:forward:input]");
    //   print_tensor<float>(weight.get_float_ptr(),
    //                       32,
    //                       "[IncMultiHeadSelfAttention:forward:weight]");
    //   print_tensor<float>(output.get_float_ptr(),
    //                       32,
    //                       "[IncMultiHeadSelfAttention:forward:output]");
    //   print_tensor<float>(
    //       bias.get_float_ptr(), 32,
    //       "[IncMultiHeadSelfAttention:forward:bias]");
    // }

    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

/*static*/
void IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // assert(input.data_type == weight.data_type);
  assert(input_grad.data_type == output_grad.data_type);
  if (use_bias) {
    assert(input_grad.data_type == bias.data_type);
  }

  if (input_grad.data_type == DT_HALF) {
    assert(!m->offload);
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_half_ptr(),
                                                    weight.get_half_ptr(),
                                                    output_grad.get_half_ptr(),
                                                    bias_ptr,
                                                    stream);
  } else if (input_grad.data_type == DT_FLOAT) {
    assert(!m->offload);
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_float_ptr(),
                                                    weight.get_float_ptr(),
                                                    output_grad.get_float_ptr(),
                                                    bias_ptr,
                                                    stream);
  } else {
    assert(false && "Unspported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention PEFT backward time = %.9fms\n", elapsed);
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    INC_DECODING_MODE,
                                    attn,
                                    attn->qSize,
                                    attn->kSize,
                                    attn->vSize,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->apply_rotary_embedding,
                                    attn->qkv_bias,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->final_bias,
                                    attn->scaling_factor,
                                    weight,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    attn->quantization_type,
                                    attn->offload) {}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    InferenceMode infer_mode,
    Op const *attn,
    int _qSize,
    int _kSize,
    int _vSize,
    int _qProjSize,
    int _kProjSize,
    int _vProjSize,
    int _oProjSize,
    bool _apply_rotary_embedding,
    bool _qkv_bias,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    bool _final_bias,
    float _scaling_factor,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn), weight_ptr(nullptr), bias_ptr(nullptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
  qSize = _qSize;
  kSize = _kSize;
  vSize = _vSize;
  // assume dimensions match for now
  assert(qSize == kSize);
  assert(kSize == vSize);
  qProjSize = _qProjSize;
  kProjSize = _kProjSize;
  assert(qProjSize == kProjSize); // required for attention QK^T matmul
  vProjSize = _vProjSize;
  oProjSize = _oProjSize;
  size_t size_of_dt = data_type_size(attn->data_type);
  quantization_type = _quantization_type;
  offload = _offload;

  global_num_q_heads = _global_num_q_heads;
  global_num_kv_heads = _global_num_kv_heads;
  num_q_heads = _num_q_heads;
  num_kv_heads = _num_kv_heads;
  hidden_size = num_q_heads * qProjSize;

  weightSize =
      ((qSize * qProjSize + oProjSize * (vProjSize > 0 ? vProjSize : vSize)) *
           num_q_heads +
       (kSize * kProjSize + vSize * vProjSize) * num_q_heads) *
      size_of_dt;
  if (quantization_type != DT_NONE) {
    quantized_weightSize = get_quantization_to_byte_size(
        attn->data_type, quantization_type, weightSize);
  }
  // biasSize = _bias ? oProjSize * size_of_dt * 4 : 0;

  int qkv_bias_size =
      qProjSize * num_q_heads + (kProjSize + vProjSize) * num_q_heads;
  int final_bias_size = oProjSize;
  biasSize =
      (_qkv_bias ? qkv_bias_size : 0) + (final_bias ? final_bias_size : 0);

  // has_load_weights = (bool *)calloc(1, sizeof(bool));
  //*has_load_weights = false;
  apply_rotary_embedding = (bool *)calloc(1, sizeof(bool));
  *apply_rotary_embedding = _apply_rotary_embedding;
  qkv_bias = (bool *)calloc(1, sizeof(bool));
  *qkv_bias = _qkv_bias;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;
  final_bias = (bool *)calloc(1, sizeof(bool));
  *final_bias = _final_bias;

  // allocate weight and bias in the reserve space for cpu offloading
  if (offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(weightSize);
    bias_ptr = gpu_mem_allocator.allocate_reserved_untyped(biasSize);
  }

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = BatchConfig::max_tokens_per_batch();
    size_t qkv_max_proj_size = max_tokens_per_batch * (qProjSize * num_q_heads +
                                                       kProjSize * num_q_heads +
                                                       vProjSize * num_q_heads);
    size_t key_cache_size = 0, value_cache_size = 0;
    switch (infer_mode) {
      case INC_DECODING_MODE:
      case TREE_VERIFY_MODE: {
        key_cache_size = num_q_heads * kProjSize *
                         BatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length();
        value_cache_size = num_q_heads * vProjSize *
                           BatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length();
        break;
      }
      case BEAM_SEARCH_MODE: {
        key_cache_size = num_q_heads * kProjSize *
                         BeamSearchBatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length() *
                         BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        value_cache_size = num_q_heads * vProjSize *
                           BeamSearchBatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length() *
                           BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    size_t tokeninfo_size = max_tokens_per_batch;
    size_t qk_prod_size =
        max_tokens_per_batch * BatchConfig::max_sequence_length() * num_q_heads;
    size_t attn_heads_size = max_tokens_per_batch * num_q_heads * vProjSize;
    size_t complex_size = (max_tokens_per_batch * (qProjSize * num_q_heads +
                                                   kProjSize * num_q_heads)) /
                          2;
    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size +
         2 * qk_prod_size + attn_heads_size) *
            size_of_dt +
        tokeninfo_size * sizeof(BatchConfig::PerTokenInfo) +
        complex_size * sizeof(cuFloatComplex); // more components will
    // assume we have only one peft requests.
    size_t key_grad_cache_size =
        num_q_heads * kProjSize * BatchConfig::max_sequence_length();
    size_t value_grad_cache_size =
        num_q_heads * vProjSize * BatchConfig::max_sequence_length();
    size_t query_activation_size =
        num_q_heads * qProjSize * BatchConfig::max_sequence_length();
    size_t softmax_activation_size = num_q_heads *
                                     BatchConfig::max_sequence_length() *
                                     BatchConfig::max_sequence_length();

    totalSize += (key_grad_cache_size + value_grad_cache_size +
                  query_activation_size + softmax_activation_size);

    if (offload) {
      // assert that we have enough reserved work space left
      size_t totalSharedSize =
          infer_mode == TREE_VERIFY_MODE
              ? totalSize -
                    (key_cache_size + value_cache_size + qkv_max_proj_size) *
                        size_of_dt
              : totalSize - (key_cache_size + value_cache_size) * size_of_dt;

      size_t instance_size =
          size_of_dt *
          (infer_mode == TREE_VERIFY_MODE
               ? key_cache_size + value_cache_size + qkv_max_proj_size
               : key_cache_size + value_cache_size);

      if (quantization_type != DT_NONE) {
        totalSharedSize += quantized_weightSize;
      }
      assert(gpu_mem_allocator.reserved_total_size -
                 gpu_mem_allocator.reserved_allocated_size >=
             totalSharedSize);
      gpu_mem_allocator.create_legion_instance(reserveInst, instance_size);
    } else {
      gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
    }

    // in tree_verify, enable devQKVProjArray;
    if (!offload || infer_mode == TREE_VERIFY_MODE) {
      devQKVProjArray = gpu_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size * size_of_dt);
    } else {
      devQKVProjArray = gpu_mem_allocator.allocate_reserved_untyped(
          qkv_max_proj_size * size_of_dt);
      // offset += qkv_max_proj_size * size_of_dt;
    }

    // use key value cache in all mode.
    keyCache = gpu_mem_allocator.allocate_instance_untyped(key_cache_size *
                                                           size_of_dt);
    valueCache = gpu_mem_allocator.allocate_instance_untyped(value_cache_size *
                                                             size_of_dt);

    query_activation_buffer = gpu_mem_allocator.allocate_instance_untyped(
        query_activation_size * size_of_dt);
    softmax_activation_buffer = gpu_mem_allocator.allocate_instance_untyped(
        softmax_activation_size * size_of_dt);
    keyGradCache = gpu_mem_allocator.allocate_instance_untyped(
        key_grad_cache_size * size_of_dt);
    valueGradCache = gpu_mem_allocator.allocate_instance_untyped(
        value_grad_cache_size * size_of_dt);

    if (offload) {
      token_infos =
          gpu_mem_allocator.allocate_reserved<BatchConfig::PerTokenInfo>(
              tokeninfo_size);
      // offset += sizeof(BatchConfig::PerTokenInfo) * tokeninfo_size;
      qk_prods = gpu_mem_allocator.allocate_reserved_untyped(qk_prod_size *
                                                             size_of_dt);
      // offset += qk_prod_size * size_of_dt;
      qk_prods_softmax = gpu_mem_allocator.allocate_reserved_untyped(
          qk_prod_size * size_of_dt);
      // offset += qk_prod_size * size_of_dt;
      attn_heads = gpu_mem_allocator.allocate_reserved_untyped(attn_heads_size *
                                                               size_of_dt);
      // offset += attn_heads_size * size_of_dt;
      complex_input =
          gpu_mem_allocator.allocate_reserved<cuFloatComplex>(complex_size);
      // offset += complex_size * sizeof(cuFloatComplex);
    } else {
      token_infos =
          gpu_mem_allocator.allocate_instance<BatchConfig::PerTokenInfo>(
              tokeninfo_size);
      qk_prods = gpu_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = gpu_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
      attn_heads = gpu_mem_allocator.allocate_instance_untyped(attn_heads_size *
                                                               size_of_dt);
      complex_input =
          gpu_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }
    if (!offload) {
      assert(gpu_mem_allocator.reserved_total_size ==
             gpu_mem_allocator.reserved_allocated_size);
    }
  }
  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

}; // namespace FlexFlow
