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

#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"

using namespace std;
using namespace FlexFlow;

class FileDataLoader {
public:
  FileDataLoader(std::string _input_path,
                 std::string _weight_file_path,
                 int _num_heads,
                 int _num_kv_heads,
                 size_t _hidden_dim,
                 size_t _qkv_inner_dim,
                 int _tensor_partition_num);

  BatchConfig::TokenId *generate_requests(int num, int length);

  template <typename DT>
  void load_single_weight_tensor(FFModel *ff,
                                 Tensor weight,
                                 int weight_idx,
                                 std::string const &layername);

  void load_quantization_weight(FFModel *ff,
                                Tensor weight,
                                int weight_idx,
                                std::string const &layername,
                                bool use_full_precision);
  void load_weights(FFModel *ff,
                    std::unordered_map<std::string, Layer *> weights_layers,
                    bool use_full_precision);

  void load_positions(FFModel *ff,
                      Tensor pt,
                      ParallelTensor position_pt,
                      int max_seq_length,
                      int offset);

private:
  int num_heads, num_kv_heads, tensor_partition_num;
  size_t hidden_dim, qkv_inner_dim;
  std::string input_path;
  std::string weight_file_path;
};