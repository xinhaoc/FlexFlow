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

#include "falcon.h"

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void FALCON::create_falcon_model(FFModel &ff,
                                 std::string const &model_config_file_path,
                                 std::string const &weight_file_path,
                                 InferenceMode mode,
                                 bool use_full_precision) {
  FalconConfig falcon_config(model_config_file_path);
  falcon_config.print();

  if (ff.config.tensor_parallelism_degree > falcon_config.n_head ||
      falcon_config.n_head % ff.config.tensor_parallelism_degree != 0 ||
      ff.config.tensor_parallelism_degree > falcon_config.n_head_kv ||
      falcon_config.n_head_kv % ff.config.tensor_parallelism_degree != 0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

  Tensor input;
  {
    assert(falcon_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token;
  std::vector<int> axes = {0};

  if (use_full_precision) {
    token = ff.embedding(input,
                         falcon_config.vocab_size,
                         falcon_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init);
  } else {
    token = ff.embedding(input,
                         falcon_config.vocab_size,
                         falcon_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init);
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("word_embeddings_weight", embedding);

  for (int i = 0; i < falcon_config.n_layer; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);
    // step 1: attention
    Tensor att_norm =
        ff.layer_norm(token, axes, true, falcon_config.layer_norm_epsilon);
    Layer *attention_norm = ff.layers.back();

    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_input_layernorm_weight",
                           attention_norm);
    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multiquery_self_attention(
            att_norm,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,
            false,
            false,
            false,
            DT_NONE,
            NULL,
            true);
        break;
      }

      case TREE_VERIFY_MODE: {
        mha = ff.inc_multiquery_self_attention_verify(
            att_norm,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,    /*dropout*/
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
        );
        break;
      }

      case INC_DECODING_MODE: {
        mha = ff.inc_multiquery_self_attention(
            att_norm,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,    /*dropout*/
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }
    Layer *attention_layer = ff.layers.back();

    // multi query
    //  weights_layers.emplace("layers_" + std::to_string(i) +
    //                             "_self_attention_dense_weight",
    //                         attention_layer);

    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);
    Tensor dense_h_to_4h =
        ff.dense(att_norm, falcon_config.hidden_size * 4, AC_MODE_NONE, false);
    Layer *dense_h_to_4h_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_mlp_dense_h_to_4h_weight",
                           dense_h_to_4h_layer);
    dense_h_to_4h = ff.gelu(dense_h_to_4h);
    Tensor mlp_output =
        ff.dense(dense_h_to_4h, falcon_config.hidden_size, AC_MODE_NONE, false);
    Layer *dense_4h_to_h_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_mlp_dense_4h_to_h_weight",
                           dense_4h_to_h_layer);

    token = ff.add(token, mha);
    token = ff.add(token, mlp_output);
  }
  // final normalization and linear
  Tensor ln_f =
      ff.layer_norm(token, axes, true, falcon_config.layer_norm_epsilon);
  Layer *ln_f_layer = ff.layers.back();
  weights_layers.emplace("ln_f_weight", ln_f_layer);

  Tensor lm_head =
      ff.dense(ln_f, falcon_config.vocab_size, AC_MODE_NONE, false);
  Layer *lm_head_layer = ff.layers.back();
  weights_layers.emplace("lm_head_weight", lm_head_layer);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    output = ff.beam_top_k(softmax, falcon_config.max_beam_width, false);
  } else {
    output = ff.arg_top_k(lm_head, /*k=*/1, false);
    // output = ff.argmax(lm_head, /*beam_Search*/ false);
  }

  // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  InferenceManager *im = InferenceManager::get_inference_manager();
  im->compile_model_and_allocate_buffer(&ff);
  FileDataLoader fileloader("",
                            weight_file_path,
                            falcon_config.n_head,
                            falcon_config.n_head_kv,
                            falcon_config.hidden_size,
                            falcon_config.hidden_size / falcon_config.n_head,
                            ff.config.tensor_parallelism_degree);
  std::cout << "------laod weights ----------" << std::endl;
  fileloader.load_weights(&ff, weights_layers, use_full_precision);
  std::cout << "------load weight finished----------" << std::endl;

  // init operators
  im->init_operators_inference(&ff);
}

}; // namespace FlexFlow
