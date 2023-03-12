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

#include "minigpt.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("minigpt");

// future read from config file
MiniGPTConfig::MiniGPTConfig(void) {
  //todo read from config/param file
  num_layers = 6;
  embedding_prob_drop = 0.1;
  n_embd = 768;
  resid_pdrop = 0.1;
  vocab_size = 50257;
  block_size = 1024;

}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  MiniGPTConfig minigptconfig;

  FFModel ff(ffConfig);

  //todo init params from pre-trained model
  Tensor input;
  Tensor pos;
  {
    int const token_dims[] = {ffConfig.batchSize, 10, minigptconfig.n_embd};
    int const pos_dims[] = {1 10, minigptconfig.n_embd};
    input = ff.create_tensor<3>(token_dims, DT_FLOAT);
    pos = ff.create_tensor<3>(pos_dims, DT_INT64)
  }
  
  //word&position embedding
  Tensor token_embedding = ff.embedding(input, minigptconfig.vocab_size);
  Tensor position_embedding = ff.embedding(pos, minigptconfig.block_size);
  Tensor x = ff.add(token_embedding, position_embedding)；
  x =ff.dropout(x, minigptconfig.embedding_prob_drop);

  // n-layers transformer block
  for (int i = 0; i < minigptconfig.num_layers; i++) {
    // get q, k, v
    float const *data = NULL;
    x = ff.layer_norm(minigptconfig.n_embd, layer_norm)
    // //get the latest layer
    // Layer *l = ff.layers.back();
    // //get Tensor access
    // assert(len(l->weights) == 2)
    // //copy data to weights tensor
    // Tensor weight = weights[0];
    // Tensor bias = weights[1];
    // weight.set_tensor(ff, 0, data);
    // bias.set_tensor(ff, 0, data);

    x = ff.dense(x, minigptconfig.n_embd * 3)
    q, k, v = ff.split(x, minigptconfig.n_embd, dim=2);
    // multihead attention
    mha = ff.multihead_attention(q, k, v)
    x = ff.add(x, mha);
    //mlp
    c_fc = ff.dense(x, minigptconfig.n_embd * 4);
    act = ff.gelu(c_fc);
    c_proj = ff.dense(act, minigptconfig.n_embd);
    dropout = ff.dropout(c_proj, minigptconfig.resid_pdrop);

    x = ff.add(x, dropout);
  }

  x = ff.layer_norm(x, minigptconfig.n_embd);
  x = ff.dense(x, vocab_size);

  // optimizer
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  ff.compile(optimizer, LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics);
  // Data Loader
  ParallelTensor input_pt, label_pt;
  ff.get_parallel_tensor_from_tensor(input, input_pt);
  ff.get_parallel_tensor_from_tensor(ff.label_tensor, label_pt);
  DataLoader loader(ff, minigptconfig, input, ff.label_tensor);
  loader.next_batch(ff);
  loader.reset();
  ff.init_operators();

  //train
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    loader.reset();
    ff.reset_metrics();
    int iterations = loader.num_samples / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      // Only load data once for random input
      if (iter == 0 && epoch == 0) {
        loader.next_batch(ff);
      }
      runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }

  fprintf(stderr, "----------train end--------------");
}