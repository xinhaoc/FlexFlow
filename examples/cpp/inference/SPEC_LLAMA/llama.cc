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

#include "models/llama.h"
#include "flexflow/inference.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("llama");

void parse_input_args(char **argv, int argc, LLAMA::Config &config) {
  for (int i = 1; i < argc; i++) {
    // input
    if (!strcmp(argv[i], "--dataset")) {
      config.input_path = std::string(argv[++i]);
      continue;
    }

    // weights
    if (!strcmp(argv[i], "--weights")) {
      config.weight_file_path = std::string(argv[++i]);
      continue;
    }
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  LLAMA::Config llama_config;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv, argc, llama_config);
  InferenceManager im(ffconfig, llama_config.batchSize, 1);
  RequestManager rm;
  // Add a single request
  std::vector<BatchConfig::TokenId> prompt{
      1, 306, 4658, 278, 6593, 310, 2834, 338};
  rm.register_new_request(prompt, llama_config.sentence_len);

  FFModel beam_model(ffconfig), tree_model(ffconfig), inc_model(ffconfig);
  LLAMA::create_llama_model(beam_model, im, llama_config, 1, BEAM_SEARCH_MODE);
  LLAMA::create_llama_model(tree_model, im, llama_config, 1, TREE_VERIFY_MODE);
  LLAMA::create_llama_model(inc_model, im, llama_config, 1, INC_DECODING_MODE);

  // entry---------------------------
  int depth = 0;
  std::map<int, Future> beam_future_handlers, tree_future_handler;
  std::map<int, BeamSearchBatchConfig> beam_batch_configs;
  std::map<int, TreeVerifyBatchConfig> tree_batch_configs;

  bool new_req = true;
  TreeVerifyBatchConfig tree_bc;

  while (depth < llama_config.max_beam_depth) {
    int bid = 0;
    if (beam_future_handlers.find(bid) == beam_future_handlers.end()) {
      BeamSearchBatchConfig bc;
      BeamInferenceResult ir;
      bc = rm.prepare_next_batch_beam(bc, ir);

      std::cout << "sub_requests: " << bc.sub_requests[0] << "\n";
      FutureMap fm = im.inference(&beam_model, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      beam_future_handlers[bid] = fm.get_future(0);
      beam_batch_configs[bid] = bc;
    } else {
      // have luanched this bid
      Future future = beam_future_handlers[bid];
      if (!future.is_ready(true /*subscribe*/)) {
        continue;
      } else {
        std::cout << "future is ready...." << std::endl;
      }
      // process end
      BeamInferenceResult ir = future.get_result<BeamInferenceResult>();
      BeamSearchBatchConfig bc = beam_batch_configs[bid];
      depth = bc.beamRequestsInfo[0].current_depth;
      bc = rm.prepare_next_batch_beam(bc, ir);

      std::cout << "llama current depth: " << depth << std::endl;
      std::cout << "sub_requests: " << bc.sub_requests[0] << "\n";
      FutureMap fm = im.inference(&beam_model, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      beam_future_handlers[bid] = fm.get_future(0);
      beam_batch_configs[bid] = bc;

      // tranverse the tree in dfs order;
      if (depth >= llama_config.max_beam_depth) {
        // std::cout << "tranverse the tree"
        //           << "\n";
        // rm.tranverse_beam_tree(bc);
        tree_bc = rm.convert_beam_to_tree_batch_config(bc);
      }
    }
  }

  // original
  {
    std::vector<BatchConfig::TokenId> tokens{1,
                                             306,
                                             4658,
                                             278,
                                             6593,
                                             310,
                                             2834,
                                             338,
                                             593,
                                             595,
                                             17252,
                                             5031,
                                             993,
                                             616,
                                             368,
                                             2302};
    BatchConfig bc;
    bc.num_tokens = 16;
    bc.requestsInfo[0].num_tokens_in_batch = bc.num_tokens;
    bc.requestsInfo[0].token_start_offset = 0;
    bc.requestsInfo[0].max_sequence_length = 347;
    bc.requestsInfo[0].request_guid = 1000000;
    bc.request_completed[0] = false;
    for (int i = 0; i < bc.num_tokens; i++) {
      bc.tokensInfo[i].token_id = tokens[i];
      bc.tokensInfo[i].abs_depth_in_request = i;
      bc.tokensInfo[i].request_index = 0;
    }
    FutureMap fm = im.inference(&inc_model, 0, bc);
    assert(fm.get_future_map_domain().get_volume() == 1);
    Future future = fm.get_future(0);
    InferenceResult ir = future.get_result<InferenceResult>();
    for (int i = 0; i < bc.num_tokens; i++) {
      printf("decoding_tokens[%d] = %d\n", i, ir.token_ids[i]);
    }
  }

  // verification
  {
    std::vector<BatchConfig::TokenId> tokens{1,
                                             306,
                                             4658,
                                             278,
                                             6593,
                                             310,
                                             2834,
                                             338,
                                             593,
                                             595,
                                             17252,
                                             5031,
                                             993,
                                             616,
                                             368,
                                             2302};
    tree_bc.num_tokens = 16;
    tree_bc.requestsInfo[0].num_tokens_in_batch = tree_bc.num_tokens;
    for (int i = 0; i < tree_bc.num_tokens; i++) {
      tree_bc.tokensInfo[i].token_id = tokens[i];
      tree_bc.tokensInfo[i].abs_depth_in_request = i;
      tree_bc.tokensInfo[i].request_index = 0;
    }
    FutureMap fm = im.inference(&tree_model, 0, tree_bc);
    assert(fm.get_future_map_domain().get_volume() == 1);
    Future future = fm.get_future(0);
    InferenceResult ir = future.get_result<InferenceResult>();
    for (int i = 0; i < tree_bc.num_tokens; i++) {
      printf("verify_tokens[%d] = %d\n", i, ir.token_ids[i]);
    }
  }

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}