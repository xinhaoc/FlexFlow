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

#include "flexflow/inference.h"
#include "models/opt.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("opt");

void parse_input_args(char **argv, int argc, OPT::Config &config) {
  for (int i = 1; i < argc; i++) {
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
  OPT::Config opt_config;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv, argc, opt_config);
  InferenceManager im(ffconfig, opt_config.batchSize, 1);
  RequestManager rm;
  // Add a single request
  std::vector<BatchConfig::TokenId> prompt = {
      2, 5625, 16, 10, 2721, 183, 8, 38, 236};
  rm.register_new_request(prompt, opt_config.sentence_len);
  
  std::cout<<"regist"<<std::endl;
  FFModel inc_model(ffconfig);
  OPT::create_opt_model(inc_model, im, opt_config, 1, INC_DECODING_MODE);
  std::cout<<"model create"<<std::endl;
  // entry---------------------------
  int depth = 0;
  std::map<int, Future> future_handlers;
  std::map<int, BatchConfig> batch_configs;
  int sentence_length = 0;
  while (true) {
    int bid = 0;
    if (future_handlers.find(bid) == future_handlers.end()) {
      BatchConfig bc;
      InferenceResult ir;
      
      bc = rm.prepare_next_batch(bc, ir);
      std::cout<<"prepare batch"<<std::endl;
      FutureMap fm = im.inference(&inc_model, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      future_handlers[bid] = fm.get_future(0);
      batch_configs[bid] = bc;
    } else {
      Future future = future_handlers[bid];
      if (!future.is_ready(true /*subscribe*/)) {
        continue;
      } else {
        std::cout << "future is ready...." << std::endl;
      }
      // process end
      InferenceResult ir = future.get_result<InferenceResult>();
      BatchConfig bc = batch_configs[bid];
      bc = rm.prepare_next_batch(bc, ir);
      sentence_length += bc.num_tokens;
      FutureMap fm = im.inference(&inc_model, bid, bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      future_handlers[bid] = fm.get_future(0);
      batch_configs[bid] = bc;
    }
  }
}
void FlexFlow::register_custom_tasks() {}