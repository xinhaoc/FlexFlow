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
#include "flexflow/model.h"
#include "flexflow/ops/kernels/masked_fill_kernels.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;

using namespace FlexFlow::Kernels::MaskedFill;

bool operator==(MaskedFillParams const &lhs, MaskedFillParams const &rhs) {
  return lhs.filled_value == rhs.filled_value;
}

bool MaskedFillParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  if (!input.first.is_valid()) {
    return false;
  }
  if (!input.second.is_valid()) {
    return false;
  }
  ParallelTensorShape A = input.first;
  ParallelTensorShape B = input.second;
  int numdim = std::min(A.num_dims, B.num_dims);
  for (int i = 0; i < numdim; i++) {
    if (A.dims[i].size > 1 && B.dims[i].size > 1) {
      if (A.dims[i] != B.dims[i]) {
        return false;
      }
    }
  }
  return true;
}

MaskedFillParams MaskedFill::get_params() const {
  MaskedFillParams params;
  params.filled_value = this->filled_value;
  return params;
}

Tensor FFModel::masked_fill(const Tensor input,
                            const Tensor mask,
                            float value,
                            char const *name) {
  Layer *masked_fill = new Layer(this,
                                 OP_MASKED_FILL,
                                 DT_FLOAT,
                                 name,
                                 2 /*inputs*/,
                                 0 /*weights*/,
                                 1 /*output*/,
                                 input,
                                 mask);
  // https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch-tensor-masked-fill
  // assert(broadcastable(input, mask));
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < input->num_dims; i++) {
    dims[i] = input->dims[i];
  }
  masked_fill->outputs[0] = create_tensor_legion_ordering(input->num_dims,
                                                          dims,
                                                          input->data_type,
                                                          masked_fill,
                                                          0,
                                                          true /*create_grad*/);
  masked_fill->add_float_property("filled_value", value);
  layers.push_back(masked_fill);
  return masked_fill->outputs[0];
}

Op *MaskedFill::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float filled_value;
  layer->get_float_property("filled_value", filled_value);
  return new MaskedFill(model, inputs[0], inputs[1], filled_value, layer->name);
}

MaskedFill::MaskedFill(FFModel &model,
                       MaskedFillParams const &params,
                       std::pair<ParallelTensor, ParallelTensor> const &inputs,
                       char const *name)
    : MaskedFill(
          model, inputs.first, inputs.second, params.filled_value, name) {}

MaskedFill::MaskedFill(FFModel &model,
                       const ParallelTensor input,
                       const ParallelTensor mask,
                       float _filled_value,
                       char const *name)
    : Op(model,
         OP_MASKED_FILL,
         input->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input,
         mask),
      filled_value(_filled_value) {
  assert(input->data_type == mask->data_type);
  int numdim = std::max(input->num_dims, mask->num_dims);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    if (i >= input->num_dims) {
      dims[i] = mask->dims[i];
    } else if (i >= mask->num_dims) {
      dims[i] = input->dims[i];
    } else if (input->dims[i].size == mask->dims[i].size) {
      assert(input->dims[i] == mask->dims[i]);
      dims[i] = input->dims[i];
    } else if (input->dims[i].size == 1) {
      dims[i] = mask->dims[i];
    } else if (mask->dims[i].size == 1) {
      dims[i] = input->dims[i];
    } else {
      assert(false && "Operands could not be broadcast together");
      exit(0);
    }
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      mask->num_dims, dims, input->data_type, this);
}

void MaskedFill::serialize(Legion::Serializer &sez) const {
  MaskedFillParams params = get_params();
  sez.serialize(params.filled_value);
}

using PCG::Node;
/*static*/
Node MaskedFill::deserialize(FFModel &ff,
                             Legion::Deserializer &dez,
                             ParallelTensor inputs[],
                             int num_inputs) {
  assert(num_inputs == 2);
  float filled_value;
  dez.deserialize(filled_value);
  MaskedFillParams params;
  params.filled_value = filled_value;
  return ff.get_or_create_node<MaskedFill>({inputs[0], inputs[1]}, params);
}

Op *MaskedFill::materialize(FFModel &ff,
                            ParallelTensor inputs[],
                            int num_inputs) const {
  MaskedFillParams params = get_params();
  return new MaskedFill(ff, params, {inputs[0], inputs[1]}, this->name);
}

void MaskedFill::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(MASKEDFILL_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(MaskedFill)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *MaskedFill::init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  MaskedFill const *maskedfill = (MaskedFill const *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  MaskedFillMeta *m = new MaskedFillMeta(handle, maskedfill);
  return m;
}

void MaskedFill::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(MASKEDFILL_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void MaskedFill::forward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  MaskedFillMeta const *m = *((MaskedFillMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR mask = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, mask, output);
}

void MaskedFill::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(MASKEDFILL_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void MaskedFill::backward_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  MaskedFillMeta const *m = *((MaskedFillMeta **)task->local_args);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR mask = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  backward_kernel_wrapper(m, output_grad, mask, input_grad);
}

bool MaskedFill::measure_operator_cost(Simulator *sim,
                                       MachineView const &mv,
                                       CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_mask, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_mask)) {
    return false;
  }
  MaskedFillMeta *m = new MaskedFillMeta(sim->handler, this);
  sim->free_all();
  bool out_of_memory = false;
  Domain input_domain = sub_input.get_domain();
  void *input_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW input_acc(
      inputs[0]->data_type, input_domain, input_ptr);
  Domain mask_domain = sub_mask.get_domain();
  void *mask_ptr = sim->allocate(sub_mask.get_volume(), inputs[1]->data_type);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW mask_acc(inputs[1]->data_type, mask_domain, mask_ptr);
  out_of_memory = out_of_memory || (input_ptr == NULL) || (mask_ptr == NULL);
  Domain out_domain = sub_output.get_domain();
  void *output_ptr =
      sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW output_acc(
      outputs[0]->data_type, out_domain, output_ptr);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  std::function<void()> forward, backward;
  forward = [&] { forward_kernel_wrapper(m, input_acc, mask_acc, output_acc); };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    backward = [&] {
      backward_kernel_wrapper(m, output_acc, mask_acc, input_acc);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure MaskedFill] name(%s) forward_time(%.4lf) "
           "backward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure MaskedFill] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::MaskedFillParams>::operator()(
    FlexFlow::MaskedFillParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.filled_value);
  return key;
}
}; // namespace std
