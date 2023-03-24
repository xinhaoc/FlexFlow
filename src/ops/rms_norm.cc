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

#include "flexflow/ops/rms_norm.h"
#include "flexflow/model.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::Predicate;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::TaskArgument;
using Legion::TaskLauncher;

bool operator==(RMSNormParams const &lhs, RMSNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.eps = rhs.eps;
}

bool RMSNormParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

RMSNormParams RMSNorm::get_params() const {
  RMSNormParams params;
  params.layer_guid = this->layer_guid;
  params.eps = this->eps;
  return params;
}

Tensor FFModel::rms_norm(const Tensor input, float eps, char const *name) {
  Layer *rm = new Layer(this,
                        OP_RMS_NORM,
                        DT_FLOAT,
                        name,
                        1 /*inputs*/,
                        1 /*weights*/,
                        1 /*outputs*/,
                        input);
  rm->outputs[0] = create_tensor_legion_ordering(
      input->num_dims, input->dims, DT_FLOAT, rm, 0, true /*create_grad*/);

  // weights
  // TODO weight dims check
  rm->weights[0] = create_weight_legion_ordering(1,
                                                 input->dims,
                                                 input->data_type,
                                                 rm,
                                                 true /*create_grad*/,
                                                 nullptr,
                                                 CHOSEN_SYNC_TYPE);
  ln->add_float_property("eps", eps);
  layers.push_back(bm);
  return rm->outputs[0];
}

Op *RMSNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float eps;
  layer->get_int_property("eps", eps);
  return new RMSNorm(model, layer->layer_guid, inputs[0], eps, layer->name);
}

RMSNorm::RMSNorm(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor _input,
                 float _eps,
                 char const *name)
    : Op(model,
         OP_RMS_NORM,
         input->data_type,
         name,
         1 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         input),
{

  // output has the same parallel dims as input
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < input->num_dims; i++) {
    dims[i] = input->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      input->num_dims, dims, input->data_type, this);
  // weights
  Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
  weights[0] =
      model.create_parallel_weight_legion_ordering(input->num_dims,
                                                   dims,
                                                   input->data_type,
                                                   this /*owner_op*/,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
}

void RMSNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RMSNROM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(RMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *RMSNorm::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  RMSNorm *rn = (RMSNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  RMSNormMeta *meta = new RMSNormMeta(handle, rn);
  return meta;
}

void RMSNorm::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(RMSNROM_FWD_TASK_ID,
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
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): weight
*/
void RMSNorm::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(task->regions.size() == 3);
  assert(regions.size() == 3);
  RMSNormMeta const *m = *((RMSNormMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR index = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, index, output);

  //   RMSNorm::forward_kernel_wrapper<float>(
  //       m, in_ptr, out_ptr, gamma_ptr, beta_ptr);
}

void RMSNorm::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(RMSNROM_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);

  // regions[3](I): weight
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(3, FID_DATA);
  // regions[4](I/O): weight_grad
  launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region_grad));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void RMSNorm::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  RMSNormMeta const *m = *((RMSNormMeta **)task->local_args);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->weight_type[0], regions[3], task->regions[3], FID_DATA, ctx, runtime);
  backward_kernel_wrapper(m, output_grad, index, input_grad);
}

bool RMSNorm::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_index)) {
    return false;
  }
  RMSNormMeta *m = new RMSNormMeta(sim->handler, this);

  sim->free_all();
  void *in_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  assert(in_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  void *out_ptr = sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  assert(out_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  //   // FIXME weight_ptr
  //   void *weight_ptr = sim->allocate(sub_output.get_volume(),
  //   outputs[0]->data_type);

  bool out_of_memory = (in_ptr == NULL) || (out_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, in_ptr, out_ptr, gamma_ptr, beta_ptr);
  };

  if (sim->computationMode == COMP_MODE_TRAINING) {
    void *in_grad_ptr = sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(in_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *out_grad_ptr = NULL;
    out_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(out_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    float *gamma_grad_ptr = NULL, *beta_grad_ptr = NULL;

    out_of_memory = (in_grad_ptr == NULL) || (out_grad_ptr == NULL) ||
                    (((gamma_grad_ptr == NULL) || (beta_grad_ptr == NULL)) &&
                     (m->elementwise_affine));
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }

    backward = [&] {
      backward_kernel_wrapper<float>(m,
                                     out_grad_ptr,
                                     in_ptr,
                                     in_grad_ptr,
                                     gamma_ptr,
                                     gamma_grad_ptr,
                                     beta_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure RMSNorm] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure RMSNorm] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time);
  }

  return true;
}

void RMSNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->eps);
}

using PCG::Node;
/*static*/
Node RMSNorm::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  float eps;
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  dez.deserialize(eps);

  RMSParams params;
  params.layer_guid = layer_guid;
  params.eps = eps;
  return ff.get_or_create_node<RMSNorm>(inputs[0], params);
}

Op *LayerNorm::materialize(FFModel &ff,
                           ParallelTensor inputs[],
                           int num_inputs) const {
  RMSParams params = get_params();
  return new RMSNorm(
      ff, params, inputs[0], this->name, true /*allocate_weights*/);
}

} // namespace FlexFlow