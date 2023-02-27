#ifndef _FLEXFLOW_COMPARISON_H
#define _FLEXFLOW_COMPARISON_H

#include "flexflow/model.h"
#include "flexflow/ops/comparison_params.h"

namespace FlexFlow {

class Comparison : public Op {
public:
  using Params = ComparisonParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;

  Comparison(FFModel &model,
                OperatorType type,
                const ParallelTensor x,
                const ParallelTensor y,
                bool inplace_a,
                char const *name);
  Comparison(FFModel &model,
                Params const &params,
                Input const &inputs,
                char const *name = nullptr,
                bool inplace_a = false);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  void map_output_tensors(FFModel &model) override;
  bool can_inplace_output() override;
  bool has_inplace_output() override;
  void do_inplace_output() override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  Params get_params() const;

public:
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_COMPARISON_H
