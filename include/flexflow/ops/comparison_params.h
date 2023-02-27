#ifndef _FLEXFLOW_COMPARISON_PARAMS_H
#define _FLEXFLOW_COMPARISON_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ComparisonParams {
  OperatorType type;

  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(ComparisonParams const &, ComparisonParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ComparisonParams> {
  size_t operator()(FlexFlow::ComparisonParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_COMPARISON_PARAMS_H
