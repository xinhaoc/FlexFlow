#ifndef _FLEXFLOW_MASKED_FILL_PARAMS_H
#define _FLEXFLOW_MASKED_FILL_PARAMS_H
_FLEXFLOW_MASKED_FILL_PARAMS_H
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct MaskedFillParams {
  float filled_value;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(MaskedFillParams const &, MaskedFillParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::MaskedFillParams> {
  size_t operator()(FlexFlow::MaskedFillParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_MASKED_FILL_PARAMS_H
