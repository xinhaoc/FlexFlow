#ifndef _FLEXFLOW_OPS_KERNELS_COMPARISON_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMPARISON__KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Gather;

class ComparisonMeta : public OpMeta {
public:
  ComparisonMeta(FFHandler handler, Gather const *gather);

public:
  int legion_dim;
};

namespace Kernels {
namespace Gather {
void forward_kernel_wrapper(ComparisonMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &index,
                            GenericTensorAccessorW const &output);
void backward_kernel_wrapper(ComparisonMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &index,
                             GenericTensorAccessorW const &input_grad);
namespace Internal {

void forward_kernel(ComparisonMeta const *m,
                    float const *in1_ptr,
                    float const *in2_ptr,
                    float *out_ptr,
                    ffStream_t stream);
void backward_kernel(ComparisonMeta const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr,
                     ffStream_t stream);

} // namespace Internal
} // namespace Comparison
} // namespace Kernels
} // namespace FlexFlow


#endif // _FLEXFLOW_OPS_KERNELS_COMPARISON__KERNELS_H
