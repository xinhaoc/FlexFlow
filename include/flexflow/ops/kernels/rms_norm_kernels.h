#ifndef _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Gather;

class RMSNormMeta : public OpMeta {
public:
  RMSNormMeta(FFHandler handler, RMSNorm const *rms);

public:
  float eps;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace RMSNorm {
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight);
void backward_kernel_wrapper(RMSNormMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &weight,
                             GenericTensorAccessorW const &input_grad);
namespace Internal {

void forward_kernel(float const *input_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    Legion::coord_t output_size,
                    Legion::coord_t stride,
                    Legion::coord_t dim_size,
                    ffStream_t stream);
void backward_kernel(float const *output_grad_ptr,
                     float const *weight_ptr,
                     float *input_grad_ptr,
                     Legion::coord_t output_size,
                     Legion::coord_t stride,
                     Legion::coord_t dim_size,
                     ffStream_t stream);
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H