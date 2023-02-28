#ifndef _FLEXFLOW_OPS_KERNELS_MASKED_FILL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_MASKED_FILL_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class MaskedFill;

class MaskedFillMeta : public OpMeta {
public:
  MaskedFillMeta(FFHandler handler, MaskedFill const *masked_fill);

public:
  float filled_value;
};

namespace Kernels {
namespace MaskedFill {
void forward_kernel_wrapper(MaskedFillMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &mask,
                            GenericTensorAccessorW const &output);
void backward_kernel_wrapper(MaskedFillMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &mask,
                             GenericTensorAccessorW const &input_grad);
namespace Internal {
template <typename IndexType>
void forward_kernel(float const *input_ptr,
                    IndexType const *index_ptr,
                    float *output_ptr,
                    Legion::coord_t output_size,
                    Legion::coord_t stride,
                    Legion::coord_t dim_size,
                    ffStream_t stream);
template <typename IndexType>
void backward_kernel(float const *output_grad_ptr,
                     IndexType const *index_ptr,
                     float *input_grad_ptr,
                     Legion::coord_t output_size,
                     Legion::coord_t stride,
                     Legion::coord_t dim_size,
                     ffStream_t stream);
} // namespace Internal
} // namespace MaskedFill
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_MASKED_FILL_KERNELS_H
