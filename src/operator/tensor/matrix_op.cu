/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile GPU version
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

NNVM_REGISTER_OP(crop)
.set_attr<FCompute>("FCompute<gpu>", Crop<gpu>);

NNVM_REGISTER_OP(_crop_assign)
.set_attr<FCompute>("FCompute<gpu>", CropAssign<gpu>);

NNVM_REGISTER_OP(_crop_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", CropAssignScalar<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", Slice<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceGrad_<gpu>);

NNVM_REGISTER_OP(flip)
.set_attr<FCompute>("FCompute<gpu>", Flip<gpu>);

NNVM_REGISTER_OP(dot)
.set_attr<FCompute>("FCompute<gpu>", DotForward_<gpu>);

NNVM_REGISTER_OP(_backward_dot)
.set_attr<FCompute>("FCompute<gpu>", DotBackward_<gpu>);

NNVM_REGISTER_OP(batch_dot)
.set_attr<FCompute>("FCompute<gpu>", BatchDotForward_<gpu>);

NNVM_REGISTER_OP(_backward_batch_dot)
.set_attr<FCompute>("FCompute<gpu>", BatchDotBackward_<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(trunc)
.set_attr<FCompute>("FCompute<gpu>", Trunc<gpu>);

NNVM_REGISTER_OP(_backward_trunc)
.set_attr<FCompute>("FCompute<gpu>", TruncGrad_<gpu>);

NNVM_REGISTER_OP(trunc_array)
.set_attr<FCompute>("FCompute<gpu>", TruncArray<gpu>);

NNVM_REGISTER_OP(_backward_trunc_array)
.set_attr<FCompute>("FCompute<gpu>", TruncArrayGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet
