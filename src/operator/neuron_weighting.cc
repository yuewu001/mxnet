/*********************************************************************************
*     File Name           :     neuron_weighting.cc
*     Created By          :     yuewu
*     Creation Date       :     [2017-01-04 16:42]
*     Last Modified       :     [2017-01-18 15:15]
*     Description         :      
**********************************************************************************/

#include "./neuron_weighting-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateNeuronWeightingOp<cpu>(int dtype) {
  Operator *op = NULL;
  switch (dtype) {
    case mshadow::kFloat32:
      op = new NeuronWeightingOp<cpu, float>;
      break;
    case mshadow::kFloat64:
      op = new NeuronWeightingOp<cpu, double>;
      break;
    case mshadow::kFloat16:
      LOG(FATAL) << "float16 neuron weighting layer is currently"
        "only supported by CuDNN version.";
      break;
    default:
      LOG(FATAL) << "Unsupported type " << dtype;
  }

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *NeuronWeightingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateNeuronWeightingOp, (*in_type)[0]);
}

//DMLC_REGISTER_PARAMETER(NeuronWeightingParam);

MXNET_REGISTER_OP_PROPERTY(NeuronWeighting, NeuronWeightingProp)
  .describe(R"(Apply weighing to input neurons.
  It maps the input of shape `(batch_size, input_dims)` to the shape of
  `(batch_size, input_dims)`. Learnable parameters are the weights.)")
  .add_argument("data", "Symbol", "Input data to the NeuronWeightingOp.")
  .add_argument("weights", "Symbol", "Weighting vector.");
  }  // namespace op
}  // namespace mxnet
