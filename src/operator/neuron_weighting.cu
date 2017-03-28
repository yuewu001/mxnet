/*********************************************************************************
*     File Name           :     neuron_weighting.cu
*     Created By          :     yuewu
*     Creation Date       :     [2017-01-04 16:49]
*     Last Modified       :     [2017-02-16 16:01]
*     Description         :      
**********************************************************************************/

#include "./neuron_weighting-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateNeuronWeightingOp<gpu>(int dtype, NeuronWeightingParam param) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NeuronWeightingOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
