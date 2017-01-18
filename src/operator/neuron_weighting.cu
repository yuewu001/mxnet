/*********************************************************************************
*     File Name           :     neuron_weighting.cu
*     Created By          :     yuewu
*     Creation Date       :     [2017-01-04 16:49]
*     Last Modified       :     [2017-01-18 15:14]
*     Description         :      
**********************************************************************************/

#include "./neuron_weighting-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateNeuronWeightingOp<gpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NeuronWeightingOp<gpu, DType>();
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
