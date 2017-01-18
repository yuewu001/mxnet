/*********************************************************************************
*     File Name           :     neuron_weighting-inl.h
*     Created By          :     yuewu
*     Creation Date       :     [2017-01-03 16:15]
*     Last Modified       :     [2017-01-18 15:15]
*     Description         :     neuron weighting for model simplification
**********************************************************************************/

#ifndef MXNET_OPERATOR_NEURON_WEIGHTING_INL_H__
#define MXNET_OPERATOR_NEURON_WEIGHTING_INL_H__

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace neuron {
enum NeuronWeightingOpInputs {kData, kWeights};
enum NeuronWeightingOpOutputs {kOut};
}  // neuron

/**
 * \brief This is the implementation of neuron weighting operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class NeuronWeightingOp : public Operator {
  void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(out_data.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[neuron::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> weights = in_data[neuron::kWeights].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[neuron::kOut].get<xpu, 4, DType>(s);

    Assign(out, req[neuron::kOut], data * broadcast<1>(weights, data.shape_));
  }

  void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    CHECK_EQ(req.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[neuron::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> weights = in_data[neuron::kWeights].FlatTo1D<xpu, DType>(s);

    Tensor<xpu, 4, DType> m_out_grad = out_grad[neuron::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> m_in_grad = in_grad[neuron::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> gweights = in_grad[neuron::kWeights].FlatTo1D<xpu, DType>(s);

    //gradient of data
    Assign(m_in_grad, req[neuron::kData], m_out_grad * broadcast<1>(weights, m_out_grad.shape_));

    //gradient of weights
    Assign(gweights, req[neuron::kWeights], sumall_except_dim<1>(m_out_grad * data));
  }
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateNeuronWeightingOp(int dtype);

#if DMLC_USE_CXX11
// OperatorProperty allows C++11, while Operator do not rely on it.
/*!
 * \brief OperatorProperty is a object that stores all information about Operator.
 * It also contains method to generate context(device) specific operators.
 *
 * It also contains various functions that can be optimally overriden to
 * provide optimization chance for computation engine.
 */
class NeuronWeightingProp: public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "weights"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, weights]";

    const TShape &dshape = (*in_shape)[neuron::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    SHAPE_ASSIGN_CHECK(*in_shape, neuron::kWeights, Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);

    return true;
  }

  bool InferType(std::vector<int> *in_type,
                          std::vector<int> *out_type,
                          std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);

    return true;
  }

  OperatorProperty* Copy() const override {
    NeuronWeightingProp* nw_prop = new NeuronWeightingProp();
    return nw_prop;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const override;
  std::string TypeString() const override {
    return "NeuronWeighting";
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[neuron::kOut], in_data[neuron::kData], in_data[neuron::kWeights]};
  }
};

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif
