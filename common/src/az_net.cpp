#include "az_net.hpp"
#include "residual_block.hpp"
#include <torch/nn/options/conv.h>

AZNetImpl::AZNetImpl(int64_t in_channels, int64_t board_size, int64_t policy_size, int64_t num_blocks) :
  conv_in(conv3x3(in_channels, 256)),
  bn_in(256),
  blocks(make_blocks(256, num_blocks)),
  conv_p(conv1x1(256, 2)),
  bn_p(2),
  fc_p(board_size*2, policy_size),
  conv_v(conv1x1(256, 1)),
  bn_v(1),
  fc1_v(board_size, 128),
  fc2_v(128, 1),
  relu(torch::nn::ReLU()) {
  // encoding
  register_module("conv_in", conv_in);
  register_module("bn_in", bn_in);

  register_module("blocks", blocks);

  register_module("conv_p", conv_p);
  register_module("bn_p", bn_p);
  register_module("fc_p", fc_p);

  register_module("conv_v", conv_v);
  register_module("bn_v", bn_v);
  register_module("fc1_v", fc1_v);
  register_module("fc2_v", fc2_v);

  register_module("relu", relu);
}

NetOutputs AZNetImpl::forward(torch::Tensor x) {
  // encoding
  auto out = conv_in->forward(x);
  out = bn_in->forward(out);
  out = relu->forward(out);
  // resnet
  out = blocks->forward(out);
  // policy head
  auto p = conv_p->forward(out);
  p = bn_p->forward(p);
  p = relu->forward(p);
  p = p.flatten(1);
  p = fc_p->forward(p);
  // value head
  auto v = conv_v->forward(out);
  v = bn_v->forward(v);
  v = relu->forward(v);
  v = v.flatten(1);
  v = fc1_v->forward(v);
  v = relu->forward(v);
  v = fc2_v->forward(v);
  v = torch::tanh(v);

  NetOutputs outputs = {p, v};
  return outputs;
}

torch::nn::Sequential make_blocks(int64_t channels, int64_t blocks) {
  torch::nn::Sequential layers;
  for (int64_t i = 1; i != blocks; ++i) {
    layers->push_back(ResidualBlock(channels, channels));
  }
  return layers;
}  

torch::nn::Conv2d conv1x1(int64_t in_channels, int64_t out_channels, int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(stride)
        .padding(0)
        .bias(false));
}
