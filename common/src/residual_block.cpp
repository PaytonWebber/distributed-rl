#include "residual_block.hpp"

ResidualBlockImpl::ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride) :
  conv1(conv3x3(in_channels, out_channels, stride)),
  relu(torch::nn::ReLU()),
  bn1(out_channels),
  conv2(conv3x3(in_channels, out_channels, stride)),
  bn2(out_channels) {
  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("relu", relu);
  register_module("conv2", conv2);
  register_module("bn2", bn2);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
  auto out = conv1->forward(x);
  out = bn1->forward(out);
  out = relu->forward(out);
  out = conv2->forward(out);
  out = bn2->forward(out);
  out += x;
  out = relu->forward(out);

  return out;
}

torch::nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(1)
        .bias(false));
}
