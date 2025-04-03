#pragma once

#include <cstdint>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

class ResidualBlockImpl : public torch::nn::Module {
  public:
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1);
    torch::Tensor forward(torch::Tensor x);

  private:
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::ReLU relu;
};

torch::nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, int64_t stride = 1);

TORCH_MODULE(ResidualBlock);
