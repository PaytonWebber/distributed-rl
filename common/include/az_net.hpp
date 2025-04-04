#pragma once

#include <cstdint>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/torch.h>
#include "residual_block.hpp"

struct NetOutputs {
  torch::Tensor pi;
  torch::Tensor v;
};

class AZNetImpl : public torch::nn::Module {
  public:
    AZNetImpl(int64_t in_channels, int64_t board_size, int64_t policy_size, int64_t blocks = 10); 
    NetOutputs forward(torch::Tensor x);

  private:
    torch::nn::Conv2d conv_in, conv_p, conv_v;
    torch::nn::Linear fc_p, fc1_v, fc2_v;
    torch::nn::BatchNorm2d bn_in, bn_p, bn_v;
    torch::nn::Sequential blocks;
    torch::nn::ReLU relu;
};

torch::nn::Sequential make_blocks(int64_t channels = 32, int64_t blocks = 5);
torch::nn::Conv2d conv1x1(int64_t in_channels, int64_t out_channels, int64_t stride = 1);

TORCH_MODULE(AZNet);
