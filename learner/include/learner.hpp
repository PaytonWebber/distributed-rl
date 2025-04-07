#pragma once

#include <vector>
#include "az_net.hpp"
#include <torch/torch.h>

struct Experience {
  std::vector<float> state;
  std::vector<float> policy;
  float reward;
};

struct Config {
  const float learning_rate;
  const std::string checkpoint_dir;
};

class Learner {
  public:
    Learner(AZNet &network, torch::Device &device, Config config);

    void train_step(std::vector<Experience> &mini_batch);

    AZNet network;

  private:
    torch::Device device;
    Config config;
    torch::optim::Adam optimizer;
};
