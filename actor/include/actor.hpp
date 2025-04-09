#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include "othello.hpp"
#include "az_mcts.hpp"
#include "az_net.hpp"

struct Experience {
  std::vector<float> state;
  std::vector<float> policy;
  float reward;
};

int sample_from_policy(const std::vector<float> &policy, const float temperature);
float get_temperature(int move_count);

class Actor {
  public:
    Actor(AZNet &net, torch::Device &device, float C, int64_t simulations);

    std::vector<Experience> self_play();

    AZNet &net;

  private:
    MCTS mcts;
};
