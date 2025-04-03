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

int sample_from_policy(const std::vector<float> &policy);

class Actor {
  public:
    Actor(AZNet net, float C, int64_t simulations);

    std::vector<Experience> self_play();

  private:
    MCTS mcts;
    
};
