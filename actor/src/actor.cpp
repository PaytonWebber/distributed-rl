#include "actor.hpp"

int sample_from_policy(const std::vector<float> &policy,
                       const float temperature) {
  static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));

  if (temperature == 0.0f) {
    return std::distance(policy.begin(),
                         std::max_element(policy.begin(), policy.end()));
  }

  std::vector<float> adjusted;
  for (const auto p : policy) {
    adjusted.push_back(std::pow(p, 1.0f / temperature));
  }

  std::discrete_distribution<> dist(adjusted.begin(), adjusted.end());
  return dist(gen);
}

float get_temperature(int move_count) {
    const int early_move_threshold = 20;
    const int mid_move_threshold = 40;

    if (move_count < early_move_threshold) {
        return 1.0f;
    } else if (move_count < mid_move_threshold) {
        // linearly decay temperature from 1.0 to 0.0 between early and mid moves
        return 1.0f - float(move_count - early_move_threshold) / float(mid_move_threshold - early_move_threshold);
    } else {
        return 0.0f;  // deterministic (argmax) in the late game
    }
}

Actor::Actor(AZNet &net, torch::Device &device, float C, int64_t simulations)
    : net(net), mcts(std::ref(net), device, C, simulations, true) {}

std::vector<Experience> Actor::self_play() {
  std::vector<Experience> game_samples;
  std::vector<Player> sample_players;
  OthelloState state;

  int move_count = 0;
  while (!state.is_terminal()) {
    auto [_, policy] = mcts.search(state);

    float temperature = get_temperature(move_count);

    int action = sample_from_policy(policy, temperature);

    Experience experience;
    experience.state = state.board();
    experience.policy = policy;
    experience.reward = 0.0;
    game_samples.push_back(experience);
    sample_players.push_back(state.current_player);

    state = state.step(action);
    move_count++;
  }

  int final_reward = state.reward(Player::Black);
  for (size_t i = 0; i < game_samples.size(); ++i) {
    game_samples[i].reward =
        (sample_players[i] == Player::Black) ? final_reward : -final_reward;
  }

  return game_samples;
}
