#include "actor.hpp"

int sample_from_policy(const std::vector<float> &policy) {
  static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::discrete_distribution<> dist(policy.begin(), policy.end());
  return dist(gen);
}

Actor::Actor(AZNet net, float C, int64_t simulations) :
  mcts(net, C, simulations, true) {}

std::vector<Experience> Actor::self_play() {
  std::vector<Experience> game_samples;
  std::vector<Player> sample_players;
  OthelloState state;

  while (!state.is_terminal()) {
    auto [_, policy] = mcts.search(state);
    int action = sample_from_policy(policy);

    Experience experience;
    experience.state = state.board();
    experience.policy = policy;
    experience.reward = 0.0;
    game_samples.push_back(experience);
    sample_players.push_back(state.current_player);

    state = state.step(action);
  }

  int final_reward = state.reward(Player::Black);
  for (size_t i = 0; i < game_samples.size(); ++i) {
    game_samples[i].reward = (sample_players[i] == Player::Black) ? final_reward : -final_reward;
  }

  return game_samples;
}
