#include "az_mcts.hpp"
#include "az_net.hpp"
#include "minimax_bot.hpp"
#include "othello.hpp"
#include  <random>

int select_random_move(const std::vector<int>& legal_actions) {
    assert(!legal_actions.empty() && "No legal actions to choose from!");

    static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    
    std::uniform_int_distribution<std::size_t> dist(0, legal_actions.size() - 1);
    std::size_t idx = dist(gen);
    
    return legal_actions[idx];
}


int main(int argc, char *argv[]) {
  AZNet net = AZNet(2, 36, 37, 2);
  if (argc > 1) {
    const std::string checkpoint_path = argv[1];
    std::cout << "Loading model from: " << checkpoint_path << std::endl;
    torch::load(net, checkpoint_path);
  }

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  net->to(device);
  net->eval();

  MCTS az_mcts(std::ref(net), std::ref(device), 1.414, 200, false);
  Player az_player = Player::Black;

  int az_wins = 0;
  int minimax_wins = 0;
  int draws = 0;
  const int eval_games = 100;
  for (int game_num = 0; game_num < eval_games; ++game_num) {
    std::cout << "AZ Player: "
              << (az_player == Player::Black ? "Black" : "White") << std::endl;
    OthelloState state;
    int random_move = select_random_move(state.legal_actions());
    state = state.step(random_move);
    while (!state.is_terminal()) {
      if (state.current_player == az_player) {
        auto [best_move, _] = az_mcts.search(state);
        state = state.step(best_move);
      } else {
        int action = MinimaxBot::choose_move(state, 6);
        state = state.step(action);
      }
    }
    int reward = state.reward(az_player);
    switch (reward) {
    case -1:
      minimax_wins++;
      break;
    case 1:
      az_wins++;
      break;
    default:
      draws++;
    }
    az_player = (az_player == Player::Black ? Player::White : Player::Black);
  }
  std::cout << "AZ Wins: " << az_wins << std::endl;
  std::cout << "Minimax Wins: " << minimax_wins << std::endl;
  std::cout << "Draws: " << draws << std::endl;
  return 0;
}
