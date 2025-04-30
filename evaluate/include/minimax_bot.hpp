#pragma once
#include "othello.hpp"
#include <limits>
#include <array>

class MinimaxBot {
public:
  static int choose_move(const OthelloState& root, int depth = 4) {
    Player perspective = root.current_player;
    int alpha = std::numeric_limits<int>::min();
    int beta  = std::numeric_limits<int>::max();
    int best_action = OthelloState::PASS;

    for (int a : root.legal_actions()) {
      int v = minimax(root.step(a), depth - 1, alpha, beta, perspective);
      if (v > alpha) {
        alpha = v;
        best_action = a;
      }
    }
    return best_action;
  }

private:
  static inline const std::array<int, 36> PST = {
    +30, -12, +0,  +0, -12, +30,
    -12, -15, -3,  -3, -15, -12,
     +0,  -3, +1,  +1,  -3,  +0,
     +0,  -3, +1,  +1,  -3,  +0,
    -12, -15, -3,  -3, -15, -12,
    +30, -12, +0,  +0, -12, +30
  };

  static int evaluate(const OthelloState& s, Player perspective) {
    if (s.is_terminal())
      return static_cast<int>(1000000 * s.reward(perspective));

    int black_count = __builtin_popcountll(s.bitboard_black);
    int white_count = __builtin_popcountll(s.bitboard_white);
    int piece_diff = (perspective == Player::Black)
                     ? (black_count - white_count)
                     : (white_count - black_count);

    // mobility
    int my_moves  = static_cast<int>(s.legal_actions_for_player(perspective).size());
    int opp_moves = static_cast<int>(s.legal_actions_for_player(other(perspective)).size());
    int mobility  = (my_moves - opp_moves);

    // positional score
    int pst_score = 0;
    for (int idx = 0; idx < 36; ++idx) {
      uint64_t mask = 1ULL << idx;
      if (s.bitboard_black & mask)
        pst_score += (perspective == Player::Black ? PST[idx] : -PST[idx]);
      else if (s.bitboard_white & mask)
        pst_score += (perspective == Player::White ? PST[idx] : -PST[idx]);
    }

    return 10 * piece_diff +  5 * mobility + pst_score;
  }

  static int minimax(const OthelloState& state,
                     int depth,
                     int alpha,
                     int beta,
                     Player perspective) {
    if (depth == 0 || state.is_terminal())
      return evaluate(state, perspective);

    bool maximizing = (state.current_player == perspective);
    const auto& actions = state.legal_actions();

    if (maximizing) {
      int value = std::numeric_limits<int>::min();
      for (int a : actions) {
        value = std::max(value,
                         minimax(state.step(a), depth - 1, alpha, beta, perspective));
        alpha = std::max(alpha, value);
        if (alpha >= beta) { break; }
      }
      return value;
    } else {
      int value = std::numeric_limits<int>::max();
      for (int a : actions) {
        value = std::min(value,
                         minimax(state.step(a), depth - 1, alpha, beta, perspective));
        beta = std::min(beta, value);
        if (beta <= alpha) { break; }
      }
      return value;
    }
  }
};
