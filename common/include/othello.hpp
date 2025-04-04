#pragma once

#include <cinttypes>
#include <iostream>
#include <vector>
#include <torch/torch.h>


enum class Player { Black, White };

enum class Cell { Empty, Black, White };

inline Player other(Player p) {
  return (p == Player::Black) ? Player::White : Player::Black;
}

class OthelloState {
public:
  static constexpr int SIZE = 8;
  static constexpr int PASS = 64;

  // row-major order
  uint64_t bitboard_black;
  uint64_t bitboard_white;

  Player current_player;
  unsigned pass_count; // number of consecutive passes (2 means game over)

  static const std::vector<int> &pass_vector() {
    static std::vector<int> pass_vector = {-1};
    return pass_vector;
  }
  static const std::vector<std::pair<int, int>> &directions() {
    static std::vector<std::pair<int, int>> dirs = {{-1, -1}, {-1, 0}, {-1, +1},
                                                    {0, -1},  {0, +1}, {+1, -1},
                                                    {+1, 0},  {+1, +1}};
    return dirs;
  }

  OthelloState()
      : bitboard_black(0), bitboard_white(0), current_player(Player::Black),
        pass_count(0) {
    // center positions
    set_cell(3, 3, Player::White);
    set_cell(3, 4, Player::Black);
    set_cell(4, 3, Player::Black);
    set_cell(4, 4, Player::White);
  }

  bool is_terminal() const {
    if (pass_count == 2) {
      return true;
    }
    if (legal_actions() == pass_vector() &&
        legal_actions_for_player(other(current_player)) == pass_vector()) {
      return true;
    }
    return false;
  }

  float reward(Player to_play) const {
    if (!is_terminal()) {
      return 0;
    }
    int black_count = __builtin_popcountll(bitboard_black);
    int white_count = __builtin_popcountll(bitboard_white);
    if (black_count == white_count) {
      return 0.0;
    }
    if (to_play == Player::Black) {
      return (black_count > white_count) ? 1.0 : -1.0;
    } else {
      return (white_count > black_count) ? 1.0 : -1.0;
    }
  }

  OthelloState step(int action) const {
    OthelloState new_state(*this);
    if (action == PASS) {
      new_state.current_player = other(current_player);
      new_state.pass_count++;
      return new_state;
    }

    new_state.pass_count = 0;
    int r = action / SIZE;
    int c = action % SIZE;
    int idx = r * SIZE + c;
    uint64_t mask = 1ULL << idx;
    Player p = current_player;
    Player opp = other(p);

    if (p == Player::Black) { new_state.bitboard_black |= mask; }
    else { new_state.bitboard_white |= mask; }
    
    // for each direction, check for flips
    for (auto d : directions()) {
      int dr = d.first, dc = d.second;
      int cur_r = r + dr, cur_c = c + dc;
      std::vector<int> positions_to_flip;
      // the first cell in this direction must be an opponent piece
      if (!on_board(cur_r, cur_c)) { continue; }
      if (get_cell(cur_r, cur_c) != cell_from_player(opp)) { continue; }
      // collect consecutive opponent pieces
      while (on_board(cur_r, cur_c) &&
             get_cell(cur_r, cur_c) == cell_from_player(opp)) {
        positions_to_flip.push_back(cur_r * SIZE + cur_c);
        cur_r += dr;
        cur_c += dc;
      }
      // check if we end with one of our own pieces
      if (on_board(cur_r, cur_c) &&
          get_cell(cur_r, cur_c) == cell_from_player(p)) {
        // flip the collected opponent pieces
        for (int pos : positions_to_flip) {
          uint64_t posMask = 1ULL << pos;
          if (p == Player::Black) {
            new_state.bitboard_black |= posMask;
            new_state.bitboard_white &= ~posMask;
          } else {
            new_state.bitboard_white |= posMask;
            new_state.bitboard_black &= ~posMask;
          }
        }
      }
    }
    new_state.current_player = opp;
    return new_state;
  }

  void render() const {
    std::cout << "  A B C D E F G H\n";
    for (int r = 0; r < SIZE; ++r) {
      std::cout << r + 1 << " ";
      for (int c = 0; c < SIZE; ++c) {
        Cell cell = get_cell(r, c);
        char ch = '.';
        if (cell == Cell::Black)
          ch = 'X';
        else if (cell == Cell::White)
          ch = 'O';
        std::cout << ch << " ";
      }
      std::cout << "\n";
    }
    int black_count = __builtin_popcountll(bitboard_black);
    int white_count = __builtin_popcountll(bitboard_white);
    std::cout << "Current scores:\n"
              << "Black - "
              << black_count
              << "\n"
              << "White - "
              << white_count
              << "\n";
    std::cout << "Next to move: "
              << (current_player == Player::Black ? "Black (X)" : "White (O)")
              << "\n";
  }

  std::vector<float> board() const {
    std::vector<float> rep((SIZE * SIZE) * 2, 0.0); // two channels: one for each player's pieces
    int black_offset = (current_player == Player::Black ? 0 : 64);
    int white_offset = (current_player == Player::White ? 0 : 64);
    for (int i = 0; i < (SIZE * SIZE); ++i) {
      if ((bitboard_black >> i) & 1ULL) {
        rep[i+black_offset] = 1.0;
      }
      if ((bitboard_white >> i) & 1ULL) {
        rep[i+white_offset] = 1.0;
      }
    }
    return rep;
  }

  Cell get_cell(int row, int col) const {
    int idx = row * SIZE + col;
    uint64_t mask = 1ULL << idx;
    if (bitboard_black & mask) {
      return Cell::Black;
    }
    if (bitboard_white & mask) {
      return Cell::White;
    }
    return Cell::Empty;
  }

  static bool on_board(int row, int col) {
    return (row >= 0 && row < SIZE && col >= 0 && col < SIZE);
  }

  void set_cell(int row, int col, Player p) {
    int idx = row * SIZE + col;
    uint64_t mask = 1ULL << idx;
    if (p == Player::Black) {
      bitboard_black |= mask;
      bitboard_white &= ~mask;
    } else {
      bitboard_white |= mask;
      bitboard_black &= ~mask;
    }
  }

  std::vector<int> legal_actions() const {
    return legal_actions_for_player(current_player);
  }

  std::vector<int> legal_actions_for_player(Player player) const {
    std::vector<int> actions;
    for (int r = 0; r < SIZE; ++r) {
      for (int c = 0; c < SIZE; ++c) {
        if (get_cell(r, c) != Cell::Empty) { continue; }
        if (is_legal_move(r, c, player)) { actions.push_back(r * SIZE + c); }
      }
    }
    if (actions.empty()) { actions.push_back(PASS); } // only legal move is a pass
    return actions;
  }

  static Cell cell_from_player(Player p) {
    return (p == Player::Black) ? Cell::Black : Cell::White;
  }

  bool is_legal_move(int r, int c, Player player) const {
    Player opp = other(player);
    if (get_cell(r, c) != Cell::Empty) { return false; }
    for (auto d : directions()) {
      int dr = d.first, dc = d.second;
      int cur_r = r + dr, cur_c = c + dc;
      if (!on_board(cur_r, cur_c)) { continue; }
      if (get_cell(cur_r, cur_c) != cell_from_player(opp)) { continue; }
      cur_r += dr;
      cur_c += dc;
      while (on_board(cur_r, cur_c)) {
        Cell cell = get_cell(cur_r, cur_c);
        if (cell == Cell::Empty) { break; }
        if (cell == cell_from_player(player)) { return true; }// found a valid capture line
        cur_r += dr;
        cur_c += dc;
      }
    }
    return false;
  }
};
