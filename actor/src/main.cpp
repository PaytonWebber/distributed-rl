#include "actor.hpp"
#include <vector>

int main() {
  AZNet net = AZNet(2, 64, 65, 5);
  Actor actor(net, 1.414, 100);
  
  std::vector<Experience> game_samples = actor.self_play();
  for (size_t i = 0; i < game_samples.size(); ++i) {
    std::cout << game_samples[i].reward << std::endl;
  }

  return 0;
}
