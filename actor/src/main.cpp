#include "actor.hpp"
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

void to_json(json& j, const Experience& e) {
    j = json{{"state", e.state}, {"policy", e.policy}, {"reward", e.reward}};
}

int main() {
  AZNet net = AZNet(2, 64, 65, 5);

  Actor actor(net, 1.414, 100);
  std::vector<Experience> experiences = actor.self_play();

  json j = experiences;
  std::string msg = j.dump();
  
  zmq::context_t ctx;
  zmq::socket_t sock(ctx, zmq::socket_type::push);
  sock.connect("tcp://localhost:5555");
  sock.send(zmq::buffer(msg), zmq::send_flags::none);

  return 0;
}
