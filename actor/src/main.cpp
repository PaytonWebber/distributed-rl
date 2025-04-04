#include "actor.hpp"
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

void to_json(json& j, const Experience& e) {
    j = json{{"state", e.state}, {"policy", e.policy}, {"reward", e.reward}};
}

int main() {
  zmq::context_t ctx;
  zmq::socket_t sock(ctx, zmq::socket_type::push);
  sock.connect("tcp://localhost:5555");
  AZNet net = AZNet(2, 64, 65, 5);
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  net->to(device);

  Actor actor(net, device, 1.414, 100);

  while (true) {
    std::vector<Experience> experiences = actor.self_play();
    json j = experiences;
    std::string msg = j.dump();
    sock.send(zmq::buffer(msg), zmq::send_flags::none);
  }

  return 0;
}
