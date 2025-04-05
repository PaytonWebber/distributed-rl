#include "actor.hpp"
#include <torch/serialize.h>
#include <torch/serialize/input-archive.h>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

void to_json(json& j, const Experience& e) {
    j = json{{"state", e.state}, {"policy", e.policy}, {"reward", e.reward}};
}

int main() {
  zmq::context_t ctx;
  zmq::socket_t push_sock(ctx, zmq::socket_type::push);
  push_sock.connect("tcp://localhost:5555");

  zmq::socket_t sub_sock(ctx, zmq::socket_type::sub);
  sub_sock.set(zmq::sockopt::conflate, 1);
  sub_sock.set(zmq::sockopt::subscribe, "");
  sub_sock.connect("tcp://localhost:5557");

  zmq::message_t params_buffer;

  auto result = sub_sock.recv(params_buffer, zmq::recv_flags::none);
  if (!result) {
    std::cerr << "Failed to receive message from socket." << std::endl;
    return 1;
  }
  std::string model_bytes(static_cast<char*>(params_buffer.data()), params_buffer.size());
  std::istringstream iss(model_bytes, std::ios::binary);
  torch::serialize::InputArchive archive;  
  archive.load_from(iss);

  AZNet net = AZNet(2, 64, 65, 5);
  net->load(archive);
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  net->to(device);

  Actor actor(net, device, 1.414, 100);

  while (true) {
    std::vector<Experience> experiences = actor.self_play();
    json j = experiences;
    std::string msg = j.dump();
    push_sock.send(zmq::buffer(msg), zmq::send_flags::none);

    auto result = sub_sock.recv(params_buffer, zmq::recv_flags::none);
    if (!result) {
      std::cerr << "Failed to receive message from socket." << std::endl;
      return 1;
    }

    std::string model_bytes(static_cast<char*>(params_buffer.data()), params_buffer.size());
    std::istringstream iss(model_bytes, std::ios::binary);
    torch::serialize::InputArchive archive;  
    archive.load_from(iss);
    actor.net->load(archive);
  }

  return 0;
}
