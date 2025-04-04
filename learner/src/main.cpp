#include "learner.hpp"
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

void from_json(const json& j, Experience& e) {
    j.at("state").get_to(e.state);
    j.at("policy").get_to(e.policy);
    j.at("reward").get_to(e.reward);
}

int main() {
  Config config {1e-3, "models/"};
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  AZNet net = AZNet(2, 64, 65, 5);
  net->to(device);
  Learner learner(net, device, config);
  zmq::context_t ctx;
  zmq::socket_t sock(ctx, zmq::socket_type::req);
  sock.connect("tcp://localhost:5556");

  while (true) {
    sock.send(zmq::buffer("REQUEST_BATCH"), zmq::send_flags::none);

    zmq::message_t msg;
    auto result = sock.recv(msg, zmq::recv_flags::none);
    if (!result) {
      std::cerr << "Failed to receive message from socket." << std::endl;
      return 1;
    }

    std::string json_str(static_cast<char *>(msg.data()), msg.size());
    json j = json::parse(json_str);

    std::vector<Experience> experiences = j.get<std::vector<Experience>>();

    std::cout << "Received " << experiences.size() << " experiences."
              << std::endl;

    learner.train_step(experiences);
  }

  return 0;
}
