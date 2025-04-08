#include "az_net.hpp"
#include "learner.hpp"
#include <ATen/Context.h>
#include <torch/serialize.h>
#include <nlohmann/json.hpp>
#include <torch/serialize/output-archive.h>
#include <zmq.hpp>
#include <sstream>
#include <string>

using json = nlohmann::json;

void from_json(const json& j, Experience& e) {
    j.at("state").get_to(e.state);
    j.at("policy").get_to(e.policy);
    j.at("reward").get_to(e.reward);
}

void serialize_parameters(AZNet &net, std::string& payload) {
      torch::serialize::OutputArchive archive;  
      std::ostringstream oss;
      net->to(torch::kCPU);
      net->save(archive);
      archive.save_to(oss);
      payload = oss.str();
}

int main(int argc, char* argv[]) {
  const std::string OK = "OK";
  const std::string NO = "NO";
  std::string const server_ip = "tcp://hq.servebeer.com:";
  std::string const port = "5556";

  Config config {1e-3, "models/"};
  std::filesystem::create_directories(config.checkpoint_dir);

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

  AZNet net = AZNet(2, 36, 37, 2);
  net->to(device);
  Learner learner(net, device, config);

  int checkpoint = 0;
  if (argc > 1) {
    checkpoint = std::stoi(argv[1]);
    std::cout << "Loading model from checkpoint: " << checkpoint << std::endl;
    learner.load_chekpoint(checkpoint);
  }

  zmq::context_t ctx;
  zmq::socket_t sock(ctx, zmq::socket_type::req);
  
  sock.connect(server_ip + port);

  zmq::message_t ack;
  zmq::message_t batch_buffer;

  // send (UPDATE)
  sock.send(zmq::buffer("SENDING_PARAMETERS"), zmq::send_flags::none);

  // receive (ACK)
  auto result = sock.recv(ack, zmq::recv_flags::none); 

  std::string serialized_parameters;
  serialize_parameters(std::ref(learner.network), serialized_parameters);
  learner.network->to(device);
  
  // send (PARAMETERS)
  std::cout << "Sending parameters to server..." << std::endl;
  sock.send(zmq::buffer(serialized_parameters), zmq::send_flags::none);

  // receive (ACK)
  result = sock.recv(ack, zmq::recv_flags::none); 
  std::cout << "Parameters received by server." << std::endl;

  int step_count = 0;
  while (true) {

    /**************************************/
    /*********** CONTROL FLOW 1 ***********/
    /**************************************/
    
    // send (REQUEST)
    sock.send(zmq::buffer("REQUEST_BATCH"), zmq::send_flags::none);

    // receive (ACK)
    auto result = sock.recv(ack, zmq::recv_flags::none);
    if (!result) {
      std::cerr << "Failed to receive message from socket." << std::endl;
      return 1;
    }

    if (ack.to_string() == NO) { continue; }

    /**************************************/
    /*********** CONTROL FLOW 2 ***********/
    /**************************************/

    // send  (ACK)
    sock.send(zmq::buffer("OK"), zmq::send_flags::none);

    // receive (MINi-BATCH)
    result = sock.recv(batch_buffer, zmq::recv_flags::none);
    if (!result) {
      std::cerr << "Failed to receive message from socket." << std::endl;
      return 1;
    }

    std::string json_str(static_cast<char *>(batch_buffer.data()), batch_buffer.size());
    json j = json::parse(json_str);
    std::vector<Experience> experiences = j.get<std::vector<Experience>>();
    learner.train_step(experiences);
    step_count++;
    checkpoint++;
    std::cout << step_count << std::endl;

    /**************************************/
    /*********** CONTROL FLOW 3 ***********/
    /**************************************/

    if (step_count == 1000) {
      // send (UPDATE)
      sock.send(zmq::buffer("SENDING_PARAMETERS"), zmq::send_flags::none);

      // receive (ACK)
      auto result = sock.recv(ack, zmq::recv_flags::none); 

      serialize_parameters(std::ref(learner.network), serialized_parameters);
      learner.network->to(device);

      // send (PARAMETERS)
      sock.send(zmq::buffer(serialized_parameters), zmq::send_flags::none);

      // receive (ACK)
      result = sock.recv(ack, zmq::recv_flags::none); 

      learner.save_chekpoint(checkpoint);
      step_count = 0;
    }
  }

  return 0;
}
