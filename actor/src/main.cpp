#include "actor.hpp"
#include "message_queue.hpp"

#include <atomic>
#include <torch/serialize.h>
#include <torch/serialize/input-archive.h>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <zmq.hpp>

using json = nlohmann::json;

void to_json(json& j, const Experience& e) {
    j = json{{"state", e.state}, {"policy", e.policy}, {"reward", e.reward}};
}

void sub_thread_fn(zmq::context_t &ctx, MessageQueue<std::string> &queue,
                   const std::string &server_ip, const std::string &sub_port,
                   std::atomic<bool> &run) {
  zmq::socket_t sub_sock(ctx, zmq::socket_type::sub);
  sub_sock.set(zmq::sockopt::conflate, 1);
  sub_sock.set(zmq::sockopt::subscribe, "");
  sub_sock.connect(server_ip + sub_port);

  while (run) {
    zmq::message_t param_buffer;
    auto result = sub_sock.recv(param_buffer, zmq::recv_flags::none);
    if (!result) { sleep(1); }
    else {
      std::string model_bytes(static_cast<char *>(param_buffer.data()),
                              param_buffer.size());
      queue.push(std::move(model_bytes));
    }
  }
}

void update_params(AZNet &net, std::string &param_bytes) {
  std::istringstream iss(param_bytes, std::ios::binary);
  torch::serialize::InputArchive archive;
  archive.load_from(iss);
  net->load(archive);
}

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

  zmq::context_t ctx;
  std::string const server_ip = "tcp://hq.servebeer.com:";
  std::string const push_port = "5555";
  std::string const sub_port = "5557";

  zmq::socket_t push_sock(ctx, zmq::socket_type::push);
  push_sock.connect(server_ip + push_port);

  MessageQueue<std::string> param_queue;
  std::atomic<bool> run(true);

  std::thread sub_thread(sub_thread_fn, std::ref(ctx), std::ref(param_queue),
                         server_ip, sub_port, std::ref(run));

  std::string param_bytes;
  while (!param_queue.pop(param_bytes)) {
    std::cout << "Waiting for model parameters... " << std::endl;
    sleep(5);
  }
  std::cout << "Received parameters. Starting self-play... " << std::endl;

  AZNet net = AZNet(2, 64, 65, 5);
  net->to(device);
  update_params(std::ref(net), param_bytes);
  Actor actor(std::ref(net), std::ref(device), 1.414, 400);

  int games_generated = 0;
  while (true) {
    std::vector<Experience> experiences = actor.self_play();
    json j = experiences;
    std::string msg = j.dump();
    push_sock.send(zmq::buffer(msg), zmq::send_flags::none);

    if (games_generated % 5) {
      std::cout << "Total Games Generated: " << games_generated << std::endl;
      bool update = false;
      while (!param_queue.empty()) {
        update = param_queue.pop(param_bytes);
      }
      if (update) {
        update_params(std::ref(net), param_bytes);
      }
    }
    games_generated++;
  }

  run = false;
  if (sub_thread.joinable()) { sub_thread.join(); }

  return 0;
}
