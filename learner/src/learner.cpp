#include "learner.hpp"
#include "az_net.hpp"
#include <torch/optim/adagrad.h>

Learner::Learner(AZNet &network, torch::Device &device, Config config)
    : network(network), device(device), config(config),
      optimizer(network->parameters(), torch::optim::AdamOptions(1e-3)) {}

void Learner::train_step(std::vector<Experience> &mini_batch) {
  network->train();

  std::vector<torch::Tensor> state_tensors;
  std::vector<torch::Tensor> policy_tensors;
  std::vector<float> rewards;

  for (size_t i = 0; i < mini_batch.size(); ++i) {
    auto state_tensor = torch::tensor(mini_batch[i].state).reshape({2, 6, 6});
    auto policy_tensor = torch::tensor(mini_batch[i].policy);
    rewards.push_back(mini_batch[i].reward);
    state_tensors.push_back(state_tensor);
    policy_tensors.push_back(policy_tensor);
  }

  auto state_batch = torch::stack(state_tensors).to(device);
  auto policy_batch = torch::stack(policy_tensors).to(device);
  auto rewards_batch = torch::tensor(rewards).to(device).unsqueeze(1);

  NetOutputs outputs = network->forward(state_batch);

  auto value_loss = torch::mse_loss(outputs.v, rewards_batch);
  auto log_probs = torch::log_softmax(outputs.pi, /*dim=*/1);
  auto policy_loss = -(policy_batch * log_probs).sum(1).mean();
  auto loss = value_loss + policy_loss;

  optimizer.zero_grad();
  loss.backward();
  optimizer.step();

  std::cout << std::fixed << " | Value Loss: " << std::setw(8)
            << value_loss.item<float>() << " | Policy Loss: " << std::setw(8)
            << policy_loss.item<float>() << " | Total Loss: " << std::setw(8)
            << loss.item<float>() << std::endl;
}

void Learner::save_chekpoint(int checkpoint) {
  std::string checkpoint_path = config.checkpoint_dir + "model_iter_" +
                                std::to_string(checkpoint) + ".pt";
  torch::save(network, checkpoint_path);
}

void Learner::load_chekpoint(int checkpoint) {
  std::string checkpoint_path = config.checkpoint_dir + "model_iter_" +
                                std::to_string(checkpoint) + ".pt";
  torch::load(network, checkpoint_path);
}
