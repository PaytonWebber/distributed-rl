from actor_client import ActorClient
import zmq
import numpy as np
import time


def format_experience(model_hash, states, policies, rewards):
    data = {
        "model_hash": model_hash,
        "states": states,
        "policies": policies,
        "rewards": rewards,
    }
    return data


class Actor:
    def __init__(self, config):
        self.config = config
        self.client = ActorClient(
            config["server"]["ip"],
            push_port=config["actor"]["ports"]["PUSH"],
            sub_port=config["actor"]["ports"]["SUB"],
        )
        self.model_hash = None
        self.model_weights = None

    def start(self):
        while True:
            model_update = self.client.get_model_update()
            self.model_hash = model_update["model_hash"]
            self.model_weights = model_update["weights"]
            print(self.model_hash)
            states = np.random.rand(10, 9).tolist()
            policies = np.random.rand(10, 9).tolist()
            rewards = np.random.rand(10, 1).tolist()
            experience = format_experience(self.model_hash, states, policies, rewards)
            self.client.send_experience(experience)
