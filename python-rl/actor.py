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
        self.model_hash = "adfc134xkjlk134ASD"
        self.model = None

    def start(self):
        while True:
            time.sleep(5)
            states = np.random.rand(10, 9).tolist()
            policies = np.random.rand(10, 9).tolist()
            rewards = np.random.rand(10, 1).tolist()
            experience = format_experience(self.model_hash, states, policies, rewards)
            self.client.send_experience(experience)
