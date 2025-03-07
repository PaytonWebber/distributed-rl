from actor_client import ActorClient
import zmq
import numpy as np


class Actor:
    def __init__(self, config):
        self.config = config
        self.client = ActorClient(
            config["server"]["ip"],
            push_port=config["actor"]["ports"]["PUSH"],
            sub_port=config["actor"]["ports"]["SUB"],
        )
        self.model = None

    def start(self):
        while True:
            experience = np.random.rand(3, 2)
            self.client.send_experience(f"experience: {experience}")
