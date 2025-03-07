from learner_client import LearnerClient
import zmq
import numpy as np
import time
import random
import string


class Learner:
    def __init__(self, config):
        self.config = config
        self.model_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.model_weights = np.random.rand(4).tolist()
        self.client = LearnerClient(
            config["server"]["ip"], config["learner"]["ports"]["REQ"]
        )

    def start(self):
        _ = self.client.send_model_update(self.model_hash, self.model_weights)
        while True:
            mini_batch_reply = self.client.request_mini_batch()
            if mini_batch_reply != "NOT ENOUGH EXPERIENCES":
                self.train(mini_batch_reply)
                _ = self.client.send_model_update(self.model_hash, self.model_weights)
            time.sleep(30)

    def train(self, reply):
        print("Starting to train ...")
        while reply != "NOT ENOUGH EXPERIENCES":
            reply = self.client.request_mini_batch()
        self.model_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.model_weights = np.random.rand(4).tolist()
        print("Training over ...")

