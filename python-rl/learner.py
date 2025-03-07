from learner_client import LearnerClient
import zmq


class Learner:
    def __init__(self, config):
        self.config = config
        self.client = LearnerClient(
            config["server"]["ip"], config["learner"]["ports"]["REQ"]
        )

    def start(self):
        while True:
            self.client.request_mini_batch()
            reply = self.client.get_reply()
            print(reply)
            if reply != b"NOT ENOUGH EXPERIENCES":
                break
