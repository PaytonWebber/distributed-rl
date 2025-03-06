import zmq

class ActorClient:
    def __init__(self, context, server_ip, push_port=5555, sub_port=5556):
        self.context = context

        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{server_ip}:{push_port}")

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_ip}:{sub_port}")

    def send_experience(self, experience):
        self.push_socket.send_json(experience)

    def receieve_weights(self):
        return self.sub_socket.recv_json()

if __name__ == "__main__":
    context = zmq.Context()
    client = ActorClient(context, "localhost")
    client.send_experience({
        "state": [0, 1, 0],
        "policy": [x for x in range(100000)],
        "reward": 1.0,
    })
