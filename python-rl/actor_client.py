import zmq


class ActorClient:
    def __init__(self, server_ip, push_port=5555, sub_port=5556):
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{server_ip}:{push_port}")

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_ip}:{sub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def send_experience(self, experience):
        self.push_socket.send_json(experience)

    def receieve_weights(self):
        return self.sub_socket.recv()
