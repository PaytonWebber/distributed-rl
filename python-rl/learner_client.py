import zmq


class LearnerClient:
    def __init__(self, server_ip, req_port=5557):
        self.context = zmq.Context()
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{server_ip}:{req_port}")

    def request_mini_batch(self):
        self.req_socket.send(b"test")

    def get_reply(self):
        return self.req_socket.recv()
