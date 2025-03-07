import zmq
import json


class LearnerClient:
    def __init__(self, server_ip, req_port=5557):
        self.context = zmq.Context()
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{server_ip}:{req_port}")

    def send_model_update(self, model_hash, weights):
        model_update = {"model_hash": model_hash, "weights": weights}
        self.req_socket.send_json(model_update)
        reply = self.req_socket.recv_string()
        return reply

    def request_mini_batch(self):
        request_message = "REQUEST_MINI_BATCH"
        self.req_socket.send_string(request_message)
        reply = self.req_socket.recv_string()
        try:
            mini_batch = json.loads(reply)
            return mini_batch
        except json.JSONDecodeError:
            return reply
