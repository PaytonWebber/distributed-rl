import zmq
 
class LearnerClient:
    def __init__(self, context, server_ip, req_port=5557):
        self.context = context
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{server_ip}:{req_port}")

    def send_request(self):
        self.req_socket.send(b'test')

    def get_reply(self):
        return self.req_socket.recv()

if __name__ == "__main__":
    context = zmq.Context()
    client = LearnerClient(context, "localhost")
    client.send_request()
    print(client.get_reply())

