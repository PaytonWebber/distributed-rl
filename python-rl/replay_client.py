import time
import zmq

context = zmq.Context()
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

for r in range(10):
    print(f"Sending request {r} ...")
    socket.send(b"Hello Rust Server")

    message = socket.recv()
    print("Received reply %s [ %s ]" % (r, message))
