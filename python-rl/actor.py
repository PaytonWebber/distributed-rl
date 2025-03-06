import zmq
import numpy as np

context = zmq.Context()

# SUB socket to receive model weights
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:5557")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# PUSH socket to send experiences
actor_socket = context.socket(zmq.PUSH)
actor_socket.connect("tcp://localhost:5556")

# Step 2: Wait for Initial Model Weights
model_weights = None
while model_weights is None:
    print("[Actor] Waiting for Initial Weights from Server...")
    weights_message = sub_socket.recv()
    model_weights = np.frombuffer(weights_message[:-32], dtype=np.float64)
    model_hash = weights_message[-32:].decode("utf-8")
    print(f"[Actor] Received Initial Weights: {model_weights}, Hash: {model_hash}")

while True:
    # Step 6: Receive Updated Model Weights
    weights_message = sub_socket.recv()
    model_weights = np.frombuffer(weights_message[:-32], dtype=np.float64)
    model_hash = weights_message[-32:].decode("utf-8")
    print(f"[Actor] Received Updated Weights: {model_weights}, Hash: {model_hash}")

    # Step 3: Send Experience with Hash
    experience = np.random.rand(4)
    actor_socket.send(experience.tobytes() + model_hash.encode("utf-8"))
    print(f"[Actor] Sent Experience: {experience}, Hash: {model_hash}")
