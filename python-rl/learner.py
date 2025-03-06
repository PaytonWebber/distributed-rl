import zmq
import numpy as np

context = zmq.Context()

# **PUSH socket to send updated weights to server**
learner_push_socket = context.socket(zmq.PUSH)
learner_push_socket.connect("tcp://localhost:5555")

# **PULL socket to receive mini-batches from server**
learner_pull_socket = context.socket(zmq.PULL)
learner_pull_socket.connect("tcp://localhost:5558")

# Step 1: Send Initial Model Weights to Server (No Acknowledgment Expected)
initial_weights = np.random.rand(10)  # Example: 10 model weights
learner_push_socket.send(initial_weights.tobytes())  # Send initial weights
print(f"[Learner] Sent Initial Weights: {initial_weights}")

# Step 2: **Wait for the first mini-batch from server**
mini_batch_bytes = learner_pull_socket.recv()  # **PULL socket receives data**
mini_batch = np.frombuffer(mini_batch_bytes, dtype=np.float64).reshape(-1, 4)  # Deserialize
print(f"[Learner] Received First Mini-Batch:\n{mini_batch}")

while True:
    # Step 3: Receive Mini-Batch from Server (Binary Data)
    mini_batch_bytes = learner_pull_socket.recv()  # **Use PULL socket**
    mini_batch = np.frombuffer(mini_batch_bytes, dtype=np.float64).reshape(-1, 4)  # Deserialize
    print(f"[Learner] Received Mini-Batch:\n{mini_batch}")

    # Process mini-batch (Simulated learning step)
    new_weights = np.random.rand(10)  # Generate new weights

    # Step 4: Send Updated Weights to Server
    learner_push_socket.send(new_weights.tobytes())  # Serialize and send
    print(f"[Learner] Sent Updated Weights: {new_weights}")
