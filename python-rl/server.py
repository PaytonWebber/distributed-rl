import zmq
import time
import numpy as np
import threading
import hashlib

# Constants
BUFFER_SIZE = 4  # Number of experiences before sending to the learner
experience_buffer = []  # Stores experiences
model_weights = None  # Stores current model weights
model_hash = None  # Stores model weight hash


def generate_hash(data):
    """Generate a hash from a NumPy array to track weight versions."""
    return hashlib.md5(data.tobytes()).hexdigest()


def handle_learner_weights(learner_pull_socket, learner_push_socket, pub_socket):
    """Handles receiving initial & updated weights from learner."""
    global model_weights, model_hash

    # Step 1: Receive Initial Weights from Learner
    model_weights = np.frombuffer(learner_pull_socket.recv(), dtype=np.float64)
    model_hash = generate_hash(model_weights)
    print(f"[Learner] Received Initial Weights: {model_weights}, Hash: {model_hash}")

    # Step 2: Publish Initial Weights to Actors
    pub_socket.send(model_weights.tobytes() + model_hash.encode("utf-8"))
    print("[Server] Published Initial Weights to Actors.")

    while True:
        # Step 5: Receive New Weights from Learner
        new_weights = np.frombuffer(learner_pull_socket.recv(), dtype=np.float64)
        model_hash = generate_hash(new_weights)
        print(f"[Learner] Received Updated Weights: {new_weights}, Hash: {model_hash}")

        # Step 6: Publish New Weights to Actors
        model_weights = new_weights
        pub_socket.send(model_weights.tobytes() + model_hash.encode("utf-8"))
        print("[Server] Published Updated Weights to Actors.")


def handle_experience(actor_socket, learner_push_socket):
    """Handles experience collection from actors and sends mini-batches to learner."""
    global experience_buffer

    while True:
        # Step 3: Receive Experience with Hash
        message = actor_socket.recv()
        received_hash = message[-32:].decode("utf-8")  # Extract last 32 bytes as hash
        experience = np.frombuffer(message[:-32], dtype=np.float64)  # Extract data

        print(f"[Actor] Received Experience: {experience}, Hash: {received_hash}")

        if received_hash != model_hash:
            print("[Warning] Mismatched Model Hash! Experience ignored.")
            continue  # Ignore outdated experiences

        experience_buffer.append(experience)

        # Step 4: Once buffer is full, send mini-batch to learner
        if len(experience_buffer) >= BUFFER_SIZE:
            mini_batch = np.stack(experience_buffer)  # Convert list to NumPy array
            experience_buffer.clear()  # Clear buffer after sending
            print(f"[Server] Sending Mini-Batch to Learner:\n{mini_batch}")

            learner_push_socket.send(mini_batch.tobytes())  # **PUSH mini-batch to learner**


def handle_weight_publish(pub_socket):
    """Handles publishing weights to actors."""
    while model_weights is None:
        print("[Server] Waiting for Initial Weights from Learner...")
        time.sleep(1)

    while True:
        time.sleep(5)  # Publish periodically
        pub_socket.send(model_weights.tobytes() + model_hash.encode("utf-8"))
        print("[Server] Periodically Published Weights to Actors.")


def main():
    context = zmq.Context()

    # Socket for receiving experiences from actors
    actor_socket = context.socket(zmq.PULL)
    actor_socket.bind("tcp://0.0.0.0:5556")

    # **Change to PULL to receive initial weights first**
    learner_pull_socket = context.socket(zmq.PULL)
    learner_pull_socket.bind("tcp://0.0.0.0:5555")

    # **Use PUSH to send mini-batches to learner**
    learner_push_socket = context.socket(zmq.PUSH)
    learner_push_socket.bind("tcp://0.0.0.0:5558")

    # Socket for publishing new weights to actors
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://0.0.0.0:5557")

    # Start handlers in separate threads
    threading.Thread(target=handle_learner_weights, args=(learner_pull_socket, learner_push_socket, pub_socket), daemon=True).start()
    threading.Thread(target=handle_experience, args=(actor_socket, learner_push_socket), daemon=True).start()
    threading.Thread(target=handle_weight_publish, args=(pub_socket,), daemon=True).start()

    print("[Server] Running...")
    while True:
        time.sleep(1)  # Keep the main thread alive


if __name__ == "__main__":
    main()
