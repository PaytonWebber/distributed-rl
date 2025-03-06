import zmq
import zmq.asyncio
import asyncio
import numpy as np
import hashlib

BUFFER_SIZE = 1000
experience_buffer = []
model_weights = None
model_hash = None

ctx = zmq.asyncio.Context()

# Sockets
actor_socket = ctx.socket(zmq.PULL)
actor_socket.bind("tcp://0.0.0.0:5556")

learner_push_socket = ctx.socket(zmq.PUSH)
learner_push_socket.bind("tcp://0.0.0.0:5558")

learner_socket = ctx.socket(zmq.ROUTER)
learner_socket.bind("tcp://0.0.0.0:5555")

pub_socket = ctx.socket(zmq.PUB)
pub_socket.bind("tcp://0.0.0.0:5557")


def generate_hash(data):
    """Generate a hash from a NumPy array to track weight versions."""
    return hashlib.md5(data.tobytes()).hexdigest().encode("utf-8")


async def handle_learner_weights():
    """Handles receiving initial & updated weights from learner."""
    global model_weights, model_hash

    while True:
        frames = await learner_socket.recv_multipart()
        weights_bytes = frames[-1]  # Extract weights
        model_weights = np.frombuffer(weights_bytes, dtype=np.float64)
        model_hash = generate_hash(model_weights)
        # print(f"[Learner] Received Weights: {model_weights}, Hash: {model_hash.decode()}")

        # Publish new weights
        await pub_socket.send(model_weights.tobytes() + model_hash)


async def handle_experience():
    """Handles experience collection from actors and sends mini-batches to learner."""
    global experience_buffer

    while True:
        message = await actor_socket.recv()
        experience_bytes, received_hash = message[:-32], message[-32:]

        if received_hash != model_hash:
            print("[Server] Ignored Outdated Experience.")
            continue  # Ignore outdated experiences

        experience = np.frombuffer(experience_bytes, dtype=np.float64)
        experience_buffer.append(experience)

        if len(experience_buffer) >= BUFFER_SIZE:
            mini_batch = np.stack(experience_buffer)
            experience_buffer.clear()
            await send_mini_batch(mini_batch)


async def send_mini_batch(mini_batch):
    """Send mini-batch asynchronously to learner."""
    # print(f"[Server] Sending Mini-Batch to Learner:\n{mini_batch}")
    await learner_push_socket.send(mini_batch.tobytes())


async def handle_weight_publish():
    """Continuously publishes weights even if they don't change."""
    global model_weights
    while model_weights is None:
        await asyncio.sleep(0.1)  # Wait for initial weights

    while True:
        await asyncio.sleep(0.05)  # Publish every 50ms
        await pub_socket.send(model_weights.tobytes() + model_hash)
        # print("[Server] Published Weights to Actors.")


async def main():
    """Start all tasks asynchronously."""
    await asyncio.gather(
        handle_learner_weights(),
        handle_experience(),
        handle_weight_publish(),
    )


if __name__ == "__main__":
    asyncio.run(main())
