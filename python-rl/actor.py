import zmq
import zmq.asyncio
import asyncio
import numpy as np

ctx = zmq.asyncio.Context()

# SUB socket to receive model weights
sub_socket = ctx.socket(zmq.SUB)
sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
sub_socket.connect("tcp://localhost:5557")

# PUSH socket to send experiences
push_socket = ctx.socket(zmq.PUSH)
push_socket.connect("tcp://localhost:5556")

# Global variables for model weights and hash
model_weights = None
model_hash = None


def extract_weights_and_hash(message):
    """Extract weights and hash from the received model weights message."""
    model_weights = np.frombuffer(message[:-32], dtype=np.float64)
    model_hash = message[-32:]
    return model_weights, model_hash


async def receive_weights():
    """Receives weights and stores the latest hash."""
    global model_weights, model_hash

    # Wait for initial weights
    message = await sub_socket.recv()
    model_weights, model_hash = extract_weights_and_hash(message)
    # print(f"[Actor] Received Initial Weights: {model_weights}, Hash: {model_hash.decode(errors='ignore')}")

    while True:
        message = await sub_socket.recv()
        model_weights, model_hash = extract_weights_and_hash(message)
        # print(f"[Actor] Updated Weights: {model_weights}, Hash: {model_hash.decode(errors='ignore')}")


async def generate_experience():
    """Continuously generates experience while having weights."""
    global model_weights, model_hash  # 🔥 FIX: Ensure global access

    while model_weights is None:
        await asyncio.sleep(0.1)  # Wait for first weights

    while True:
        # Generate experience at a high rate
        experience = np.random.rand(4)
        experience_with_hash = experience.tobytes() + model_hash  # Attach hash
        await push_socket.send(experience_with_hash)
        await asyncio.sleep(0.01)  # Generate experiences every 10ms


async def main():
    """Handles actor interactions asynchronously."""
    await asyncio.gather(
        receive_weights(),
        generate_experience(),
    )


if __name__ == "__main__":
    asyncio.run(main())
