import zmq
import zmq.asyncio
import asyncio
import numpy as np
import time  # 🔥 Import time to track batch arrival times

ctx = zmq.asyncio.Context()

# DEALER Socket for Bi-Directional Communication
learner_socket = ctx.socket(zmq.DEALER)
learner_socket.connect("tcp://localhost:5555")

# PULL Socket to Receive Mini-Batches
mini_batch_socket = ctx.socket(zmq.PULL)
mini_batch_socket.connect("tcp://localhost:5558")


async def main():
    """Handles learner interactions asynchronously."""

    # Step 1: Send Initial Model Weights
    initial_weights = np.random.rand(10)
    await learner_socket.send_multipart([b"identity", initial_weights.tobytes()])
    print(f"[Learner] Sent Initial Weights: {initial_weights}")

    last_time = time.time()  # 🔥 Track the time of the last received batch

    while True:
        mini_batch_bytes = await mini_batch_socket.recv()
        current_time = time.time()  # 🔥 Get the time of the new batch
        elapsed_time = current_time - last_time  # 🔥 Calculate time since last batch

        mini_batch = np.frombuffer(mini_batch_bytes, dtype=np.float64).reshape(-1, 4)
        # print(f"[Learner] Received Mini-Batch:\n{mini_batch}")
        print(f"⏳ Time since last batch: {elapsed_time:.6f} seconds")  # 🔥 Print time difference

        # wait 5 seconds
        await asyncio.sleep(5)        # print("waited 5 seconds")

        # Simulated learning step
        new_weights = np.random.rand(10)

        last_time = time.time()  # 🔥 Update the time of the last received batch
        # Step 3: Send Updated Weights to Server
        await learner_socket.send_multipart([b"identity", new_weights.tobytes()])
        # print(f"[Learner] Sent Updated Weights")



if __name__ == "__main__":
    asyncio.run(main())
