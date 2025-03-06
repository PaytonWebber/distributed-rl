from actor_client import ActorClient
import zmq

if __name__ == "__main__":
    context = zmq.Context()
    client = ActorClient(context, "localhost")
    for _ in range (5):
        client.send_experience({
            "state": [0, 1, 0],
            "policy": [x for x in range(100000)],
            "reward": 1.0,
        })
