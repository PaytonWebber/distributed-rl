import yaml
import argparse
from actor import Actor
from learner import Learner


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["actor", "learner"],
        required=True,
        help="Specify whether this instance is an actor or learner.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.type == "actor":
        actor = Actor(config)
        actor.start()
    else:
        learner = Learner(config)
        learner.start()
