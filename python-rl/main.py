import yaml
import argparse
from actor import Actor


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("config.yaml")
    actor = Actor(config)
    actor.start()
