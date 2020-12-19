""" Main function for light. """

import argparse
import json

from light.gan import train_gan
from light.mosaic import create_mosaic


def main(args: argparse.Namespace):
    """ Main function for light. """

    # Read in config file.
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # Call appropriate command.
    if args.command == "gan":
        train_gan(config)
    elif args.command == "mosaic":
        create_mosaic(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="What to run. Options: ['gan', 'mosaic'].")
    parser.add_argument("config_path", type=str, help="Path to config file.")
    args = parser.parse_args()
    main(args)
