import os
import argparse
import json
import glob


def main(args):

    # Load in base config.
    with open(args.base_config, "r") as base_config_file:
        base_config = json.load(base_config_file)
    save_name = base_config["save_name"]

    # Get list of images.
    image_paths = glob.glob(os.path.join(args.data_dir, "*"))

    # Generate configs.
    config_paths = []
    for image_path in image_paths:
        for trial in range(args.trials_per_image):

            # Set config for current trial.
            image_name = os.path.basename(image_path)
            config = dict(base_config)
            config["image_path"] = image_path
            config["seed"] = trial
            config["save_name"] = "%s_%s_%d" % (save_name, image_name, trial)

            # Dump config.
            config_name = "%s_%s_%d.json" % (save_name, image_name, trial)
            config_path = os.path.join(os.path.dirname(args.base_config), config_name)
            with open(config_path, "w") as config_file:
                json.dump(config, config_file)
            config_paths.append(config_path)

    # Run experiments.
    for config_path in config_paths:
        os.system("python3 main.py mosaic %s" % config_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory holding images.")
    parser.add_argument(
        "trials_per_image",
        type=int,
        help="Number of trials per image in data directory.",
    )
    parser.add_argument("base_config", type=str, help="Path to base config.")
    args = parser.parse_args()

    main(args)
