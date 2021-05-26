import os
import argparse
import json
import glob


TRIALS = 3
NUM_SPLITS_VALS = [1500]
UNIFORMITY_VALS = [0.3, 0.4]
NUM_WORKERS = 8


def main(args):

    # Load in base config.
    with open(args.base_config, "r") as base_config_file:
        base_config = json.load(base_config_file)
    save_name = base_config["save_name"]

    # Get list of images.
    image_paths = glob.glob(os.path.join(args.data_dir, "*"))
    num_images = len(image_paths)

    # Output initial information.
    total_trials = TRIALS * len(UNIFORMITY_VALS) * len(NUM_SPLITS_VALS) * num_images
    print("Number of input images: %d" % num_images)
    print("Trials per image: %d" % TRIALS)
    print("Values of num_splits: %s" % str(NUM_SPLITS_VALS))
    print("Values of uniformity: %s" % str(UNIFORMITY_VALS))
    print("Total mosaics to create: %d" % total_trials)

    # Generate configs.
    config_paths = []
    for image_path in image_paths:
        for num_splits in NUM_SPLITS_VALS:
            for uniformity in UNIFORMITY_VALS:
                for trial in range(TRIALS):

                    # Set config for current trial.
                    image_name = os.path.basename(image_path)
                    config = dict(base_config)
                    config["image_path"] = image_path
                    config["num_splits"] = num_splits
                    config["uniformity"] = uniformity
                    config["seed"] = trial
                    config["save_name"] = "%s_%s_%d_%.2f_%d" % (
                        save_name,
                        image_name,
                        num_splits,
                        uniformity,
                        trial,
                    )

                    # Dump config.
                    config_name = "%s.json" % config["save_name"]
                    config_path = os.path.join(
                        os.path.dirname(args.base_config), config_name
                    )
                    with open(config_path, "w") as config_file:
                        json.dump(config, config_file)
                    config_paths.append(config_path)

    # Run experiments.
    cmd = "parallel -j %d :::" % NUM_WORKERS
    for i, config_path in enumerate(config_paths):
        cmd += ' "python3 main.py mosaic %s"' % config_path
    os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory holding images.")
    parser.add_argument("base_config", type=str, help="Path to base config.")
    args = parser.parse_args()

    main(args)
