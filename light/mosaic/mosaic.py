""" Create a gradient-colored mosaic from an image. """

import os
import random
import pickle
import json
from typing import Dict, Any

from PIL import Image

from light.mosaic import Canvas, Circle, Triangle, Square


def create_mosaic(config: Dict[str, Any]) -> None:
    """ Create a gradient-colored mosaic from an image. """

    # Check uniqueness of save name.
    save_dir = None
    if config["save_name"] is not None:
        save_dir = os.path.join("results", config["save_name"])
        if os.path.isdir(save_dir):
            print(
                "Results directory '%s' already exists! Save names must be unique."
                % save_dir
            )
            exit()
        else:
            os.makedirs(save_dir)

    # Set random seed.
    random.seed(config["seed"])

    # Read in image, instantiate canvas, partition and color canvas.
    image = Image.open(config["image_path"])
    canvas = Canvas(
        image=image,
        num_splits=config["num_splits"],
        max_child_area=config["max_child_area"],
        num_samples=config["num_samples"],
    )
    canvas.partition()
    mosaic = canvas.color()

    # Save out results.
    if save_dir is not None:

        # Save config.
        config_path = os.path.join(save_dir, "%s_config.json" % config["save_name"])
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Save original image.
        dot_pos = config["image_path"].rfind(".")
        image_ext = config["image_path"][dot_pos + 1 :]
        image_path = os.path.join(
            save_dir, "%s_image.%s" % (config["save_name"], image_ext)
        )
        image.save(image_path)

        # Save canvas.
        canvas_path = os.path.join(save_dir, "%s_canvas.pkl" % config["save_name"])
        with open(canvas_path, "wb") as canvas_file:
            pickle.dump(canvas, canvas_file)

        # Save mosaic.
        mosaic_path = os.path.join(
            save_dir, "%s_mosaic.%s" % (config["save_name"], image_ext)
        )
        mosaic.save(mosaic_path)
