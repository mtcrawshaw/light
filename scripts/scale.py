import os
import argparse
import json
import glob

from PIL import Image


def main(args):

    # Get list of images.
    image_paths = glob.glob(os.path.join(args.data_dir, "*"))

    # Scale and save each new image.
    if args.data_dir[-1] == "/":
        args.data_dir = args.data_dir[:-1]
    parent_dir = os.path.dirname(args.data_dir)
    dir_name = os.path.basename(args.data_dir)
    scale_data_dir = os.path.join(parent_dir, dir_name + "_scaled")
    os.makedirs(scale_data_dir)
    for image_path in image_paths:

        # Load, and scale image.
        image = Image.open(image_path)
        new_size = (
            int(image.width * args.scale_factor),
            int(image.height * args.scale_factor),
        )
        image = image.resize(new_size)

        # Save scaled image.
        image_name = os.path.basename(image_path)
        scale_image_path = os.path.join(scale_data_dir, image_name)
        image.save(scale_image_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory holding images.")
    parser.add_argument("scale_factor", type=float, help="Factor to scale images by.")
    args = parser.parse_args()

    main(args)
