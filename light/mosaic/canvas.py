""" Definition of Canvas, the main object used to create a mosaic. """

import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression

from light.mosaic import Circle, Triangle, Square


PIECE_TYPES = [Circle, Triangle, Square]


class Canvas:
    """ Main object used to create a mosaic. """

    def __init__(
        self,
        image: Image,
        num_splits: int = 100,
        max_child_area: float = 0.7,
        num_samples: int = 100,
        boundary_width: int = 3,
    ) -> None:
        """ Init function for Canvas. """

        self.image = image
        self.num_splits = num_splits
        self.max_child_area = max_child_area
        self.num_samples = num_samples
        self.boundary_width = boundary_width
        self.area = Square(
            (0, 0),
            (image.width, image.height),
            max_child_area=max_child_area,
            num_samples=num_samples,
            boundary_width=boundary_width,
        )

    def partition(self) -> None:
        """ Partition canvas into `self.num_splits` pieces. """

        for _ in range(self.num_splits):
            all_pieces = self.area.descendants()
            split_piece = random.choice(all_pieces)
            piece_type = random.choice(PIECE_TYPES)
            split_piece.partition(piece_type)

    def color(self) -> Image:
        """ Color each piece of canvas with linear regression over original colors. """

        colored = Image.new(self.image.mode, self.image.size)
        pieces = self.area.descendants()

        valid_pos = lambda im, p: (0 <= p[0] < im.width) and (0 <= p[1] < im.height)
        for piece in pieces:

            # Collect all positions in piece and their corresponding colors.
            piece_positions = [
                pos
                for pos in piece.unique_inside_positions()
                if valid_pos(self.image, pos)
            ]
            piece_pixels = np.array(
                [self.image.getpixel(pos) for pos in piece_positions]
            )
            piece_positions = np.array(piece_positions)

            # Perform linear regression on each channel.
            regressors = []
            for channel in range(3):
                piece_channels = piece_pixels[:, channel]
                regressor = LinearRegression().fit(piece_positions, piece_channels)
                regressors.append(regressor)

            # Set color for each position in piece as a function of regressor.
            reds = np.expand_dims(regressors[0].predict(piece_positions), 1)
            greens = np.expand_dims(regressors[1].predict(piece_positions), 1)
            blues = np.expand_dims(regressors[2].predict(piece_positions), 1)
            regressed_colors = np.concatenate([reds, greens, blues], axis=1)
            regressed_colors = np.rint(regressed_colors).astype(int)
            for i, pos in enumerate(piece_positions):
                colored.putpixel(pos, tuple(regressed_colors[i]))

        return colored
