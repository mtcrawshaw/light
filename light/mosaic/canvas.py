""" Definition of Canvas, the main object used to create a mosaic. """

import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression

from light.mosaic import Triangle


class Canvas:
    """ Main object used to create a mosaic. """

    def __init__(
        self, image: Image, num_splits: int = 100, uniformity: float = 1.0,
    ) -> None:
        """ Init function for Canvas. """

        self.image = image
        self.num_splits = num_splits
        self.uniformity = uniformity
        self.pieces = []

    def partition(self) -> None:
        """ Partition canvas into `self.num_splits` pieces. """

        alpha_min = 0.5 - (1.0 - self.uniformity) / 2.0
        alpha_max = 0.5 + (1.0 - self.uniformity) / 2.0

        # Sample initial triangle with one vertex in the image corner. We do this so
        # that the image area is partitioned into only triangles after the initial
        # triangle is sampled.
        corners = [
            (0, 0),
            (self.image.width - 1, 0),
            (self.image.width - 1, self.image.height - 1),
            (0, self.image.height - 1),
        ]
        common = random.choice(range(len(corners)))
        v0 = corners[common]
        c1 = corners[(common + 1) % len(corners)]
        c2 = corners[(common + 2) % len(corners)]
        c3 = corners[(common + 3) % len(corners)]

        # Sample points along edges opposite from common corner.
        alpha1 = random.uniform(alpha_min, alpha_max)
        v1 = (
            round(c1[0] * alpha1 + c2[0] * (1 - alpha1)),
            round(c1[1] * alpha1 + c2[1] * (1 - alpha1)),
        )
        alpha2 = random.uniform(alpha_min, alpha_max)
        v2 = (
            round(c2[0] * alpha2 + c3[0] * (1 - alpha2)),
            round(c2[1] * alpha2 + c3[1] * (1 - alpha2)),
        )

        # Create triangles from image corners and sampled vertices.
        self.pieces = [
            Triangle(vertices=(v0, v1, v2)),
            Triangle(vertices=(v0, c1, v1)),
            Triangle(vertices=(v1, c2, v2)),
            Triangle(vertices=(v0, v2, c3)),
        ]

        # Partition the existing triangles iteratively.
        for i in range(self.num_splits):

            # Sample a piece and split it.
            weights = [piece.area() for piece in self.pieces]
            split_piece = random.choices(self.pieces, weights=weights)[0]
            new_pieces = split_piece.partition(self.uniformity)

            # Add resulting pieces to total list and remove sampled piece.
            self.pieces.remove(split_piece)
            self.pieces += new_pieces

    def color(self) -> Image:
        """ Color each piece of canvas with linear regression over original colors. """

        colored = Image.new(self.image.mode, self.image.size)

        valid_pos = lambda im, p: (0 <= p[0] < im.width) and (0 <= p[1] < im.height)
        for piece in self.pieces:

            # Collect all positions in piece and their corresponding colors.
            piece_positions = [
                pos for pos in piece.inside_positions() if valid_pos(self.image, pos)
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
