""" Definition of Canvas, the main object used to create a mosaic. """

import random

from PIL import Image

from light.mosaic import Circle, Triangle, Square


PIECE_TYPES = [Circle, Triangle, Square]



class Canvas:
    """ Main object used to create a mosaic. """

    def __init__(self, image: Image, num_splits: int = 100, max_child_area: float = 0.7, num_samples: int = 100) -> None:
        """ Init function for Canvas. """

        self.image = image
        self.num_splits = num_splits
        self.max_child_area = max_child_area
        self.num_samples = num_samples
        self.area = Square((0, 0), (image.width, image.height))

    def partition(self) -> None:
        """ Partition canvas into `self.num_splits` pieces. """

        for _ in range(self.num_splits):
            all_pieces = self.area.descendants()
            split_piece = random.choice(all_pieces)
            piece_type = random.choice(PIECE_TYPES)
            split_piece.partition(piece_type)

    def color(self) -> Image:
        """ Color each piece of canvas with linear regression over original colors. """

        return self.image
