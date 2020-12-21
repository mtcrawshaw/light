""" Definition of Canvas, the main object used to create a mosaic. """

import random

from PIL import Image

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
        for piece in pieces:
            boundary = piece.boundary_positions()
            for pos in boundary:
                if 0 <= pos[0] < colored.width and 0 <= pos[1] < colored.height:
                    colored.putpixel(pos, (255, 255, 255))

        return colored
