""" Definition of Square object used as a piece of a mosaic. """

import random
from math import sqrt
from typing import List, Tuple, Dict, Any

from light.mosaic import Shape


class Square(Shape):
    """ A piece of a mosaic with a square shape. """

    def __init__(
        self,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        **kwargs: Dict[str, Any]
    ) -> None:
        """ Init function for Square. """

        super().__init__()
        self.top_left = top_left
        self.bottom_right = bottom_right

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        ((left, top), (right, bottom)) = (self.top_left, self.bottom_right)
        return left <= pos[0] <= right and top <= pos[0] <= bottom

    def is_boundary(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is on the boundary of `self`, False otherwise. """

        left = abs(pos[0] - self.top_left[0]) <= self.boundary_width
        top = abs(pos[1] - self.top_left[1]) <= self.boundary_width
        right = abs(pos[0] - self.bottom_right[0]) <= self.boundary_width
        bottom = abs(pos[1] - self.bottom_right[1]) <= self.boundary_width
        return (left or top or right or bottom)

    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns coordinates of top left and bottom right of bounding box around `self`.
        """

        return (self.top_left, self.bottom_right)

    def area(self) -> float:
        """ Returns the area of `self`. """

        ((left, top), (right, bottom)) = (self.top_left, self.bottom_right)
        return (right - left) * (bottom - top)

    @classmethod
    def sample_inside(cls, shape: Shape) -> "Square":
        """
        Return a randomly sampled instance of `Square` that lies inside `shape` and
        doesn't overlap any children of `shape`.
        """

        valid_positions = shape.unique_inside_positions()

        valid = False
        while not valid:
            center = random.choice(valid_positions)
            dist = lambda a, b: sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            max_radius = max(dist(center, pos) for pos in valid_positions)
            radius = random.choice(range(1, int(max_radius)))
            top_left = (center[0] - radius, center[1] - radius)
            bottom_right = (center[1] + radius, center[1] + radius)

            square = cls(
                top_left,
                bottom_right,
                max_child_area=shape.max_child_area,
                num_samples=shape.num_samples,
                boundary_width=shape.boundary_width,
            )
            valid = all(pos in valid_positions for pos in square.inside_positions())

        return square
