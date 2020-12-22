""" Definition of Circle object used as a piece of a mosaic. """

import random
from math import sqrt, pi
from typing import List, Tuple, Dict, Any

from light.mosaic import Shape


class Circle(Shape):
    """ A piece of a mosaic with a circle shape. """

    def __init__(
        self, center: Tuple[int, int], radius: int, **kwargs: Dict[str, Any]
    ) -> None:
        """ Init function for Circle. """

        super().__init__(**kwargs)
        self.center = center
        self.radius = radius

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        dist = sqrt((self.center[0] - pos[0]) ** 2 + (self.center[1] - pos[1]) ** 2)
        return dist <= self.radius

    def is_boundary(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is on the boundary of `self`, False otherwise. """

        dist = sqrt((self.center[0] - pos[0]) ** 2 + (self.center[1] - pos[1]) ** 2)
        return abs(dist - self.radius) <= self.boundary_width

    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns coordinates of top left and bottom right of bounding box around `self`.
        """

        left = self.center[0] - self.radius
        top = self.center[1] - self.radius
        right = self.center[0] + self.radius
        bottom = self.center[1] + self.radius
        return ((left, top), (right, bottom))

    def area(self) -> float:
        """ Returns the area of `self`. """

        return pi * self.radius ** 2

    @classmethod
    def sample_inside(
        cls, shape: Shape, valid_positions: List[Tuple[int, int]]
    ) -> "Circle":
        """
        Return a randomly sampled instance of `Circle` that lies inside `shape` and
        doesn't overlap any children of `shape`.
        """

        valid = False
        while not valid:
            center = random.choice(valid_positions)
            dist = lambda a, b: sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            max_radius = max(dist(center, pos) for pos in valid_positions)
            radius = random.choice(range(1, int(max_radius)))

            circle = cls(
                center,
                radius,
                max_child_area=shape.max_child_area,
                num_samples=shape.num_samples,
                boundary_width=shape.boundary_width,
            )
            valid = all(pos in valid_positions for pos in circle.inside_positions())

        return circle
