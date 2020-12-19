""" Definition of Circle object used as a piece of a mosaic. """

from math import sqrt, pi
from typing import List, Tuple

from light.mosaic.shape import Shape


class Circle(Shape):
    """ A piece of a mosaic with a circle shape. """

    def __init__(self, center: Tuple[int, int], radius: int) -> None:
        """ Init function for Circle. """

        super().__init__()
        self.center = center
        self.radius = radius

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        dist = sqrt((self.center[0] - pos[0]) ** 2 + (self.center[1] - pos[1]) ** 2)
        return dist <= self.radius

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
