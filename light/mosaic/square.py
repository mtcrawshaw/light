""" Definition of Square object used as a piece of a mosaic. """

from typing import List, Tuple

from light.mosaic.shape import Shape


class Square(Shape):
    """ A piece of a mosaic with a square shape. """

    def __init__(
        self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]
    ) -> None:
        """ Init function for Square. """

        super().__init__()
        self.top_left = top_left
        self.bottom_right = bottom_right

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        ((left, top), (right, bottom)) = (self.top_left, self.bottom_right)
        return left <= pos[0] <= right and top <= pos[0] <= bottom

    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns coordinates of top left and bottom right of bounding box around `self`.
        """

        return (self.top_left, self.bottom_right)

    def area(self) -> float:
        """ Returns the area of `self`. """

        ((left, top), (right, bottom)) = (self.top_left, self.bottom_right)
        return (right - left) * (bottom - top)
