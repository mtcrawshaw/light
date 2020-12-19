""" Definition of Shape object used as a piece of a mosaic. """

from typing import List, Tuple


class Shape:
    """
    A piece of a mosaic. Note that we follow Pillow's convention of placing the origin
    at the top left of the image, so that values of the y-axis increase as you move
    down. This is an abstract class and shouldn't be instantiated.
    """

    def __init__(self) -> None:
        """ Init function for Shape. """

        self.children: List["Shape"] = []

    def partition(self) -> None:
        """ Add a single non-overlapping Shape to children. """

        pass

    def descendants(self) -> List["Shape"]:
        """ Returns a list of all descendants of `self`, including `self`. """

        d: List["Shape"] = []
        for child in self.children:
            d.append(child.descendants())
        d.append(self)

        return d

    def insidePositions(self) -> List[Tuple[int, int]]:
        """ Returns a list of all positions inside `self`. """

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.isInside((x, y)):
                    positions.append((x, y))

        return positions

    def isInside(self, pos: Tuple[int, int]) -> bool:
        """
        Returns True if `pos` is inside `self`, False otherwise. This function should be
        overridden in subclasses.
        """

        raise NotImplementedError

    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns coordinates of top left and bottom right of bounding box around `self`.
        This function should be overridden in subclasses.
        """

        raise NotImplementedError

    def area(self) -> float:
        """
        Returns the area of `self`. This function should be overridden in subclasses.
        """

        raise NotImplementedError
