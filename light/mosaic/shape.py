""" Definition of Shape object used as a piece of a mosaic. """

from typing import List, Tuple


class Shape:
    """
    A piece of a mosaic. Note that we follow Pillow's convention of placing the origin
    at the top left of the image, so that values of the y-axis increase as you move
    down. This is an abstract class and shouldn't be instantiated.
    """

    def __init__(self, max_child_area: float = 0.7, num_samples: int = 100, boundary_width: int = 3) -> None:
        """ Init function for Shape. """

        self.max_child_area = max_child_area
        self.num_samples = num_samples
        self.boundary_width = boundary_width
        self.children: List["Shape"] = []

    def partition(self, shape_cls) -> None:
        """ Add a single non-overlapping Shape to children. """

        biggest_child = None
        biggest_child_area = None
        for _ in range(self.num_samples):
            candidate = shape_cls.sample_inside(self)
            area = candidate.area()
            if (
                biggest_child_area is None
                or biggest_child_area <= area <= self.max_child_area
            ):
                biggest_child = candidate
                biggest_child_area = area

        assert biggest_child is not None
        assert biggest_child_area is not None

        self.children.append(biggest_child)

    def descendants(self) -> List["Shape"]:
        """ Returns a list of all descendants of `self`, including `self`. """

        d: List["Shape"] = []
        for child in self.children:
            d += child.descendants()
        d.append(self)

        return d

    def inside_positions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all positions inside `self`. We define inside to include the
        boundaries of `self`.
        """

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.is_inside((x, y)):
                    positions.append((x, y))

        return positions

    def unique_inside_positions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all positions inside `self` which aren't inside of any
        children of `self.`.
        """

        positions: List[Tuple[int, int]] = self.inside_positions()
        child_positions: List[Tuple[int, int]] = []
        for child in self.children:
            child_positions += child.inside_positions()

        unique_positions: List[Tuple[int, int]] = []
        for position in positions:
            if position not in child_positions:
                unique_positions.append(position)

        return unique_positions

    def boundary_positions(self) -> List[Tuple[int, int]]:
        """
        Return all positions on the boundary of `self`.
        """

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.is_boundary((x, y)):
                    positions.append((x, y))

        return positions

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """
        Returns True if `pos` is inside `self`, False otherwise. This function should be
        overridden in subclasses.
        """

        raise NotImplementedError

    def is_boundary(self, pos: Tuple[int, int]) -> bool:
        """
        Returns True if `pos` is on the boundary of `self`, False otherwise. This
        function should be overridden in subclasses.
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

    @classmethod
    def sample_inside(cls, shape: "Shape") -> "Shape":
        """
        Return a randomly sampled instance of `Shape` that lies inside
        `shape.unique_inside_positions`. This function should be overridden in
        subclasses.
        """

        raise NotImplementedError
