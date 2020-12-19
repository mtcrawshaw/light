""" Definition of Triangle object used as a piece of a mosaic. """

from typing import List, Tuple

from light.mosaic.shape import Shape


class Triangle(Shape):
    """ A piece of a mosaic with a triangle shape. """

    def __init__(
        self, vertices: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ) -> None:
        """ Init function for Triangle. """

        super().__init__()
        self.vertices = vertices
        self.num_vertices = 3

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        (ax, ay), (bx, by), (cx, cy) = self.vertices
        px, py = pos
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        beta = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
        gamma = 1.0 - alpha - beta
        return alpha >= 0 and beta >= 0 and gamma >= 0

    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns coordinates of top left and bottom right of bounding box around `self`.
        """

        left = min(self.vertices[i][0] for i in range(self.num_vertices))
        top = min(self.vertices[i][1] for i in range(self.num_vertices))
        right = max(self.vertices[i][0] for i in range(self.num_vertices))
        bottom = max(self.vertices[i][1] for i in range(self.num_vertices))
        return ((left, top), (right, bottom))

    def area(self) -> float:
        """ Returns the area of `self`. """

        (ax, ay), (bx, by), (cx, cy) = self.vertices
        return abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0
