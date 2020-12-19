""" Definition of Triangle object used as a piece of a mosaic. """

import random
from typing import List, Tuple, Dict, Any

from light.mosaic import Shape


class Triangle(Shape):
    """ A piece of a mosaic with a triangle shape. """

    def __init__(
        self,
        vertices: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        **kwargs: Dict[str, Any]
    ) -> None:
        """ Init function for Triangle. """

        super().__init__(**kwargs)
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

    @classmethod
    def sample_inside(cls, shape: Shape) -> "Triangle":
        """
        Return a randomly sampled instance of `Triangle` that lies inside `shape` and
        doesn't overlap any children of `shape`.
        """

        valid_positions = shape.unique_inside_positions()

        valid = False
        while not valid:
            vertices = tuple(random.sample(valid_positions, 3))
            (ax, ay), (bx, by), (cx, cy) = vertices
            if (by - cy) * (ax - cx) + (cx - bx) * (ay - cy) == 0:
                continue

            tri = cls(
                vertices,
                max_child_area=shape.max_child_area,
                num_samples=shape.num_samples,
            )
            valid = all(pos in valid_positions for pos in tri.inside_positions())

        return tri
