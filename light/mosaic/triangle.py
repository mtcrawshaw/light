""" Definition of Triangle object used as a piece of a mosaic. """

from math import sqrt
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

    def __str__(self) -> str:
        """ Returns a string representation of `self`. """

        return "Vertices: %s" % str(self.vertices)

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        (ax, ay), (bx, by), (cx, cy) = self.vertices
        px, py = pos
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        beta = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
        gamma = 1.0 - alpha - beta
        return alpha >= 0 and beta >= 0 and gamma >= 0

    def is_boundary(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is on the boundary of `self`, False otherwise. """

        def is_between(
            v1: Tuple[int, int], v2: Tuple[int, int], p: Tuple[int, int]
        ) -> bool:
            """ Returns True if `p` is between `v1` and `v2`, and False otherwise. """

            A = (v2[1] - v1[1]) / (v2[0] - v1[0])
            B = -1
            C = -v1[0] * (v2[1] - v1[1]) / (v2[0] - v1[0]) + v1[1]

            proj_x = (B * (B * p[0] - A * p[1]) - A * C) / (A ** 2 + B ** 2)
            proj_y = (A * (-B * p[0] + A * p[1]) - B * C) / (A ** 2 + B ** 2)
            proj = (proj_x, proj_y)

            proj_x_between = (v1[0] <= proj[0] <= v2[0]) or (v2[0] <= proj[0] <= v1[0])
            proj_y_between = (v1[1] <= proj[1] <= v2[1]) or (v2[1] <= proj[1] <= v1[1])
            if not (proj_x_between and proj_y_between):
                return False

            dist = sqrt((proj[0] - p[0]) ** 2 + (proj[1] - p[1]) ** 2)
            return dist <= self.boundary_width

        side0 = is_between(self.vertices[0], self.vertices[1], pos)
        side1 = is_between(self.vertices[0], self.vertices[2], pos)
        side2 = is_between(self.vertices[1], self.vertices[2], pos)
        return side0 or side1 or side2

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
    def sample_inside(
        cls, shape: Shape, valid_positions: List[Tuple[int, int]]
    ) -> "Triangle":
        """
        Return a randomly sampled instance of `Triangle` that lies inside `shape` and
        doesn't overlap any children of `shape`.
        """

        valid = False
        tries = 0

        while not valid:
            tries += 1
            print("    Sample attempt %d" % tries)

            vertices = tuple(random.sample(valid_positions, 3))
            (ax, ay), (bx, by), (cx, cy) = vertices
            if (by - cy) * (ax - cx) + (cx - bx) * (ay - cy) == 0:
                continue

            tri = cls(
                vertices,
                max_child_area=shape.max_child_area,
                num_samples=shape.num_samples,
                num_workers=shape.num_workers,
                boundary_width=shape.boundary_width,
            )
            valid = all(pos in valid_positions for pos in tri.inside_positions())

        return tri
