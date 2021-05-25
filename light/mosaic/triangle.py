""" Definition of Triangle object used as a piece of a mosaic. """

import random

from typing import List, Tuple, Dict, Any


class Triangle:
    """ A piece of a mosaic with a triangle shape.  """

    def __init__(
        self, vertices: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> None:
        """ Init function for Triangle. """

        assert len(vertices) == 3
        self._vertices = vertices
        self.num_vertices = len(self._vertices)

        # This is a cache used to hold results of function calls that will never change,
        # which we use to avoid redundant computation.
        self._cache = {}

    def partition(self, uniformity) -> List["Triangle"]:
        """
        Partition `self` into four smaller Triangles, and return these four new
        Triangles in a list.
        """

        alpha_min = 0.5 - (1.0 - uniformity) / 2.0
        alpha_max = 0.5 + (1.0 - uniformity) / 2.0

        # Generate points near midpoint of each edge of `self`.
        midpoints = []
        for i in range(self.num_vertices):
            alpha = random.uniform(alpha_min, alpha_max)
            v1 = self._vertices[i]
            v2 = self._vertices[(i + 1) % self.num_vertices]

            m = (
                round(v1[0] * alpha + v2[0] * (1 - alpha)),
                round(v1[1] * alpha + v2[1] * (1 - alpha)),
            )
            midpoints.append(m)

        # Generate four triangles from original vertices and new midpoints.
        tri_vertices = [
            (midpoints[0], midpoints[1], midpoints[2]),
            (self._vertices[0], midpoints[0], midpoints[2]),
            (self._vertices[1], midpoints[1], midpoints[0]),
            (self._vertices[2], midpoints[2], midpoints[1]),
        ]
        split_pieces = [Triangle(v) for v in tri_vertices]

        return split_pieces

    def inside_positions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all positions inside `self`. We define inside to include the
        boundaries of `self`.
        """

        if "inside_positions" in self._cache:
            return self._cache["inside_positions"]

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.is_inside((x, y)):
                    positions.append((x, y))

        self._cache["inside_positions"] = list(positions)
        return positions

    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """ Returns True if `pos` is inside `self`, False otherwise. """

        (ax, ay), (bx, by), (cx, cy) = self._vertices
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

        left = min(self._vertices[i][0] for i in range(self.num_vertices))
        top = min(self._vertices[i][1] for i in range(self.num_vertices))
        right = max(self._vertices[i][0] for i in range(self.num_vertices))
        bottom = max(self._vertices[i][1] for i in range(self.num_vertices))
        return ((left, top), (right, bottom))
