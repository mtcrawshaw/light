""" Definition of Shape object used as a piece of a mosaic. """

import multiprocessing
from multiprocessing import Process
from typing import List, Tuple


class Shape:
    """
    A piece of a mosaic. Note that we follow Pillow's convention of placing the origin
    at the top left of the image, so that values of the y-axis increase as you move
    down. This is an abstract class and shouldn't be instantiated.
    """

    def __init__(
        self,
        max_child_area: float = 0.7,
        num_samples: int = 100,
        num_workers: int = 1,
        boundary_width: int = 3,
    ) -> None:
        """ Init function for Shape. """

        self.max_child_area = max_child_area
        self.num_samples = num_samples
        self.boundary_width = boundary_width
        self.num_workers = num_workers
        self.children: List["Shape"] = []

        # This is a dictionay that holds cached results from function calls to this
        # class, namely `inside_positions()`, `unique_inside_positions()`, and
        # `boundary_positions()`. We have to be careful here: any time that the class
        # data changes in a way that affects the result of these function calls, the
        # corresponding cached result needs to be deleted from `self.cache`.
        self.cache = {}

    def partition(self, shape_cls) -> None:
        """ Add a single non-overlapping Shape to children. """

        valid_positions = self.unique_inside_positions()

        # Set up multi-processed sampling.
        manager = multiprocessing.Manager()
        sample_children = manager.list()

        def worker():
            samples_per_worker = max(self.num_samples // self.num_workers, 1)
            for sample in range(samples_per_worker):
                print("  Sample %d" % sample)
                candidate = shape_cls.sample_inside(self, valid_positions)
                sample_children.append(candidate)

        # Perform sampling.
        processes = []
        for i in range(self.num_workers):
            processes.append(Process(target=worker))
            processes[i].start()
        for i in range(self.num_workers):
            processes[i].join()

        # Find biggest child.
        biggest_child = None
        biggest_child_area = None
        parent_area = self.area()
        for candidate in sample_children:
            area = candidate.area()
            if (
                biggest_child_area is None
                or biggest_child_area <= area <= self.max_child_area * parent_area
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

        if "inside_positions" in self.cache:
            return self.cache["inside_positions"]

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.is_inside((x, y)):
                    positions.append((x, y))

        self.cache["inside_positions"] = list(positions)
        return positions

    def unique_inside_positions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all positions inside `self` which aren't inside of any
        children of `self.`.
        """

        if "unique_inside_positions" in self.cache:
            return self.cache["unique_inside_positions"]

        positions: List[Tuple[int, int]] = self.inside_positions()
        child_positions: List[Tuple[int, int]] = []
        for child in self.children:
            child_positions += child.inside_positions()

        unique_positions: List[Tuple[int, int]] = []
        for position in positions:
            if position not in child_positions:
                unique_positions.append(position)

        self.cache["unique_inside_positions"] = list(unique_positions)
        return unique_positions

    def boundary_positions(self) -> List[Tuple[int, int]]:
        """
        Return all positions on the boundary of `self`.
        """

        if "boundary_positions" in self.cache:
            return self.cache["boundary_positions"]

        positions: List[Tuple[int, int]] = []
        (left, top), (right, bottom) = self.bounds()
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                if self.is_boundary((x, y)):
                    positions.append((x, y))

        self.cache["boundary_positions"] = list(positions)
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
    def sample_inside(
        cls, shape: "Shape", valid_positions: List[Tuple[int, int]]
    ) -> "Shape":
        """
        Return a randomly sampled instance of `Shape` that lies inside
        `shape.unique_inside_positions`. This function should be overridden in
        subclasses.
        """

        raise NotImplementedError
