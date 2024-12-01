#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from dataclasses import dataclass, fields, field
from functools import cached_property, cache
from itertools import pairwise
from typing import Iterator, Any


from .math import Point3, Vector3

@dataclass
class Polyline:
    """
    A structure that defines a piecewise linear path made up of a number of
    straight line segments.

    A class that represents a polyline, defined as a series of connected points
    in 3D space.

    A polyline consists of a sequence of straight line segments connecting a
    list of `Point3` objects. The class provides methods to calculate the total
    length of the polyline, iterate over its points, and retrieve individual
    segments (pairs of consecutive points).

    Attributes:

    points : list of Point3

    - A list of `Point3` objects that define the vertices
    of the polyline. At least two points are required to form a polyline.

    Methods:

    length():

    - Computes the total length of the polyline by manually iterating over
      consecutive points and summing the Euclidean distances between them. This
      method uses a more straightforward approach that may be slower for large
      polylines.

    length_vectorized():

    - Computes the total length of the polyline using NumPy for a vectorized
      approach. This method is typically faster for larger polylines as it
      leverages optimized array operations.

    segments():

    - Returns an iterator that yields consecutive pairs of points(as tuples)
      representing the segments of the polyline. Each segment is defined by two
      points.

    Raises:

    - ValueError: If fewer than two points are provided during initialization, as
    at least two points are required to define a polyline.

    Example:

    >>> points = [Point3(0, 0, 0), Point3(1, 1, 1), Point3(2, 2, 2)]
    >>> polyline = Polyline(points)
    >>> polyline.length_vectorized
    2.8284271247461903  # Total length of the polyline
    >>> for segment in polyline.segments():
    >>>     print(segment)
    (Point3(0, 0, 0), Point3(1, 1, 1))
    (Point3(1, 1, 1), Point3(2, 2, 2))

    """

    points: list[Point3] = field(default_factory=list)

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("At least two points are required to construct a Polyline!")

    @cached_property
    def length(self) -> float:
        """
        Calculate and return the total length of the Polyline.
        """
        return sum((p2 - p1).norm() for p1, p2 in self.segments())

    def segments(self) -> Iterator[tuple]:
        """
        Yield pairs of consecutive points representing each segment in the Polyline.
        """
        return pairwise(self.points)

    def __iter__(self) -> Iterator[Point3]:
        """
        Make the Polyline iterable, allowing iteration over its points.
        """
        return iter(self.points)

    def __getitem__(self, index: int) -> Point3:
        """
        Allow indexing into the Polyline points.
        """
        return self.points[index]

    def __len__(self) -> int:
        """
        Return the number of points in the Polyline.
        """
        return len(self.points)


@dataclass()
class Hole:
    """
    A class representing a hole. Essentially it is represented by a
    series of points starting from the hole collar and ending at the
    toe

    # Parameters (kwargs)

    meta: dict(str, Any)
        - A dictionary containing various metadata

    points: list(Point3)
        - The list of points making up the geometry of the hole.

    # Attributes

    polyline:Polyline
        - A polyline representing the geometry of the hole.
        - The coordinate are in terms of the ring coordinate system.
    """

    meta: dict[str, Any] = None
    points: list[Point3] = field(default_factory=list)

    def __post_init__(self):
        """
        """

        if not self.points or len(self.points) < 2:
            raise ValueError("At least two points are required to construct a Hole!")

        # Even though the class is frozen, we can use object.__setattr__ to modify attributes
        object.__setattr__(self, 'polyline', Polyline(self.points))

    @cached_property
    def collar(self):
        """
        The collar point of the hole. Usually the first point in the list.
        """

        return self.points[0]

    @cached_property
    def toe(self):
        """
        The toe point of the hole. Usually the last point in the list.
        """

        return self.points[-1]

    @property
    def length(self):
        """
        A helper method to return the length of the hole.

        NOTE: no need to cache this property as the length is already cached.
        """

        return self.polyline.length

    @cache
    def point_at_distance(self, distance:float|int) -> Point3:
        """
        This method will return a coordinate that is a certain distance from
        the collar of the hole.

        # Exceptions

        ValueError - Indicating that the distance is less than 0
        ValueError - Indicating that the distance is greater than the hole
                     length.

        """

        if distance < 0:
            raise ValueError("Distance must be greater than or equal to 0.")

        if distance > self.length:
            raise ValueError("Distance must be less than or equal to the length of the hole.")

        if distance == 0:
            return self.collar

        if distance == self.length:
            return self.toe

        cumulative_length = 0.0

        for p1, p2 in self.polyline.segments():

            segment_vector = (p2 - p1)
            segment_length = segment_vector.norm()

            # Does the distance fall on the current segment?

            if cumulative_length + segment_length >= distance:

                # compute the distance from the beginning of this segment to the
                # target distance
                remaining_distance = distance - cumulative_length

                return p1 + segment_vector.normalize() * remaining_distance

            cumulative_length += segment_length

        # Fail safe
        return self.toe

    def __iter__(self) -> Iterator:
        """
        Make it so we can iterate through the points stored in the polyline

        >>> for p in hole:
              print(p)
        """
        return iter(self.polyline)

    def __hash__(self):
        # Define a custom hash using an immutable attribute (such as points)
        return hash(tuple(self.points))


@dataclass
class Ring:
    """
    A simple data class to hold the values of the ring.
    """

    ring_id: int
    holes: list[Hole] = field(default_factory=list)  # a list of holes
    origin: Optional[Point3] = None  # the origin point of the ring
    azimuth: Optional[float] = None  # Azimuth of the ring line in degrees
    dump: Optional[float] = None
    normal: Optional[Vector3] = None

    def __str__(self) -> str:

        attributes = [
            f"Ring - {self.ring_id}",
            f"Origin = ({self.origin.x:.3f}, {self.origin.y:.3f}, {self.origin.z:.3f})" if self.origin else "Origin = None",
            f"Normal = ({self.normal.x:.3f}, {self.normal.y:.3f}, {self.normal.z:.3f})" if self.normal else "Normal = None",
            f"Azimuth = {self.azimuth:.3f}" if self.azimuth is not None else "Azimuth = None",
            f"Dump = {self.dump:.3f}" if self.dump is not None else "Dump = None"
        ]
        return "\n".join(attributes)

    def __repr__(self) -> str:
        return f"Ring(ring_id={self.ring_id}, dump={self.dump})"
