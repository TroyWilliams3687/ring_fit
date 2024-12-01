#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class VectorCommonMixin:
    """
    Mixin class that provides common operations and overloads for vector-like
    objects.

    Assumes the class that inherits from this mixin has an attribute
    `coordinates`, which is an array-like structure (e.g., a NumPy array)
    representing the vector's components.
    """

    def dot(self, other) -> float:
        """
        Calculate the dot product (inner product) between the current instance
        of the instance and the other instance.
        """

        return np.dot(self.coordinates, other).item()

    def __eq__(self, other):
        return np.all(self.coordinates == other.coordinates)

    def __len__(self):
        return len(self.coordinates)

    def __hash__(self):
        return hash(tuple(self.coordinates))

    def __array__(self, dtype=None, copy=False):
        """
        Convert to a NumPy array, supporting dtype and copy
        keywords.
        """

        return np.array(self.coordinates, dtype=dtype, copy=copy)

    def __add__(self, other):
        return  self.__class__(self.coordinates  + other)

    def __sub__(self, other):
        return self.__class__(self.coordinates  - other)

    def __mul__(self, scalar:float|int):
        """
        Multiply the current instance by the scalar value
        """

        if not isinstance(scalar, (int, float, np.integer, np.floating)):
            raise TypeError("Only scalar multiplication is supported.")

        return self.__class__(self.coordinates * scalar)

    def __rmul__(self, scalar:float|int):
        return self.__mul__(scalar)

    def __truediv__(self, scalar:float|int):
        """
        Divide the current instance of the Point by the scalar value
        """

        if type(scalar) not in (int, float, np.integer, np.floating):
            raise TypeError("Division requires a scalar of type int or float.")

        if scalar == 0:
            raise ValueError("Division by zero is not allowed.")

        return self.__class__(self.coordinates / scalar)

    def __iter__(self) -> Iterator:
        """
        Make the vector iterable.
        """

        return iter(self.coordinates)


class VectorMixin:
    """
    A mixin class providing methods specific to Vector objects or things that
    act like Vectors.

    NOTE: A Point is not quite a vector.
    """

    def norm(self) -> float:
        """
        Calculate the norm/magnitude/length of the vector.
        """

        return np.linalg.norm(self.coordinates)

    def normalize(self):
        """
        Returns a Vector that has been normalized i.e. the magnitude is 1.
        """

        m = self.norm()

        return self.__class__(self.coordinates / m) if m != 0.0 else self.__class__(self.coordinates * 0.0)


    def negation(self):
        """
        Return the inverse of the vector
        """
        return -1 * self



class Vector3(VectorCommonMixin, VectorMixin):
    """
    A representation of a 3D vector.

    This class provides vector-specific functionality, such as handling 3D
    vector operations and cross product. It also inherits basic arithmetic
    operations from VectorMathMixin.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Vector3 object.

        Args : tuple
            A sequence of coordinates (x, y, z) or array-like object.

        """


        coordinates = np.asarray(args[0] if len(args) == 1 else args)

        # Ensure the coordinates are 3D
        if coordinates.shape != (3,):
            raise ValueError("Expected 3 coordinates for a 3D vector.")

        self.coordinates = coordinates # np.array(coordinates, dtype=float)


    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

    @property
    def z(self) -> float:
        return self.coordinates[2]

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0, 0, 0)

    @staticmethod
    def up() -> "Vector3":
        return Vector3(0, 0, 1)

    @staticmethod
    def down() -> "Vector3":
        return Vector3(0, 0, -1)

    @staticmethod
    def north() -> "Vector3":
        return Vector3(0, 1, 0)

    @staticmethod
    def south() -> "Vector3":
        return Vector3(0, -1, 0)

    @staticmethod
    def east() -> "Vector3":
        return Vector3(1, 0, 0)

    @staticmethod
    def west() -> "Vector3":
        return Vector3(-1, 0, 0)

    def cross(self, other) -> "Vector3":
        """
        Compute the cross product between this vector and another vector
        """

        # Converts both Vector3 and array-like objects to a NumPy array
        other_array = np.asarray(other)

        if other_array.shape != (3,):
            raise ValueError(
                "Cross product requires a 3D vector or array-like object of length 3."
            )

        return self.__class__(np.cross(self.coordinates, other_array))

    def __str__(self) -> str:
        """
        Display a user-friendly string representation of the vector.
        """
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        """
        Display an official string representation that can ideally recreate the object.
        """

        return f"Vector3({self.coordinates.tolist()})"


def vector_from_angles(azimuth: float, dip: float) -> Vector3:
    """
    Construct a vector from the angles. Essentially we are converting a
    point defined in spherical coordinates to a vector.

    However, this spherical coordinate system is different from the traditional
    system. It is the system used by gyro survey companies that measure azimuth
    angles and dip angles at survey points a certain distance along the hole.

    The system is a right-handed coordinate system with x (easting), y (northing) and
    z (elevation).

    The azimuth is measured from true north (i.e. y vector). It ranges from: [0, 2pi].

    The dip is the angle from the horizontal XY plane. It ranges from: [-pi/2, pi/2].

    Args:

    - Azimuth - Angle (radians) measured from North Vector (0,1,0)
    on the XY plane. Usually it is measured clockwise.

    - Dump - Angle (radians), measured from the horizontal plane
    with normal (0,0,1) to the vector.

    Returns:

    - a Vector3 object representing a tangent vector pointing in the direction
    specified by the angle.

    References:

    - https://en.wikipedia.org/wiki/Spherical_coordinate_system

    NOTE:

    - This vector is not normalized explicitly.

    - This should probably be part of the orthonormal basis. That is,
    if we want angles, it should be a function of the basis.
    """

    # theta = azimuth
    # phi = dip

    # Clamp the angles to the range [0, 2π]
    theta = np.mod(azimuth, 2 * np.pi)
    phi = np.mod(dip, 2 * np.pi)

    x = np.cos(phi) * np.sin(theta)
    y = np.cos(phi) * np.cos(theta)
    z = np.sin(phi)

    return Vector3((x, y, z))


def calculate_azimuth_and_dip(vector: Vector3) -> tuple[float, float]:
    """
    From the Vector, determine the azimuth and dip.

    The azimuth is measured from true north (i.e. y vector). It ranges from: [0, 2pi].

    The dip is the angle from the horizontal XY plane. It ranges from: [-pi/2, pi/2].

    Args:

    vector - Vector3

    Returns:

    tuple[azimuth, dip]

    A tuple containing the azimuth and the dip from the Vector3.

    NOTE:

    This should probably be part of the orthonormal basis. That is,
    if we want angles, it should be a function of the basis.

    """

    # make sure the vector is normalized first
    norm = vector.normalize()

    azimuth = np.arctan2(-norm.x, -norm.y) + np.pi
    dip = np.arcsin(norm.z)

    return azimuth, dip


class PointMixin:
    """
    This mixin is built for the Point3 class and is used to override some
    properties that would be inherited from the `VectorCommonMixin`

    """

    def __add__(self, other):
        """
        We can only add a Point + Vector producing a new Point
        """

        if issubclass(type(other), Point3):
            raise NotImplementedError(
                "Addition is not supported between Point3!"
            )

        return  self.__class__(self.coordinates  + other)

    def __sub__(self, other):
        """
        We can subtract a point from another point producing a vector.

        We can subtract a vector from a point producing another point.
        """

        if issubclass(type(other), Point3):
            # We get a vector if we subtract two points.
            return Vector3(self.coordinates - other)

        # We can subtract a vector from a Point3
        return self.__class__(self.coordinates  - other)


class Point3(PointMixin, VectorCommonMixin):
    """
    A representation of a 3 dimensional point. A point is different
    from a vector as it represents a location in space. A vector
    typically represents a direction and magnitude.
    """

    def __init__(self, *args, **kwargs):
        """

        args:

        - coordinates - enumerable
            - (x, y, z)
            - x - number - The x-coordinate of the Point
            - y - number - The y-coordinate of the Point
            - z - number - The z-coordinate of the Point

        """

        coordinates = np.asarray(args[0] if len(args) == 1 else args)

        # Ensure the coordinates are 3D
        if coordinates.shape != (3,):
            raise ValueError("Expected 3 coordinates for a 3D point.")

        self.coordinates = coordinates #np.array(coordinates, dtype=float)

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]


    def __str__(self) -> str:
        """
        Display a user-friendly string representation of the vector.
        """
        return f"Point3({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        """
        Display an official string representation that can ideally recreate the object.
        """

        return f"Point3({self.coordinates.tolist()})"


@dataclass
class Plane:
    """Represents a plane in 3D space.

    Attributes:

    - normal (Vector3): Normal vector of the plane.
    - point (Point3): A point on the plane (centroid).
    - d (float): Offset from the origin.
    """
    normal: Vector3
    point: Point3
    d: float

    def __str__(self) -> str:
        """
        Generates a string representation of the plane.

        Returns:

        str: Plane equation in the format `ax + by + cz + d = 0`.
        """

        a, b, c = self.normal.coordinates
        return f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {self.d:.3f} = 0"


def best_fit_plane(points: npt.NDArray[np.float64]) -> Plane:
    """
    Calculates the best-fit plane for a cloud of 3D points.

    Args:

    - points (npt.NDArray[np.float64]): A 2D array of shape (n, 3) where each row
    is a 3D point (x, y, z).

    Returns:

    - Plane: A Plane object containing the normal vector, a point on the plane
      (centroid), and the plane's offset from the origin.

    Raises:

    ValueError: If the input array does not have shape (n, 3).
    """
    if points.shape[1] != 3:
        raise ValueError("Input array must have shape (n, 3) for 3D points.")

    # Calculate the centroid of the points
    centroid_array = points.mean(axis=0)
    centroid = Point3(centroid_array)

    # Subtract the centroid to center the points
    centered_points = points - centroid_array

    # Compute the SVD
    _, _, vh = np.linalg.svd(centered_points)

    # The last row of vh corresponds to the smallest singular value
    # and is the normal vector of the plane
    normal_array = vh[-1]
    normal = Vector3(normal_array).normalize()

    # Compute the plane's offset from the origin
    d = -np.dot(normal_array, centroid_array)

    return Plane(normal=normal, point=centroid, d=d)


