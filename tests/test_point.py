#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pytest
import numpy as np

from ring_fit.math import (
    Point3,
    Vector3,
)


class TestPoint3:
    def test_add_points1(self):
        """ """

        m = Point3(1.0, 2.0, 1.0)
        n = Point3(3.0, 2.0, 1.0)

        # addition between point objects doesn't make sense in terms of points
        with pytest.raises(NotImplementedError) as e_info:
            result = m + n

        with pytest.raises(NotImplementedError) as e_info:
            result = n + m

    def test_add_points2(self):
        """ """

        m = Point3(3.0, 2.0, 1.0)
        n = Point3(-1.0, 14.5, 2.0)

        # addition between point objects doesn't make sense in terms of points
        with pytest.raises(NotImplementedError) as e_info:
            result = m + n

        with pytest.raises(NotImplementedError) as e_info:
            result = n + m

    # Subtract
    def test_subtract_points1(self):
        """ """

        m = Point3(1.0, 2.0, 1.0)
        n = Point3(3.0, 2.0, 1.0)

        assert m - n == Point3(-2, 0, 0)

    def test_subtract_points11(self):
        """ """

        m = Point3(1.0, 2.0, 1.0)
        n = Point3(3.0, 2.0, 1.0)

        assert n - m == Point3(2, 0, 0)

    def test_subtract_points2(self):
        """ """

        m = Point3(3.0, 2.0, 1.0)
        n = Point3(-1.0, 14.5, 2.0)

        assert m - n == Point3(4, -12.5, -1)

    def test_subtract_points21(self):
        """ """

        m = Point3(3.0, 2.0, 1.0)
        n = Point3(-1.0, 14.5, 2.0)

        assert n - m == Point3(-4, 12.5, 1)

    # Dot
    def test_dot_points1(self):
        """ """

        m = Point3(1.0, 2.0, 1.0)
        n = Point3(3.0, 2.0, 1.0)

        assert m.dot(n) == 8

    def test_dot_points2(self):
        """ """

        m = Point3(3.0, 2.0, 1.0)
        n = Point3(-1.0, 14.5, 2.0)

        assert m.dot(n) == 28


def test_point_initialization():
    # Standard initialization
    p = Point3(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert len(p) == 3


def test_add_point_vector():
    # Point + Vector -> Point
    p = Point3(1, 2, 3)
    v = Vector3(1, 1, 1)
    result = p + v
    assert isinstance(result, Point3)
    assert result.coordinates.tolist() == [2, 3, 4]


def test_subtract_point_vector():
    # Point - Vector -> Point
    p = Point3(1, 2, 3)
    v = Vector3(1, 1, 1)
    result = p - v
    assert isinstance(result, Point3)
    assert result.coordinates.tolist() == [0, 1, 2]


def test_subtract_two_points():
    # Point - Point -> Vector
    p1 = Point3(1, 2, 3)
    p2 = Point3(0, 1, 2)
    result = p1 - p2
    assert isinstance(result, Vector3)
    assert result.coordinates.tolist() == [1, 1, 1]


def test_dot_product():
    # Dot product with another Point
    p1 = Point3(1, 2, 3)
    p2 = Point3(4, 5, 6)
    result = p1.dot(p2)
    assert np.isclose(result, 32)


def test_multiplication_by_scalar():
    # Point * scalar -> Point
    p = Point3(1, 2, 3)
    result = p * 2
    assert isinstance(result, Point3)
    assert result.coordinates.tolist() == [2, 4, 6]


def test_division_by_scalar():
    # Point / scalar -> Point
    p = Point3(4, 6, 8)
    result = p / 2
    assert isinstance(result, Point3)
    assert result.coordinates.tolist() == [2, 3, 4]


def test_equality():
    # Check equality of two points
    p1 = Point3(1, 2, 3)
    p2 = Point3(1, 2, 3)
    assert p1 == p2


def test_inequality():
    # Check inequality of two points
    p1 = Point3(1, 2, 3)
    p2 = Point3(3, 2, 1)
    assert p1 != p2


def test_numpy_array_conversion():
    # Convert to numpy array
    p = Point3(1, 2, 3)
    arr = np.array(p.coordinates)
    assert np.array_equal(arr, np.array([1, 2, 3]))


def test_str():
    # String representation
    p = Point3(1, 2, 3)
    assert str(p) == "Point3(1, 2, 3)"


def test_repr():
    # Repr representation
    p = Point3(1, 2, 3)
    assert repr(p) == "Point3([1, 2, 3])"


def test_hash():
    # Hashing
    p1 = Point3(1, 2, 3)
    p2 = Point3(1, 2, 3)
    assert hash(p1) == hash(p2)
