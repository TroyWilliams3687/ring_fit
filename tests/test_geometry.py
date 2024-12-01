#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pytest
import numpy as np

from ring_fit.math import (
    Point3,
)

from ring_fit.geometry import (
    Polyline,
    Hole,
)


def test_polyline_initialization():
    # Test valid initialization
    points = [Point3(0, 0, 0), Point3(1, 1, 1)]
    polyline = Polyline(points)
    assert len(polyline) == 2
    assert polyline.points == points

    # Test invalid initialization
    with pytest.raises(ValueError, match="At least two points are required"):
        Polyline([Point3(0, 0, 0)])


def test_polyline_segments():
    points = [Point3(0, 0, 0), Point3(1, 1, 1), Point3(2, 2, 2)]
    polyline = Polyline(points)

    # Test segment generation
    segments = list(polyline.segments())
    assert len(segments) == 2
    assert segments == [(Point3(0, 0, 0), Point3(1, 1, 1)),
                        (Point3(1, 1, 1), Point3(2, 2, 2))]


def test_polyline_length():
    points = [Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 1, 0)]
    polyline = Polyline(points)

    # Calculate the expected length
    expected_length = (Point3(1, 0, 0) - Point3(0, 0, 0)).norm() + \
                      (Point3(1, 1, 0) - Point3(1, 0, 0)).norm()

    # Test length calculation
    assert polyline.length == pytest.approx(expected_length)

def test_polyline_length1():

    pl = Polyline([
        Point3(0,0,0),
        Point3(1,1,0),
        Point3(2,3,4),
    ])

    assert pl.length == 5.996789257328935

def test_polyline_iteration():
    points = [Point3(0, 0, 0), Point3(1, 0, 0)]
    polyline = Polyline(points)

    # Test iteration over points
    iterated_points = [p for p in polyline]
    assert iterated_points == points


def test_polyline_indexing():
    points = [Point3(0, 0, 0), Point3(1, 0, 0)]
    polyline = Polyline(points)

    # Test indexing
    assert polyline[0] == Point3(0, 0, 0)
    assert polyline[1] == Point3(1, 0, 0)

    # Test invalid indexing
    with pytest.raises(IndexError):
        _ = polyline[2]


def test_polyline_cached_length():
    points = [Point3(0, 0, 0), Point3(3, 4, 0)]  # Distance is 5 units
    polyline = Polyline(points)

    # Test cached length calculation
    assert polyline.length == 5

    # Modify points (this wouldn't normally happen in this immutable case)
    polyline.points.append(Point3(6, 8, 0))

    # Check that the cached length hasn't changed despite point modification
    assert polyline.length == 5  # Cached result is still used

# ----
# Hole

data = [
        ({
         'points':(
            Point3(1,0,0),
            Point3(1,1,0),
            Point3(1,2,0))
         },
         {'segments':2, 'length':2.0, 'point_distance':1.5, 'point_at_distance': Point3(1.0, 1.5, 0)}),
        ({
         'points':(
            Point3(5,8,2),
            Point3(1,1,0),
            Point3(1,2,0),
            Point3(1,2,0))
         },
         {'segments':3, 'length':9.3066, 'point_distance':1.5, 'point_at_distance':Point3(4.2777, 6.7359, 1.6388)}
        ),
       ]


@pytest.mark.parametrize('data', data)
def test_hole(data):

    bh = Hole(**data[0])

    results = data[1]

    assert len(list(bh.polyline.segments())) == results['segments']

    assert bh.length == pytest.approx(results['length'], rel=1E-10, abs=1E-4)

    cp = bh.point_at_distance(results['point_distance'])
    rp = results['point_at_distance']

    assert cp.x == pytest.approx(rp.x, rel=1E-10, abs=1E-4)
    assert cp.y == pytest.approx(rp.y, rel=1E-10, abs=1E-4)
    assert cp.z == pytest.approx(rp.z, rel=1E-10, abs=1E-4)


errors = [{'points':None},
          {'points':(Point3(0,0,0), )}]

@pytest.mark.parametrize('errors', errors)
def test_hole_errors(errors):

    with pytest.raises(ValueError):
        bh = Hole(**errors)


def test_hole_initialization():
    # Test successful initialization
    points = [Point3(0, 0, 0), Point3(1, 1, 1)]
    hole = Hole(points=points)
    assert hole.collar == points[0]
    assert hole.toe == points[-1]
    assert hole.length == pytest.approx(np.sqrt(3))

    # Test failure with fewer than two points
    with pytest.raises(ValueError, match="At least two points are required"):
        Hole(points=[Point3(0, 0, 0)])

def test_hole_length():
    # Test the length calculation of the hole
    points = [Point3(0, 0, 0), Point3(3, 4, 0)]  # Distance should be 5
    hole = Hole(points=points)
    assert hole.length == pytest.approx(5)

def test_point_at_distance_start():
    # Test distance at collar
    points = [Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 1, 0)]
    hole = Hole(points=points)

    assert hole.point_at_distance(0) == points[0]  # Collar

def test_point_at_distance_end():
    # Test distance at toe
    points = [Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 1, 0)]
    hole = Hole(points=points)

    assert hole.point_at_distance(hole.length) == points[-1]  # Toe

def test_point_at_distance_mid_segment():
    # Test a point along the first segment
    points = [Point3(0, 0, 0), Point3(5, 0, 0), Point3(5, 5, 0)]
    hole = Hole(points=points)

    # Distance 3 units from collar should be on the first segment
    point_at_3 = hole.point_at_distance(3)
    assert point_at_3 == Point3(3, 0, 0)

def test_point_at_distance_on_second_segment():
    # Test a point on the second segment
    points = [Point3(0, 0, 0), Point3(5, 0, 0), Point3(5, 5, 0)]
    hole = Hole(points=points)

    # Distance 7 units from collar should be on the second segment
    point_at_7 = hole.point_at_distance(7)
    assert point_at_7 == Point3(5, 2, 0)

def test_point_at_distance_out_of_range():
    # Test error when distance is greater than the hole length
    points = [Point3(0, 0, 0), Point3(1, 0, 0)]
    hole = Hole(points=points)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        hole.point_at_distance(-1)  # Negative distance

    with pytest.raises(ValueError, match="less than or equal to the length"):
        hole.point_at_distance(hole.length + 1)  # Distance exceeds length

def test_point_at_distance_cache():
    # Test that caching works
    points = [Point3(0, 0, 0), Point3(5, 0, 0), Point3(5, 5, 0)]
    hole = Hole(points=points)

    # First call should compute and cache
    point_at_3 = hole.point_at_distance(3)
    assert point_at_3 == Point3(3, 0, 0)

    # Modify polyline to ensure no recalculation occurs if cache is used
    hole.polyline.points = [Point3(0, 0, 0), Point3(10, 0, 0), Point3(10, 10, 0)]

    # Check that the cached result is returned (it should still be the old value)
    point_at_3_cached = hole.point_at_distance(3)
    assert point_at_3_cached == Point3(3, 0, 0)

    # Clear the cache and recalculate
    hole.point_at_distance.cache_clear()
    point_at_3_updated = hole.point_at_distance(3)
    assert point_at_3_updated == Point3(3, 0, 0)  # Should reflect new polyline geometry
