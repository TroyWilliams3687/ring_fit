#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pytest
import numpy as np

from ring_fit.math import (
    Vector3,
    calculate_azimuth_and_dip,
    vector_from_angles,
)


class TestVector3:
    # Add
    def test_add_1(self):
        """ """
        m = Vector3(1.0, 2.0, -3.0)
        n = Vector3(3.0, 2.0, 0.0)

        assert m + n == Vector3(4, 4, -3)
        assert n + n == Vector3(6, 4, 0)
        assert n + m == Vector3(4, 4, -3)

    # Subtract
    def test_subtract_1(self):
        """ """
        m = Vector3(1.0, 2.0, -3.0)
        n = Vector3(3.0, 2.0, 0.0)

        assert m - n == Vector3(-2, 0, -3)
        assert n - n == Vector3(0, 0, 0)
        assert n - m == Vector3(2, 0, 3)

    # scaler_multiply
    def test_scalar_multiply_1(self):
        """ """

        V = Vector3(1.0, 2.0, 3.0)
        assert -2.0 * V == Vector3(-2.0, -4.0, -6.0)
        assert 0.0 * V == Vector3(0.0, 0.0, 0.0)
        assert 2.0 * V == Vector3(2.0, 4.0, 6.0)

    # norm
    def test_norm(self):
        """ """

        v1 = Vector3(1.0, 2.0, 0.0)
        assert v1.norm() == 2.23606797749978969641

        v2 = Vector3(3.0, 2.0, 1.0)
        assert v2.norm() == 3.74165738677394138558

        v3 = Vector3(-1.0, 14.5, -5)
        assert v3.norm() == 15.37042614893939757483

    # normalize
    def test_normalize(self):
        """ """

        v1 = Vector3(1.0, 2.0, 0.0)

        assert (
            np.testing.assert_allclose(
                v1.normalize(),
                Vector3(0.447213595, 0.894427191, 0.0),
                atol=1e-7,
                rtol=1e-6,
            )
            == None
        )

        v2 = Vector3(3.0, 2.0, 1.0)
        assert (
            np.testing.assert_allclose(
                v2.normalize(),
                Vector3(0.80178373, 0.5345225, 0.26726124),
                atol=1e-6,
                rtol=1e-8,
            )
            == None
        )

        v3 = Vector3(-1.0, 14.5, 35.678)
        result = v3.normalize()
        assert (
            np.testing.assert_allclose(
                v3.normalize(),
                Vector3(-0.025957, 0.37638, 0.926102),
                atol=1e-6,
                rtol=1e-8,
            )
            == None
        )

    # dot
    def test_dot_1(self):
        """ """

        v1 = Vector3(1.0, 2.0, -2.0)
        v2 = Vector3(3.0, 2.0, 0.0)
        result = v1.dot(v2)

        assert result == 7.0

    def test_dot_2(self):
        """ """

        v1 = Vector3(1.0, 2.0, 1.0)
        v3 = Vector3(-1.0, 14.5, 2.0)
        result = v1.dot(v3)

        assert result == 30.0

    def test_dot_3(self):
        """ """

        v2 = Vector3(3.0, 2.0, 0.0)
        v3 = Vector3(-1.0, 14.5, 0.0)
        result = v2.dot(v3)

        assert result == 26

    # negation
    def test_negation(self):
        """ """

        v1 = Vector3(1.0, 2.0, 1.0)
        result = v1.negation()
        assert result == Vector3(-1.0, -2.0, -1.0)

        v2 = Vector3(3.0, 2.0, 0.0)
        result = v2.negation()
        assert result == Vector3(-3.0, -2.0, 0.0)

        v3 = Vector3(-1.0, 14.5, -100.0)
        result = v3.negation()
        assert result == Vector3(1.0, -14.5, 100.0)

    # Angles
    def test_angles(self):
        """ """

        v = Vector3(0.0, 1.0, 0.0)
        azimuth, dip = calculate_azimuth_and_dip(v)
        assert azimuth == 0
        assert dip == 0

        v = Vector3(1.0, 0.0, 1.0)
        azimuth, dip = calculate_azimuth_and_dip(v)
        assert azimuth == np.radians(90)
        assert dip == pytest.approx(np.radians(45))

        v = Vector3(1.0, 0.0, 14.5)
        azimuth, dip = calculate_azimuth_and_dip(v)
        assert azimuth == np.radians(90)
        assert dip == pytest.approx(np.radians(86.0548))

        v = Vector3(0.0, -1.0, -100.0)
        azimuth, dip = calculate_azimuth_and_dip(v)
        assert azimuth == np.radians(180)
        assert dip == pytest.approx(np.radians(-89.427))

        v = Vector3(-1.0, 0.0, 25.0)
        azimuth, dip = calculate_azimuth_and_dip(v)
        assert azimuth == np.radians(270)
        assert dip == pytest.approx(np.radians(87.7094))


def test_vector_initialization():
    v = Vector3(1, 2, 3)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3
    assert len(v) == 3



def test_norm():
    v = Vector3(3, 4, 0)
    assert v.norm() == 5.0


def test_normalize():
    v = Vector3(0, 3, 4)
    normed_v = v.normalize()
    assert np.isclose(normed_v.norm(), 1.0)


def test_dot_product():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, -5, 6)
    assert v1.dot(v2) == 12


def test_cross_product():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, 5, 6)
    v3 = v1.cross(v2)
    assert v3.coordinates.tolist() == [-3, 6, -3]


def test_scalar_multiplication():
    v = Vector3(1, 2, 3)
    v_scaled = v * 3
    assert v_scaled.coordinates.tolist() == [3, 6, 9]


def test_addition():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, 5, 6)
    v3 = v1 + v2
    assert v3.coordinates.tolist() == [5, 7, 9]


def test_subtraction():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, 5, 6)
    v3 = v1 - v2
    assert v3.coordinates.tolist() == [-3, -3, -3]


def test_zero_vector():
    v = Vector3.zero()
    assert v.coordinates.tolist() == [0, 0, 0]


def test_up_vector():
    v = Vector3.up()
    assert v.coordinates.tolist() == [0, 0, 1]


def test_down_vector():
    v = Vector3.down()
    assert v.coordinates.tolist() == [0, 0, -1]


def test_north_vector():
    v = Vector3.north()
    assert v.coordinates.tolist() == [0, 1, 0]


def test_south_vector():
    v = Vector3.south()
    assert v.coordinates.tolist() == [0, -1, 0]


def test_east_vector():
    v = Vector3.east()
    assert v.coordinates.tolist() == [1, 0, 0]


def test_west_vector():
    v = Vector3.west()
    assert v.coordinates.tolist() == [-1, 0, 0]



def test_negation():
    v = Vector3(1, 2, 3)
    neg_v = v.negation()
    assert neg_v.coordinates.tolist() == [-1, -2, -3]


def test_equality():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(1, 2, 3)
    assert v1 == v2


def test_inequality():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(3, 2, 1)
    assert v1 != v2


def test_numpy_array_conversion():
    v = Vector3(1, 2, 3)
    arr = np.array(v.coordinates)
    assert np.array_equal(arr, np.array([1, 2, 3]))


def test_str():
    v = Vector3(1, 2, 3)
    assert str(v) == "Vector3(1, 2, 3)"


def test_repr():
    v = Vector3(1, 2, 3)
    assert repr(v) == "Vector3([1, 2, 3])"


def test_hash():
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(1, 2, 3)
    assert hash(v1) == hash(v2)


def test_vector_from_angles_zero():
    # Azimuth and dip are zero; should point along the y-axis (north).
    v = vector_from_angles(0, 0)
    assert np.isclose(v.x, 0.0)
    assert np.isclose(v.y, 1.0)
    assert np.isclose(v.z, 0.0)


def test_vector_from_angles_azimuth_only():
    # Azimuth is pi/2, dip is zero; should point along the x-axis (east).
    v = vector_from_angles(np.pi / 2, 0)
    assert np.isclose(v.x, 1.0)
    assert np.isclose(v.y, 0.0)
    assert np.isclose(v.z, 0.0)


def test_vector_from_angles_dip_only():
    # Dip is -pi/2, azimuth is zero; should point downwards along z-axis.
    v = vector_from_angles(0, -np.pi / 2)
    assert np.isclose(v.x, 0.0)
    assert np.isclose(v.y, 0.0)
    assert np.isclose(v.z, -1.0)


def test_vector_from_angles_positive_dip():
    # Dip is pi/4, azimuth is zero; should point upwards in the z direction.
    v = vector_from_angles(0, np.pi / 4)
    assert np.isclose(v.x, 0.0)
    assert np.isclose(v.y, np.cos(np.pi / 4))
    assert np.isclose(v.z, np.sin(np.pi / 4))


def test_vector_from_angles_azimuth_full_rotation():
    # Azimuth 2*pi is equivalent to zero azimuth
    v = vector_from_angles(2 * np.pi, 0)
    assert np.isclose(v.x, 0.0)
    assert np.isclose(v.y, 1.0)
    assert np.isclose(v.z, 0.0)


def test_vector_from_angles_dip_clamping():
    # Dip is clamped to [-pi/2, pi/2]
    # Test dip at 3*pi/2 (equivalent to -pi/2)
    v = vector_from_angles(0, 3 * np.pi / 2)
    assert np.isclose(v.x, 0.0)
    assert np.isclose(v.y, 0.0)
    assert np.isclose(v.z, -1.0)


def test_vector_from_angles_general_case():
    # General case with non-zero azimuth and dip
    azimuth = np.pi / 4
    dip = np.pi / 6
    v = vector_from_angles(azimuth, dip)
    expected_x = np.cos(dip) * np.sin(azimuth)
    expected_y = np.cos(dip) * np.cos(azimuth)
    expected_z = np.sin(dip)
    assert np.isclose(v.x, expected_x)
    assert np.isclose(v.y, expected_y)
    assert np.isclose(v.z, expected_z)


def test_calculate_azimuth_and_dip_north():
    # Vector pointing north (y-axis), should have azimuth 0 and dip 0
    v = Vector3(0, 1, 0)
    azimuth, dip = calculate_azimuth_and_dip(v)
    assert np.isclose(azimuth, 0.0)
    assert np.isclose(dip, 0.0)


def test_calculate_azimuth_and_dip_east():
    # Vector pointing east (x-axis), should have azimuth pi/2 and dip 0
    v = Vector3(1, 0, 0)
    azimuth, dip = calculate_azimuth_and_dip(v)
    assert np.isclose(azimuth, np.pi / 2)
    assert np.isclose(dip, 0.0)


def test_calculate_azimuth_and_dip_down():
    # Vector pointing straight down (z-axis), should have undefined azimuth and dip -pi/2
    v = Vector3(0, 0, -1)
    azimuth, dip = calculate_azimuth_and_dip(v)
    assert np.isclose(dip, -np.pi / 2)


def test_calculate_azimuth_and_dip_up():
    # Vector pointing straight up (z-axis), should have undefined azimuth and dip pi/2
    v = Vector3(0, 0, 1)
    azimuth, dip = calculate_azimuth_and_dip(v)
    assert np.isclose(dip, np.pi / 2)


def test_calculate_azimuth_and_dip_general_case():
    # General case with a vector pointing in a specific direction
    v = Vector3(1, 1, 1).normalize()
    azimuth, dip = calculate_azimuth_and_dip(v)

    expected_azimuth = np.arctan2(-v.x, -v.y) + np.pi
    expected_dip = np.arcsin(v.z)

    assert np.isclose(azimuth, expected_azimuth)
    assert np.isclose(dip, expected_dip)


def test_calculate_azimuth_and_dip_full_rotation():
    # Vector pointing along negative y-axis; azimuth should wrap around to pi
    v = Vector3(0, -1, 0)
    azimuth, dip = calculate_azimuth_and_dip(v)
    assert np.isclose(azimuth, np.pi)
    assert np.isclose(dip, 0.0)


def test_calculate_azimuth_and_dip_diagonal_vector():
    # Vector pointing at a 45-degree angle between x and y, with some z component
    v = Vector3(1, 1, np.sqrt(2)).normalize()
    azimuth, dip = calculate_azimuth_and_dip(v)

    expected_azimuth = np.arctan2(-v.x, -v.y) + np.pi
    expected_dip = np.arcsin(v.z)

    assert np.isclose(azimuth, expected_azimuth)
    assert np.isclose(dip, expected_dip)
