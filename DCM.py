"""
DCM Module  —  Direction Cosine Matrices
=========================================
Principal rotation matrices M1, M2, M3 and Euler angle extraction.

Convention used throughout:
    Mi(θ) is the DCM for a positive right-hand rotation of angle θ about axis i.
    Rows of the DCM are the new frame's basis vectors expressed in the old frame.
"""

import numpy as np


def M1(theta: float) -> np.ndarray:
    """Principal rotation matrix about axis 1 (x-axis) by angle θ.

    Args:
        theta: Rotation angle (rad).

    Returns:
        3×3 DCM for a rotation about the first axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,  0,  0],
                     [0,  c,  s],
                     [0, -s,  c]])


def M2(theta: float) -> np.ndarray:
    """Principal rotation matrix about axis 2 (y-axis) by angle θ.

    Args:
        theta: Rotation angle (rad).

    Returns:
        3×3 DCM for a rotation about the second axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  0, -s],
                     [ 0,  1,  0],
                     [ s,  0,  c]])


def M3(theta: float) -> np.ndarray:
    """Principal rotation matrix about axis 3 (z-axis) by angle θ.

    Args:
        theta: Rotation angle (rad).

    Returns:
        3×3 DCM for a rotation about the third axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s,  0],
                     [-s,  c,  0],
                     [ 0,  0,  1]])


def DCM313_2_Euler(dcm: np.ndarray) -> np.ndarray:
    """Extract 3-1-3 Euler angles from a DCM built with that sequence.

    Recovers  [Ω, i, θ]  from a DCM of the form  M3(θ) · M1(i) · M3(Ω).

    Note: uses arctan (not arctan2), so the result is limited to (−π/2, π/2).
    If you need angles outside that range, replace arctan with arctan2 and
    supply the correct numerator/denominator signs for each component.

    Args:
        dcm: 3×3 DCM produced by a 3-1-3 Euler sequence.

    Returns:
        Euler angles  [θ₁, θ₂, θ₃]  in radians.
    """
    theta1 = np.arctan(dcm[2, 0] / (-dcm[2, 1]))
    theta2 = np.arccos(dcm[2, 2])
    theta3 = np.arctan(dcm[0, 2] / dcm[1, 2])

    return np.array([theta1, theta2, theta3])