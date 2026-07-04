"""
Skew Module  —  Skew-Symmetric (Tilde) Operator
=================================================
Builds the skew-symmetric cross-product matrix [v×] from a 3-vector v.

For  v = [v₁, v₂, v₃]ᵀ  the skew-symmetric matrix is defined as:

         ⎡  0   -v₃   v₂ ⎤
    [v×] = ⎢  v₃   0   -v₁ ⎥
         ⎣ -v₂   v₁   0  ⎦

Key property:  [v×] w  =  v × w  for any vector w.
Used throughout the attitude dynamics equations wherever a cross product appears.
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix  [v×]  for a 3-vector v.

    Args:
        v: 3-element vector  [v₁, v₂, v₃].

    Returns:
        3×3 skew-symmetric matrix such that  [v×] w = v × w.
    """
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])