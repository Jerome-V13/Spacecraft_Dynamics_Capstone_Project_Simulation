"""
EP Module  —  Euler Parameters (Quaternions)
============================================
Conversions between Direction Cosine Matrices (DCM) and Euler Parameters (EP),
also known as unit quaternions  q = [b0, b1, b2, b3]  where b0 is the scalar part.

Convention used throughout:
    q = [b0, b1, b2, b3]  with  b0 = cos(Φ/2)  and  b1..b3 = sin(Φ/2) · ê

All quaternions are returned with b0 ≥ 0 (shortest-rotation convention).
"""

import numpy as np


def EP2DCM(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a Direction Cosine Matrix (DCM).

    Uses the standard quaternion-to-DCM formula. The input must be a unit
    quaternion  [b0, b1, b2, b3]  with  b0² + b1² + b2² + b3² = 1.

    Args:
        q: Unit quaternion as a 4-element array  [b0, b1, b2, b3].

    Returns:
        3×3 DCM (orthogonal rotation matrix).
    """
    b0, b1, b2, b3 = q

    return np.array([
        [b0**2 + b1**2 - b2**2 - b3**2,  2*(b1*b2 + b0*b3),              2*(b1*b3 - b0*b2)            ],
        [2*(b1*b2 - b0*b3),               b0**2 - b1**2 + b2**2 - b3**2,  2*(b2*b3 + b0*b1)            ],
        [2*(b1*b3 + b0*b2),               2*(b2*b3 - b0*b1),              b0**2 - b1**2 - b2**2 + b3**2],
    ])


def DCM2EP(DCM: np.ndarray) -> np.ndarray:
    """Convert a DCM to a unit quaternion using Shepperd's method.

    Shepperd's method avoids division-by-zero by always dividing through the
    *largest* of the four b² values. That way you never divide by something
    close to zero, no matter what the rotation angle is.

    The sign of the result is chosen so that b0 ≥ 0 (shortest rotation).

    Args:
        DCM: 3×3 orthogonal rotation matrix.

    Returns:
        Unit quaternion  [b0, b1, b2, b3]  with b0 ≥ 0.
    """
    # Step 1 — compute all four b² candidates from the DCM trace
    b2_0 = 0.25 * (1 + np.trace(DCM))
    b2_1 = 0.25 * (1 + 2*DCM[0, 0] - np.trace(DCM))
    b2_2 = 0.25 * (1 + 2*DCM[1, 1] - np.trace(DCM))
    b2_3 = 0.25 * (1 + 2*DCM[2, 2] - np.trace(DCM))

    # Step 2 — find which one is largest and use it as the "safe" divisor
    largest = np.argmax([b2_0, b2_1, b2_2, b2_3])

    if largest == 0:
        # b0 is largest — take positive root (automatically shortest rotation)
        b0 = np.sqrt(b2_0)
        b1 = (DCM[1, 2] - DCM[2, 1]) / (4*b0)
        b2 = (DCM[2, 0] - DCM[0, 2]) / (4*b0)
        b3 = (DCM[0, 1] - DCM[1, 0]) / (4*b0)

    elif largest == 1:
        # b1 is largest — sign of b0 unknown, so check and fix if needed
        b1 = np.sqrt(b2_1)
        b0 = (DCM[1, 2] - DCM[2, 1]) / (4*b1)
        b2 = (DCM[0, 1] + DCM[1, 0]) / (4*b1)
        b3 = (DCM[2, 0] + DCM[0, 2]) / (4*b1)

    elif largest == 2:
        # b2 is largest — same sign check
        b2 = np.sqrt(b2_2)
        b0 = (DCM[2, 0] - DCM[0, 2]) / (4*b2)
        b1 = (DCM[0, 1] + DCM[1, 0]) / (4*b2)
        b3 = (DCM[1, 2] + DCM[2, 1]) / (4*b2)

    else:
        # b3 is largest — same sign check
        b3 = np.sqrt(b2_3)
        b0 = (DCM[0, 1] - DCM[1, 0]) / (4*b3)
        b1 = (DCM[2, 0] + DCM[0, 2]) / (4*b3)
        b2 = (DCM[1, 2] + DCM[2, 1]) / (4*b3)

    q = np.array([b0, b1, b2, b3])

    # Enforce shortest-rotation convention: flip all signs if b0 ended up negative
    if q[0] < 0:
        q = -q

    return q