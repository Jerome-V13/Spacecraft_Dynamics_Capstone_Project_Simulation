"""
MRP Module  —  Modified Rodrigues Parameters
=============================================
Conversions between Modified Rodrigues Parameters (MRP) and other attitude
representations, plus the shadow-set switch.

MRP are defined as:
    σ = q_vec / (1 + q₀)

where  q = [q₀, q₁, q₂, q₃]  is the unit quaternion (b0 = scalar part).

The shadow set  σ* = −σ / |σ|²  represents the same physical rotation but
avoids the MRP singularity at |σ| = 1 (i.e. 360° rotations).  Switching to
the shadow set whenever |σ| > 1 keeps the parameters well-behaved at all times.
"""

import numpy as np
import skew as sk


def EP2MRP(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to Modified Rodrigues Parameters.

    Formula:  σ = q_vec / (1 + q₀)

    Args:
        q: Unit quaternion  [b0, b1, b2, b3]  with b0 ≥ 0.

    Returns:
        MRP vector  σ = [σ₁, σ₂, σ₃].
    """
    denom = 1 + q[0]   # store once — used three times
    return np.array([q[1] / denom,
                     q[2] / denom,
                     q[3] / denom])


def MRP2EP(sigma: np.ndarray) -> np.ndarray:
    """Convert Modified Rodrigues Parameters to a unit quaternion.

    Formula:
        q₀      = (1 − |σ|²) / (1 + |σ|²)
        q_vec   = 2σ          / (1 + |σ|²)

    Args:
        sigma: MRP vector  [σ₁, σ₂, σ₃].

    Returns:
        Unit quaternion  [b0, b1, b2, b3]  with b0 ≥ 0.
    """
    sigma_sq = sigma @ sigma   # |σ|²  (scalar)
    denom    = 1 + sigma_sq

    b0 = (1 - sigma_sq) / denom
    b1 = (2 * sigma[0]) / denom
    b2 = (2 * sigma[1]) / denom
    b3 = (2 * sigma[2]) / denom

    return np.array([b0, b1, b2, b3])


def MRP_shadow(sigma: np.ndarray) -> np.ndarray:
    """Return the shadow-set MRP for the same rotation.

    Formula:  σ* = −σ / |σ|²

    The shadow set represents the identical physical rotation as σ but through
    the complementary (> 180°) path.  Switching to σ* when |σ| > 1 keeps the
    MRP magnitude below 1 and avoids the singularity at |σ| = 1.

    Args:
        sigma: MRP vector  [σ₁, σ₂, σ₃].

    Returns:
        Shadow MRP vector  σ* = [σ₁*, σ₂*, σ₃*].
    """
    sigma_sq = sigma @ sigma   # |σ|²
    return -sigma / sigma_sq


def MRP2DCM(sigma: np.ndarray) -> np.ndarray:
    """Convert Modified Rodrigues Parameters to a Direction Cosine Matrix.

    Uses the Cayley map (Rodrigues formula for MRP):
        [BN] = I  +  (8[σ×]² − 4(1 − |σ|²)[σ×]) / (1 + |σ|²)²

    Args:
        sigma: MRP vector  [σ₁, σ₂, σ₃].

    Returns:
        3×3 DCM (orthogonal rotation matrix).
    """
    sigma_sq  = sigma @ sigma
    skew_s    = sk.skew(sigma)
    denom     = (1 + sigma_sq) ** 2

    numerator = (8 * skew_s @ skew_s) - (4 * (1 - sigma_sq) * skew_s)

    return np.eye(3) + numerator / denom