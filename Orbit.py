"""
Orbit Module
============
Orbital mechanics and attitude dynamics for the Mars spacecraft simulation.

This module defines all the physics: orbital positions, reference frame DCMs,
attitude error functions, the RK4 integrator, and the spacecraft mode selector.
It's designed to be imported by Visual_Simulation.py, which calls these
functions and reads the constants defined here.

Spacecraft modes
----------------
    1  Sun-pointing   — solar panel alignment when on the sunlit side
    2  Nadir          — Mars-pointing (Hill frame) for observation
    3  Communication  — GMO relay link when LMO and GMO are within 35°

Reference frames
----------------
    N  Inertial (Mars-centred)
    H  Hill (orbit-fixed, LMO)
    B  Body (spacecraft)
    S  Sun-pointing reference
    C  Communication reference (LMO → GMO)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np

import DCM
import EP
import MRP
import skew as sk


# ──────────────────────────────────────────────────────────────────────────────
# Conversion factors
# ──────────────────────────────────────────────────────────────────────────────

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Mars constants
# ──────────────────────────────────────────────────────────────────────────────

miu          = 42_828.3   # Mars gravitational parameter  (km³/s²)
mars_radius  = 3_396.19   # Mars mean radius              (km)


# ──────────────────────────────────────────────────────────────────────────────
# Geosynchronous Mars Orbit (GMO)
# ──────────────────────────────────────────────────────────────────────────────

r_GMO        = 20_424.2                    # orbital radius          (km)
thetadot_GMO = 0.0000709003                # mean motion             (rad/s)
theta_GMO    = 250 * DEG2RAD               # initial true anomaly    (rad)

# [Ω, i, θ₀]  — GMO is equatorial so inclination and RAAN are zero
orbit_angles_GMO_t0 = np.array([0.0, 0.0, theta_GMO])


# ──────────────────────────────────────────────────────────────────────────────
# Low Mars Orbit (LMO)
# ──────────────────────────────────────────────────────────────────────────────

r_LMO        = mars_radius + 400           # orbital radius at 400 km altitude (km)
thetadot_LMO = np.sqrt(miu / r_LMO ** 3)  # mean motion from vis-viva         (rad/s)
omega_LMO    = 20 * DEG2RAD                # RAAN Ω                            (rad)
i_LMO        = 30 * DEG2RAD               # inclination i                     (rad)
theta_LMO    = 60 * DEG2RAD               # initial true anomaly θ₀           (rad)

# [Ω, i, θ₀]
orbit_angles_LMO_t0 = np.array([omega_LMO, i_LMO, theta_LMO])


# ──────────────────────────────────────────────────────────────────────────────
# Spacecraft properties
# ──────────────────────────────────────────────────────────────────────────────

# Initial MRP attitude and body angular velocity  [rad/s]
mrp_BN_t0   = np.array([0.3, -0.4, 0.5])
omega_BN_t0 = np.array([1.0, 1.75, -2.20]) * DEG2RAD

# Principal inertia tensor  [kg·m²]
Inertia = np.array([[10.0, 0.0, 0.0],
                    [0.0,  5.0, 0.0],
                    [0.0,  0.0, 7.5]])

# Full initial state vector  [σ₁ σ₂ σ₃ ω₁ ω₂ ω₃]
state_x_t0 = np.concatenate([mrp_BN_t0, omega_BN_t0])


# ──────────────────────────────────────────────────────────────────────────────
# Fixed reference frame DCMs (constant, computed once)
# ──────────────────────────────────────────────────────────────────────────────

# Sun-pointing reference: b1 = −N1, b2 = N3, b3 = N2
DCM_Ns = np.array([[-1.0, 0.0, 0.0],
                   [ 0.0, 0.0, 1.0],
                   [ 0.0, 1.0, 0.0]])

# Nadir reference (Hill → Nadir rotation): b1 = −r̂, b2 = ŷ_Hill, b3 = −ẑ_Hill
DCM_nadh = np.array([[-1.0, 0.0,  0.0],
                     [ 0.0, 1.0,  0.0],
                     [ 0.0, 0.0, -1.0]])


# ──────────────────────────────────────────────────────────────────────────────
# Controller gains   (PD control in MRP space)
#
# Tuned for a settling time of ~120 s with Imax = 10 kg·m²
# ──────────────────────────────────────────────────────────────────────────────

T_decay     = 120   # target settling time    (s)
Imax        = 10    # largest principal inertia
Imin        = 5     # smallest principal inertia

P_controlsun = 2 * Imax / T_decay
K_controlsun = (4 * Imax ** 2) / (T_decay ** 2 * Imin)


# ──────────────────────────────────────────────────────────────────────────────
# Orbital mechanics functions
# ──────────────────────────────────────────────────────────────────────────────

def inertial_orbit_pos_vel(radius: float, gravity_constant: float,
                            euler_angles: np.ndarray):
    """Compute the inertial position and velocity for a circular orbit.

    Rotates a radial position vector from the orbital plane into the inertial
    N-frame using the 3-1-3 Euler angle sequence [Ω, i, θ].

    Args:
        radius:           Orbital radius (km).
        gravity_constant: Gravitational parameter μ (km³/s²).
        euler_angles:     [Ω, i, θ]  right ascension, inclination, true anomaly (rad).

    Returns:
        DCM_NO:       3×3 DCM rotating from orbital to inertial frame.
        inertial_pos: Position vector in the N-frame (km).
        inertial_vel: Velocity vector in the N-frame (km/s).
    """
    # Build the orbital → inertial DCM  (3-1-3 sequence, transposed)
    DCM_ON = np.array(DCM.M3(euler_angles[2]) @ DCM.M1(euler_angles[1]) @ DCM.M3(euler_angles[0]))
    DCM_NO = DCM_ON.T

    inertial_pos = DCM_NO @ np.array([radius, 0.0, 0.0])

    orbital_speed = np.sqrt(gravity_constant / radius ** 3)
    inertial_vel  = DCM_NO @ np.array([0.0, orbital_speed, 0.0])

    return DCM_NO, inertial_pos, inertial_vel


def Hill_dcm(euler_angles: np.ndarray, time: float) -> np.ndarray:
    """Return the inertial-to-Hill DCM (DCM_NH) for the LMO at a given time.

    Propagates the true anomaly forward from the initial angle using the LMO
    mean motion, then builds and transposes the 3-1-3 DCM.

    Args:
        euler_angles: Initial [Ω, i, θ₀] for the LMO (rad).
        time:         Elapsed time from epoch (s).

    Returns:
        DCM_NH: 3×3 DCM mapping from Hill to inertial frame.
    """
    angles    = euler_angles.copy()                    # never mutate the original
    angles[2] = euler_angles[2] + thetadot_LMO * time

    DCM_HN = np.array(DCM.M3(angles[2]) @ DCM.M1(angles[1]) @ DCM.M3(angles[0]))
    return DCM_HN.T   # Hill → Inertial


def gmo_orbit_dcm(euler_angles: np.ndarray, time: float) -> np.ndarray:
    """Return the inertial-to-GMO DCM (DCM_NH) at a given time.

    Same logic as Hill_dcm but uses the GMO mean motion.

    Args:
        euler_angles: Initial [Ω, i, θ₀] for the GMO (rad).
        time:         Elapsed time from epoch (s).

    Returns:
        DCM_NH: 3×3 DCM mapping from GMO orbit frame to inertial frame.
    """
    angles    = euler_angles.copy()
    angles[2] = euler_angles[2] + thetadot_GMO * time

    DCM_HN = np.array(DCM.M3(angles[2]) @ DCM.M1(angles[1]) @ DCM.M3(angles[0]))
    return DCM_HN.T   # GMO orbit → Inertial


# ──────────────────────────────────────────────────────────────────────────────
# Communication reference frame
# ──────────────────────────────────────────────────────────────────────────────

def get_DCM_cN(time: float) -> np.ndarray:
    """Build the communication reference frame DCM (C relative to N) at a given time.

    The frame is defined so that:
        r1_cN  points from LMO toward GMO  (−r̂_LMO→GMO)
        r2_cN  is perpendicular to r1 in the plane containing N3
        r3_cN  completes the right-hand triad

    Args:
        time: Elapsed time from epoch (s).

    Returns:
        DCM_cN: 3×3 DCM of the communication frame relative to inertial N.
    """
    # LMO position at time t
    angles_lmo      = orbit_angles_LMO_t0.copy()
    angles_lmo[2]  += thetadot_LMO * time
    _, pos_lmo, _   = inertial_orbit_pos_vel(r_LMO, miu, angles_lmo)

    # GMO position at time t
    angles_gmo      = orbit_angles_GMO_t0.copy()
    angles_gmo[2]  += thetadot_GMO * time
    _, pos_gmo, _   = inertial_orbit_pos_vel(r_GMO, miu, angles_gmo)

    # Build orthonormal triad pointing LMO → GMO
    dr_cN  = pos_gmo - pos_lmo
    r1_cN  = -dr_cN / np.linalg.norm(dr_cN)

    n3     = np.array([0.0, 0.0, 1.0])
    r2_cN  = np.cross(dr_cN, n3) / np.linalg.norm(np.cross(dr_cN, n3))
    r3_cN  = np.cross(r1_cN, r2_cN)

    return np.array([r1_cN, r2_cN, r3_cN])


def get_omega_cN(time: float) -> np.ndarray:
    """Estimate the angular velocity of the communication frame via finite difference.

    Uses a 1-second forward difference on DCM_cN to approximate ω_cN expressed
    in the inertial N-frame.

    Args:
        time: Elapsed time from epoch (s).

    Returns:
        omega_cN: Angular velocity vector [ω₁, ω₂, ω₃] (rad/s).
    """
    dt = 1.0   # 1-second finite-difference step

    DCM_current = get_DCM_cN(time)
    DCM_next    = get_DCM_cN(time + dt)

    # ω× = −C^T · Ċ  (skew-symmetric angular velocity tensor)
    omega_skew  = -DCM_current.T @ (DCM_next - DCM_current)

    return np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])


# ──────────────────────────────────────────────────────────────────────────────
# Attitude error
# ──────────────────────────────────────────────────────────────────────────────

def attitude_error(time: float, MRP_BN: np.ndarray,
                   omega_BN: np.ndarray, mode: int):
    """Compute MRP and angular-rate errors relative to the active reference frame.

    Args:
        time:     Elapsed time from epoch (s).
        MRP_BN:   Current body attitude as MRP σ_B/N.
        omega_BN: Current body angular velocity  ω_B/N (rad/s).
        mode:     Spacecraft mode  (1=sun, 2=nadir, 3=communication).

    Returns:
        MRP_error:   Attitude error as MRP σ_B/R.
        omega_error: Angular rate error ω_B/N − ω_R/N (rad/s).
    """
    DCM_BN = MRP.MRP2DCM(MRP_BN)

    # ── Mode 1: Sun pointing ──────────────────────────────────────────────────
    if mode == 1:
        DCM_BS    = DCM_BN @ DCM_Ns
        MRP_error = MRP.EP2MRP(EP.DCM2EP(DCM_BS))

        # Sun reference has zero angular velocity in the inertial frame
        omega_error = omega_BN - DCM_BS @ np.array([0.0, 0.0, 0.0])

    # ── Mode 2: Nadir pointing ────────────────────────────────────────────────
    elif mode == 2:
        # Map from Hill frame → inertial → nadir
        DCM_Nh   = Hill_dcm(orbit_angles_LMO_t0, time)
        DCM_nadN = DCM_nadh @ DCM_Nh.T    # nadir-to-Hill composed with Hill-to-inertial^T
        DCM_Nnad = DCM_nadN.T             # nadir → inertial

        DCM_Bnad  = DCM_BN @ DCM_Nnad
        MRP_error = MRP.EP2MRP(EP.DCM2EP(DCM_Bnad))

        # Nadir frame rotates at the LMO mean motion about the Hill −z axis
        omega_Nnad  = np.array([0.0, 0.0, -thetadot_LMO])
        omega_error = omega_BN - DCM_Bnad @ omega_Nnad

    # ── Mode 3: Communication pointing ───────────────────────────────────────
    elif mode == 3:
        DCM_cN    = get_DCM_cN(time)
        DCM_Nc    = DCM_cN.T              # communication frame → inertial

        DCM_Bc    = DCM_BN @ DCM_Nc
        MRP_error = MRP.EP2MRP(EP.DCM2EP(DCM_Bc))

        omega_cN    = get_omega_cN(time)
        omega_error = omega_BN - DCM_Bc @ omega_cN

    else:
        raise ValueError(f'Unknown spacecraft mode: {mode}  (expected 1, 2, or 3)')

    return MRP_error, omega_error


# ──────────────────────────────────────────────────────────────────────────────
# Spacecraft mode selector
# ──────────────────────────────────────────────────────────────────────────────

def get_spacecraft_frame(time: float) -> int:
    """Return the active spacecraft mode at a given time.

    Decision logic:
        - If the LMO is on the sunlit side (N2 > 0) → Sun mode (1)
        - If in shadow AND within 35° of GMO        → Communication mode (3)
        - Otherwise                                 → Nadir mode (2)

    Args:
        time: Elapsed time from epoch (s).

    Returns:
        Spacecraft mode integer: 1, 2, or 3.
    """
    # Current inertial positions
    LMO_pos = Hill_dcm(orbit_angles_LMO_t0, time) @ np.array([r_LMO, 0.0, 0.0])
    GMO_pos = gmo_orbit_dcm(orbit_angles_GMO_t0, time) @ np.array([r_GMO, 0.0, 0.0])

    on_sunlit_side = LMO_pos[1] > 0   # N2 component positive → facing the Sun

    if on_sunlit_side:
        return 1   # Sun-pointing mode

    # In shadow — check angular separation to GMO for comm link
    cos_angle = (np.dot(LMO_pos, GMO_pos)
                 / (np.linalg.norm(LMO_pos) * np.linalg.norm(GMO_pos)))
    angle     = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if angle < 35 * DEG2RAD:
        return 3   # Communication mode (within 35° of GMO)

    return 2       # Nadir mode


# ──────────────────────────────────────────────────────────────────────────────
# Integrator and equations of motion
# ──────────────────────────────────────────────────────────────────────────────

def RK4(x: np.ndarray, u: np.ndarray, t: float,
        dt: float, derivatives) -> np.ndarray:
    """Classic 4th-order Runge–Kutta integrator (single step).

    Args:
        x:           State vector at time t.
        u:           Control input vector (held constant over the step).
        t:           Current time (s).
        dt:          Step size (s).
        derivatives: Callable f(x, u, t) → ẋ.

    Returns:
        New state vector at time t + dt.
    """
    k1 = derivatives(x,                u, t)
    k2 = derivatives(x + dt / 2 * k1, u, t + dt / 2)
    k3 = derivatives(x + dt / 2 * k2, u, t + dt / 2)
    k4 = derivatives(x + dt * k3,     u, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def attitude_derivatives(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
    """Equations of motion for MRP attitude + Euler's rotational equations.

    State:  x = [σ₁ σ₂ σ₃ ω₁ ω₂ ω₃]
    Input:  u = control torque vector (N·m)

    MRP kinematics:
        σ̇ = ¼ B(σ) · ω
        B(σ) = (1 − σᵀσ)I + 2[σ×] + 2σσᵀ

    Euler's equation:
        ω̇ = I⁻¹ (−ω× I ω + u)

    Args:
        x: State vector [σ, ω].
        u: Control torque (N·m).
        t: Current time — unused here, kept for integrator compatibility.

    Returns:
        State derivative [σ̇, ω̇].
    """
    sigma = x[0:3]
    omega = x[3:6]

    sigma_sq = np.dot(sigma, sigma)
    B_matrix = ((1 - sigma_sq) * np.eye(3)
                + 2 * sk.skew(sigma)
                + 2 * np.outer(sigma, sigma))

    sigma_dot = 0.25 * B_matrix @ omega
    omega_dot = np.linalg.inv(Inertia) @ (-sk.skew(omega) @ Inertia @ omega + u)

    return np.concatenate([sigma_dot, omega_dot])


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone simulation  (only runs when this file is executed directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Open-loop attitude simulation over 6500 seconds.
    Results are stored in state_x (attitudes) and state_u (control torques).
    Run this file directly to verify the dynamics before launching the 3D visualisation.
    """

    simulation_time = 6_500   # seconds
    dt              = 1       # integration step (s)
    t               = np.arange(0, simulation_time, dt)
    N               = len(t)

    state_x = np.zeros((N, 6))    # attitude history  [σ, ω]
    state_u = np.zeros((N - 1, 3))  # control torque history

    state_x[0, :] = state_x_t0

    for i in range(N - 1):

        craft_mode = get_spacecraft_frame(i)

        mrp_error, omega_error = attitude_error(
            i, state_x[i, 0:3], state_x[i, 3:6], craft_mode)

        u = -K_controlsun * mrp_error - P_controlsun * omega_error
        state_u[i, :] = u

        new_state = RK4(state_x[i, :], u, i, dt, attitude_derivatives)

        sigma = new_state[0:3]
        omega = new_state[3:6]

        # Switch to MRP shadow set if |σ| > 1 to avoid singularity
        if np.linalg.norm(sigma) > 1:
            sigma = -sigma / np.linalg.norm(sigma) ** 2

        state_x[i + 1, 0:3] = sigma
        state_x[i + 1, 3:6] = omega

    print(f'Simulation complete — {N} steps over {simulation_time} s')
    print(f'Final MRP:   {state_x[-1, 0:3]}')
    print(f'Final omega: {state_x[-1, 3:6]} rad/s')