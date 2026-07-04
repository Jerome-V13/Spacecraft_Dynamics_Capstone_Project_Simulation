"""
Visual Simulation Module
========================
Real-time 3D visualisation of Mars orbital mechanics using the Panda3D engine.

This module is the top-level entry point for the simulation. It owns the Panda3D
scene graph, all rendering logic, the camera controller, and the HUD. Orbital
mechanics and attitude dynamics are delegated to the companion modules:

    Orbit.py  —  orbital angles, inertial-frame position/velocity, RK4 integrator,
                 attitude-error functions, and spacecraft-mode selector.
    MRP.py    —  Modified Rodrigues Parameter ↔ Euler Parameter (quaternion) conversion.
    DCM.py    —  Direction Cosine Matrix primitives (M1, M3 rotation matrices).

Scene graph layout
------------------
render
├── mars                   UV sphere with equirectangular texture
├── sun                    Parent node for PointLight + visible disc
├── stars                  Background point-cloud (rendered before geometry)
├── [N1 / N2 / N3 axes]    Inertial reference frame arrows
├── gmo_ring               GMO orbit ring (LineSegs, static)
├── lmo_ring               LMO orbit ring (LineSegs, rebuilt each frame)
├── comm_cone_node         35° communication cone (LineSegs, rebuilt each frame)
├── gmo_craft              GMO satellite sphere + billboard label
└── spacecraft             Parent node — receives attitude quaternion every frame
    ├── spacecraft_model   GLB / OBJ model (or fallback sphere)
    └── [b1 / b2 / b3]     Body-frame axis triad

Controls
--------
F           Toggle camera focus  (Mars ↔ LMO spacecraft)
↑ / ↓       Double / halve simulation speed
Scroll      Zoom in / out
Left drag   Orbit camera around focus point
C           Toggle 35° communication cone
P           Cycle telemetry plot  (MRP error → ω → control torques → hidden)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math
import os
import random

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, Geom, GeomNode, GeomPoints, GeomTriangles,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    LineSegs, NodePath, Point3, PointLight, Quat,
    TextNode, Vec4, WindowProperties,
)

try:
    from panda3d import gltf  # noqa: F401  — imported for availability check only
    print("✓ panda3d-gltf available")
except ImportError:
    print("✗ panda3d-gltf not available — GLB models will not load")

# ── Local modules ──────────────────────────────────────────────────────────────
import DCM as dcm_mod
import MRP as mrp_mod
import Orbit as orbit_mod


# ──────────────────────────────────────────────────────────────────────────────
# Physical constants  (all distances in km, angles in rad)
# ──────────────────────────────────────────────────────────────────────────────

MARS_RADIUS_KM = 3_396.19                               # Volumetric mean radius
R_LMO_KM       = MARS_RADIUS_KM + 400                  # Low Mars Orbit radius
R_GMO_KM       = 20_424.2                              # Geosynchronous Mars Orbit radius
THETADOT_LMO   = math.sqrt(42_828.3 / R_LMO_KM ** 3)  # LMO mean motion (rad/s)
THETADOT_GMO   = 0.0000709003                           # GMO mean motion (rad/s)

# Panda3D scene scale factor: Mars is rendered as a sphere of radius 100 units,
# so every distance in km is multiplied by this constant.
SCENE_SCALE = 100.0 / MARS_RADIUS_KM


# ──────────────────────────────────────────────────────────────────────────────
# Plot colour palettes  (RGBA, 0–1 range)
# Defined at module level so they are shared between _create_lmo_plot,
# _clear_plot_elements, and _draw_plot without repeating the literals.
# ──────────────────────────────────────────────────────────────────────────────

MRP_COLORS     = {'e1': (1, 0, 0, 1),   'e2': (0, 1, 0, 1),   'e3': (0, 0, 1, 1)}
OMEGA_COLORS   = {'w1': (1, 0.5, 0, 1), 'w2': (0.5, 1, 0, 1), 'w3': (0, 0.5, 1, 1)}
CONTROL_COLORS = {'u1': (1, 1, 0, 1),   'u2': (0, 1, 1, 1),   'u3': (1, 0, 1, 1)}

MRP_LABELS     = {'e1': 'σ₁', 'e2': 'σ₂', 'e3': 'σ₃'}
OMEGA_LABELS   = {'w1': 'ω₁', 'w2': 'ω₂', 'w3': 'ω₃'}
CONTROL_LABELS = {'u1': 'u₁', 'u2': 'u₂', 'u3': 'u₃'}


# ──────────────────────────────────────────────────────────────────────────────
# Scene-geometry factory functions
# These are plain functions (not methods) so they can be unit-tested and reused
# without instantiating the full simulation.
# ──────────────────────────────────────────────────────────────────────────────

def make_starfield(num_stars: int = 4_000, radius: float = 8_000) -> GeomNode:
    """Return a GeomNode of randomly distributed point-stars on a large sphere.

    Stars are placed far beyond all scene geometry so they appear at infinity.
    Each star receives a random brightness and a subtle warm/cool colour tint to
    approximate the spectral variation of real stars.

    Args:
        num_stars: Number of star points to generate.
        radius:    Radius of the containing sphere (scene units).
    """
    fmt    = GeomVertexFormat.getV3c4()
    vdata  = GeomVertexData('stars', fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    color  = GeomVertexWriter(vdata, 'color')
    points = GeomPoints(Geom.UHStatic)

    for i in range(num_stars):
        # Uniform distribution on a sphere (rejection-free method)
        u     = random.uniform(-1, 1)
        theta = random.uniform(0, 2 * math.pi)
        r     = math.sqrt(1 - u * u)
        vertex.addData3(radius * r * math.cos(theta),
                        radius * r * math.sin(theta),
                        radius * u)

        brightness = random.uniform(0.5, 1.0)
        tint       = random.uniform(0.9, 1.0)   # subtle warm ↔ cool variation
        color.addData4(brightness, brightness * tint, brightness, 1)

        points.addVertex(i)
        points.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(points)
    node = GeomNode('starfield')
    node.addGeom(geom)
    return node


def make_uv_sphere(radius: float = 1.0, slices: int = 64, stacks: int = 64) -> GeomNode:
    """Build a UV-sphere mesh with per-vertex normals and texture coordinates.

    Vertices are laid out in latitude (stacks) × longitude (slices) order.
    Texture coordinates map a standard equirectangular image correctly around
    the sphere, with the seam at u=0/1 and the poles at v=0 and v=1.

    Args:
        radius: Sphere radius in scene units.
        slices: Longitudinal subdivisions (more = smoother silhouette).
        stacks: Latitudinal subdivisions.
    """
    fmt      = GeomVertexFormat.getV3n3t2()
    vdata    = GeomVertexData('sphere', fmt, Geom.UHStatic)
    vertex   = GeomVertexWriter(vdata, 'vertex')
    normal   = GeomVertexWriter(vdata, 'normal')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    for i in range(stacks + 1):
        for j in range(slices + 1):
            theta = i * math.pi / stacks        # polar angle  0 → π
            phi   = j * 2 * math.pi / slices    # azimuth      0 → 2π

            nx = math.sin(theta) * math.cos(phi)
            ny = math.sin(theta) * math.sin(phi)
            nz = math.cos(theta)

            vertex.addData3(radius * nx, radius * ny, radius * nz)
            normal.addData3(nx, ny, nz)
            texcoord.addData2(j / slices, i / stacks)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(stacks):
        for j in range(slices):
            v0 = i * (slices + 1) + j
            tris.addVertices(v0,     v0 + (slices + 1), v0 + 1)
            tris.addVertices(v0 + 1, v0 + (slices + 1), v0 + (slices + 2))

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('sphere')
    node.addGeom(geom)
    return node


def make_orbit_ring(omega: float, inc: float, radius: float,
                    color: tuple, slices: int = 200) -> NodePath:
    """Draw a full Keplerian orbit ring as a LineSegs NodePath.

    The orbit is defined by its right-ascension of ascending node (Ω) and
    inclination (i).  The ring is sampled uniformly in true anomaly.

    Args:
        omega:  RAAN Ω in radians.
        inc:    Inclination i in radians.
        radius: Orbital radius in km.
        color:  RGBA colour tuple.
        slices: Number of straight-line segments used to approximate the circle.
    """
    lines = LineSegs()
    lines.setColor(*color)
    lines.setThickness(1.5)

    first = True
    for i in range(slices + 1):
        theta = 2 * math.pi * i / slices
        pos   = orbit_point(theta, omega, inc, radius)
        if first:
            lines.moveTo(*pos)
            first = False
        else:
            lines.drawTo(*pos)

    return NodePath(lines.create())


def make_comm_cone_lines(height: float, half_angle_deg: float,
                         segments: int = 32,
                         color: tuple = (0.6, 0.6, 0.6, 1)) -> NodePath:
    """Build a wireframe cone that visualises the communication window.

    The cone apex sits at the **local origin** and its axis points along local +Y.
    This matches Panda3D's convention so that ``NodePath.lookAt(gmo_pos)`` aims
    the axis toward GMO correctly.

    The 35° half-angle criterion is measured from Mars' centre: any point inside
    this cone satisfies the LMO–GMO angular communication requirement.

    Args:
        height:         Axial length from apex to base (scene units).
        half_angle_deg: Half-angle of the cone (degrees).
        segments:       Number of segments in the base circle.
        color:          RGBA colour tuple.
    """
    base_radius = height * math.tan(math.radians(half_angle_deg))

    lines = LineSegs()
    lines.setColor(*color)
    lines.setThickness(1.5)

    # Sample the base circle (at Y = height in local space)
    pts = [
        (base_radius * math.cos(2 * math.pi * i / segments),
         height,
         base_radius * math.sin(2 * math.pi * i / segments))
        for i in range(segments + 1)
    ]

    # Draw the base circle
    first = True
    for p in pts:
        if first:
            lines.moveTo(*p)
            first = False
        else:
            lines.drawTo(*p)

    # Draw radial spokes from apex to base
    step = max(1, segments // 12)
    for i in range(0, segments, step):
        lines.moveTo(0, 0, 0)
        lines.drawTo(*pts[i])

    return NodePath(lines.create())


def make_axes_node(length: float = 20.0, thickness: float = 2.5) -> NodePath:
    """Create a body-frame axis triad (b1=red, b2=green, b3=blue) with labels.

    The returned NodePath should be parented directly to a spacecraft node so
    that the axes rotate with the spacecraft's attitude quaternion.  Labels use
    billboard rendering so they always face the camera.

    Args:
        length:    Length of each axis line in scene units.
        thickness: Rendering thickness of the axis lines.
    """
    parent = NodePath('axes_node')
    lines  = LineSegs()
    lines.setThickness(thickness)

    # (tip_position, label_position, RGBA, label_text)
    axis_defs = [
        ((length, 0, 0),      (length + 3, 0, 0),      (1, 0, 0, 1), 'b1'),
        ((0, length, 0),      (0, length + 3, 0),      (0, 1, 0, 1), 'b2'),
        ((0, 0, length),      (0, 0, length + 3),      (0, 0, 1, 1), 'b3'),
    ]

    for tip_pos, label_pos, rgba, name in axis_defs:
        lines.setColor(*rgba)
        lines.moveTo(0, 0, 0)
        lines.drawTo(*tip_pos)

    line_node = parent.attachNewNode(lines.create())
    line_node.setLightOff()

    # Billboard text labels — always face the camera, depth-tested off
    for _, label_pos, rgba, name in axis_defs:
        text_node = TextNode(f'{name}_label')
        text_node.setText(name)
        text_node.setTextColor(*rgba)
        text_node.setAlign(TextNode.ACenter)

        label_np = parent.attachNewNode(text_node)
        label_np.setPos(*label_pos)
        label_np.setScale(3.0)
        label_np.setBillboardPointEye()
        label_np.setDepthTest(False)
        label_np.setDepthWrite(False)
        label_np.setLightOff()

    return parent


# ──────────────────────────────────────────────────────────────────────────────
# Orbital / mathematical helpers
# ──────────────────────────────────────────────────────────────────────────────

def vec_angle_deg(v1, v2) -> float:
    """Return the angle in degrees between two 3-vectors (always 0–180°).

    Safe against zero-length inputs — returns 180° in that degenerate case.
    """
    a  = np.asarray(v1, dtype=float)
    b  = np.asarray(v2, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    cos_ang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))


def orbit_point(theta: float, omega: float, inc: float, radius: float) -> list:
    """Convert a true anomaly to an inertial Cartesian position in scene units.

    Applies R3(−Ω) · R1(−i) to rotate the position from the orbital plane into
    the inertial N-frame, then scales from km to Panda3D scene units.

    Args:
        theta:  True anomaly (rad).
        omega:  RAAN Ω (rad).
        inc:    Inclination i (rad).
        radius: Orbital radius (km).

    Returns:
        [x, y, z] list in Panda3D scene units.
    """
    pos_O = np.array([radius * math.cos(theta),
                      radius * math.sin(theta),
                      0.0])
    DCM   = np.array(dcm_mod.M3(-omega)) @ np.array(dcm_mod.M1(-inc))
    pos_N = DCM @ pos_O
    return [p * SCENE_SCALE for p in pos_N]


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation class
# ──────────────────────────────────────────────────────────────────────────────

class MarsSimulation(ShowBase):
    """Panda3D application that visualises Mars orbital mechanics in real time.

    Initialisation is split across several private _setup_* methods so that
    each responsibility is easy to locate and modify independently.
    """

    # ── Entry point ───────────────────────────────────────────────────────────

    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0, 0, 0, 1)   # deep space black
        self.render.setShaderAuto()            # enable per-pixel lighting
        self.disableMouse()                    # use custom camera controller

        self._setup_window()
        self._setup_scene()
        self._setup_spacecraft()
        self._setup_simulation_state()
        self._setup_camera_state()
        self._setup_ui()
        self._setup_keybindings()
        self._setup_tasks()

    # ── Window ────────────────────────────────────────────────────────────────

    def _setup_window(self):
        """Set window size and hide the OS cursor for an immersive look."""
        props = WindowProperties()
        props.setSize(1280, 720)
        #props.setCursorHidden(True)  # Uncomment to hide the OS cursor for immersive look
        self.win.requestProperties(props)

    # ── 3-D scene (Mars, lighting, orbits, GMO, comm cone) ────────────────────

    def _setup_scene(self):
        """Build Mars, lighting, starfield, orbit rings, comm cone, and GMO craft."""

        # ── Mars globe ────────────────────────────────────────────────────────
        mars_node = make_uv_sphere(radius=100.0, slices=64, stacks=64)
        self.mars = self.render.attachNewNode(mars_node)
        self.mars.setTexture(self.loader.loadTexture('Mars.jpg'), 1)

        # ── Sun (PointLight along N2 + visible yellow disc) ───────────────────
        plight = PointLight('plight')
        plight.setColor(Vec4(1, 1, 1, 1))
        self.sun = self.render.attachNewNode('sun')
        plnp = self.sun.attachNewNode(plight)
        self.sun.setPos(0, 10_000, 0)
        self.render.setLight(plnp)

        # Visible sun disc — lighting disabled so it always appears fully bright
        sun_obj = make_uv_sphere(radius=40.0, slices=32, stacks=32)
        self.sun_model = self.sun.attachNewNode(sun_obj)
        self.sun_model.setColor(1, 0.9, 0.0, 1)
        self.sun_model.setLightOff()

        # ── Ambient fill — dim reddish tint on the Mars night side ────────────
        alight = AmbientLight('alight')
        alight.setColor(Vec4(0.25, 0.05, 0.05, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # ── Starfield (rendered in a separate background bin) ─────────────────
        star_node = make_starfield(num_stars=4_000, radius=8_000)
        self.stars = self.render.attachNewNode(star_node)
        self.stars.setLightOff()
        self.stars.setRenderModeThickness(2)
        self.stars.setBin('background', 0)
        self.stars.setDepthWrite(False)
        self.stars.setDepthTest(False)

        # ── Inertial reference frame arrows ───────────────────────────────────
        self._draw_inertial_frame()

        # ── GMO orbit ring (equatorial, no inclination) ───────────────────────
        gmo_ring = make_orbit_ring(omega=0.0, inc=0.0, radius=R_GMO_KM,
                                   color=(0.5, 0.5, 0.5, 1))
        gmo_ring.reparentTo(self.render)
        gmo_ring.setLightOff()

        # ── LMO orbit ring (inclined — angles from Orbit module) ──────────────
        try:
            lmo_omega  = float(orbit_mod.orbit_angles_LMO_t0[0])
            lmo_inc    = float(orbit_mod.orbit_angles_LMO_t0[1])
            lmo_radius = float(orbit_mod.r_LMO)
        except Exception:
            lmo_omega, lmo_inc, lmo_radius = math.radians(20), math.radians(30), R_LMO_KM

        self.lmo_ring = make_orbit_ring(omega=lmo_omega, inc=lmo_inc,
                                        radius=lmo_radius, color=(1, 1, 1, 1))
        self.lmo_ring.reparentTo(self.render)
        self.lmo_ring.setLightOff()

        # ── Communication window cone (35° half-angle, apex at Mars centre) ───
        # Reoriented every frame to point at GMO; recoloured green when active.
        self.comm_threshold_deg = 35.0   # maximum LMO–GMO separation for active link
        self.comm_preview_deg   = 30.0   # angular separation that triggers wide PiP view
        self.comm_preview_band  = 15.0   # ±tolerance around comm_preview_deg (deg)
        self.last_angle_sep     = 180.0  # LMO–GMO separation cache (updated each frame)
        self.comm_cone_height   = R_GMO_KM * SCENE_SCALE * 1.05  # slightly past GMO orbit
        self.comm_cone_node     = self.render.attachNewNode('comm_cone')
        self.comm_cone_node.setLightOff()
        self._rebuild_comm_cone(color=(0.6, 0.6, 0.6, 1))
        self.comm_cone_visible  = True
        self.comm_cone_node.hide()  # start hidden; toggle with 'C' key

        # ── GMO satellite (grey sphere + billboard label) ─────────────────────
        gmo_node       = make_uv_sphere(radius=10, slices=16, stacks=16)
        self.gmo_craft = self.render.attachNewNode(gmo_node)
        self.gmo_craft.setColor(0.5, 0.5, 0.5, 1)
        self.gmo_craft.setLightOff()

        gmo_label = TextNode('gmo_label')
        gmo_label.setText('GMO orbit')
        gmo_label.setTextColor(0.5, 0.5, 0.5, 1)
        gmo_label.setAlign(TextNode.ACenter)
        gmo_label_np = self.gmo_craft.attachNewNode(gmo_label)
        gmo_label_np.setPos(0, 0, 15)
        gmo_label_np.setScale(8.0)
        gmo_label_np.setBillboardPointEye()
        gmo_label_np.setLightOff()

    # ── LMO spacecraft model + body-frame axes ────────────────────────────────

    def _setup_spacecraft(self):
        """Attach the LMO spacecraft model and body-frame axis triad."""
        # Parent node — receives the attitude quaternion each frame.
        # Keep it separate from the model node so the model's local orientation
        # offset does not interfere with the attitude quaternion.
        self.spacecraft = self.render.attachNewNode('lmo_spacecraft')
        self.spacecraft.setScale(0.1)

        self.spacecraft_model = self.spacecraft.attachNewNode('lmo_spacecraft_model')
        self._load_spacecraft_model()

        # Body-frame axis triad rotates with the spacecraft attitude
        axis_node = make_axes_node(length=40.0, thickness=3.0)
        axis_node.reparentTo(self.spacecraft)
        axis_node.setLightOff()

    def _load_spacecraft_model(self):
        """Try satellite.glb → satellite.obj → fallback UV sphere."""
        for path in ('satellite.glb', 'satellite.obj'):
            try:
                model = self.loader.loadModel(path)
                if not model.isEmpty():
                    print(f'✓ Loaded spacecraft model: {path}')
                    model.reparentTo(self.spacecraft_model)
                    self.spacecraft_model.setHpr(0, 0, 0)
                    return
                print(f'✗ {path} is empty')
            except Exception as exc:
                print(f'✗ Could not load {path}: {exc}')

        print('→ Using fallback sphere for spacecraft')
        sc_node = make_uv_sphere(radius=4, slices=16, stacks=16)
        self.spacecraft_model.attachNewNode(sc_node)
        self.spacecraft_model.setColor(1, 1, 1, 1)

    # ── Simulation / attitude state ───────────────────────────────────────────

    def _setup_simulation_state(self):
        """Initialise sim time, attitude state, and quaternion interpolation buffers."""
        self.sim_time  = 0.0
        self.sim_speed = 100.0   # simulated seconds per real second

        # Attitude state vector: [σ₁ σ₂ σ₃ ω₁ ω₂ ω₃]  (MRP + body rates rad/s)
        try:
            self.att_state = orbit_mod.state_x_t0.copy()
        except Exception:
            self.att_state = np.zeros(6)

        # Integer sim-second counter — attitude RK4 runs at exactly 1 Hz
        self.last_integ_sec = -1

        # Quaternion buffers for slerp interpolation between 1-Hz RK4 steps
        self.last_att_state = self.att_state.copy()
        self.next_att_state = self.att_state.copy()
        self.last_quat_arr  = np.array([1.0, 0.0, 0.0, 0.0])
        self.next_quat_arr  = np.array([1.0, 0.0, 0.0, 0.0])

        # Pre-seed the first interpolation step with one RK4 step
        try:
            mode = orbit_mod.get_spacecraft_frame(0)
            mrp_err, omega_err = orbit_mod.attitude_error(
                0, self.att_state[:3], self.att_state[3:], mode)
            u0        = (-orbit_mod.K_controlsun * mrp_err
                         - orbit_mod.P_controlsun * omega_err)
            new_state = orbit_mod.RK4(self.att_state, u0, 0, 1,
                                      orbit_mod.attitude_derivatives)
            self.next_att_state = new_state
            ep = mrp_mod.MRP2EP(new_state[:3])
            self.last_quat_arr = self.next_quat_arr = np.array(ep[:4])
        except Exception:
            pass

    # ── Camera state ──────────────────────────────────────────────────────────

    def _setup_camera_state(self):
        """Initialise camera orbit parameters and smooth focus-transition state."""
        self.camera_distance = 600    # distance from focus point (scene units)
        self.camera_heading  = 0.0    # azimuth  (degrees, clockwise from −Y)
        self.camera_pitch    = 20.0   # elevation (degrees above horizon)
        self.last_mouse_x    = 0.0
        self.last_mouse_y    = 0.0
        self.mouse_dragging  = False

        # Focus target: 'mars' orbits Mars centre; 'lmo' follows the spacecraft
        self.focus_target = 'mars'

        # Smooth transition state — interpolates focus point and distance
        self.focus_transitioning          = False
        self.focus_transition_elapsed     = 0.0
        self.focus_transition_duration    = 1.0   # seconds (real time)
        self.focus_transition_start_pos   = Point3(0, 0, 0)
        self.focus_transition_start_dist  = self.camera_distance
        self.focus_transition_target_dist = self.camera_distance

    # ── HUD and subwindow ─────────────────────────────────────────────────────

    def _setup_ui(self):
        """Create onscreen text labels, load font, and open the PiP subwindow."""
        # Font — try several system paths; fall back to Panda3D default (None)
        font_candidates = ["NotoSans-VariableFont_wdth,wght.ttf"]
        self.ui_font = None
        for path in font_candidates:
            if os.path.exists(path):
                self.ui_font = self.loader.loadFont(path)
                break

        def label(text, pos, scale=0.06, may_change=True):
            """Helper: create a left-aligned white OnscreenText."""
            return OnscreenText(text=text, pos=pos, scale=scale,
                                fg=(1, 1, 1, 1), align=TextNode.ALeft,
                                mayChange=may_change, font=self.ui_font)

        # Status labels (top-left corner)
        self.time_text        = label('Sim Time: 0.0 s',           (-1.85,  0.92))
        self.speed_text       = label(f'Speed: {self.sim_speed:.0f}x', (-1.85, 0.84))
        self.focus_text       = label('Focus: MARS',                (-1.85,  0.76))
        self.comm_status_text = label('COMM: --',                   (-1.85,  0.68), scale=0.055)
        self.comm_status_text.hide()  # start hidden; only show when comm cone is visible

        # Elapsed real-time counter (top-right corner)
        self.sec_text = OnscreenText(
            text='Elapsed: 0s', pos=(1.18, 0.86), scale=0.045,
            fg=(1, 1, 1, 1), align=TextNode.ARight,
            mayChange=True, font=self.ui_font)

        # Telemetry plot cycling state
        self.current_plot_index = 0
        self.plot_names         = [
            'MRP Error', 'Angular Velocity', 'Control Torques', 'Hidden']
        self.plot_title_text    = None

        # LMO picture-in-picture subwindow
        self.lmo_display_region = None
        self.lmo_cam            = None
        self.lmo_border         = None
        self._create_lmo_subwindow()

    # ── Key bindings ──────────────────────────────────────────────────────────

    def _setup_keybindings(self):
        """Register all keyboard and mouse event handlers."""
        self.accept('mouse1',     self.start_drag)
        self.accept('mouse1-up',  self.stop_drag)
        self.accept('wheel_up',   self.zoom_in)
        self.accept('wheel_down', self.zoom_out)
        self.accept('arrow_up',   self.speed_up)
        self.accept('arrow_down', self.slow_down)
        self.accept('f',          self.toggle_focus)
        self.accept('p',          self.cycle_plot)
        self.accept('c',          self.toggle_comm_cone)

    # ── Panda3D task registration ─────────────────────────────────────────────

    def _setup_tasks(self):
        """Register the per-frame and periodic update tasks with the task manager."""
        self.taskMgr.add(self.update_simulation, 'update_simulation')
        self.taskMgr.add(self.update_camera,     'update_camera')
        self.taskMgr.doMethodLater(1.0, self._every_second_task, 'every_second_task')


    # ══════════════════════════════════════════════════════════════════════════
    # Per-frame simulation update
    # ══════════════════════════════════════════════════════════════════════════

    def update_simulation(self, task):
        """Advance orbital positions, attitude dynamics, and comm-cone state."""

        # Accumulate sim time from real frame time.
        # Using task.dt (real time) means changing sim_speed never causes jumps.
        self.sim_time += task.dt * self.sim_speed
        self.time_text.setText(f'Sim Time: {self.sim_time:.1f} s')

        # ── LMO spacecraft position ───────────────────────────────────────────
        try:
            angles_lmo    = orbit_mod.orbit_angles_LMO_t0.copy()
            angles_lmo[2] = angles_lmo[2] + orbit_mod.thetadot_LMO * self.sim_time
            _, pos_lmo_km, _ = orbit_mod.inertial_orbit_pos_vel(
                orbit_mod.r_LMO, orbit_mod.miu, angles_lmo)
            pos_lmo = [p * SCENE_SCALE for p in pos_lmo_km]
        except Exception:
            # Fallback: simple circular orbit if Orbit module unavailable
            theta_lmo = math.radians(60) + THETADOT_LMO * self.sim_time
            pos_lmo   = orbit_point(theta_lmo, math.radians(20), math.radians(30), R_LMO_KM)
        self.spacecraft.setPos(*pos_lmo)

        # ── Attitude integration (runs at 1 Hz, smoothed by slerp at render rate)
        current_sec = int(math.floor(self.sim_time))
        while self.last_integ_sec < current_sec:
            self.last_integ_sec += 1
            self._run_attitude_step(self.last_integ_sec)

        # Slerp between last and next 1-Hz quaternions for smooth visual rotation
        try:
            frac = self.sim_time - math.floor(self.sim_time)
            qarr = self._slerp_quat(self.last_quat_arr, self.next_quat_arr, frac)
            self.spacecraft.setQuat(Quat(qarr[0], qarr[1], qarr[2], qarr[3]))
        except Exception:
            pass

        # ── GMO satellite position ────────────────────────────────────────────
        try:
            angles_gmo    = orbit_mod.orbit_angles_GMO_t0.copy()
            angles_gmo[2] = angles_gmo[2] + orbit_mod.thetadot_GMO * self.sim_time
            _, pos_gmo_km, _ = orbit_mod.inertial_orbit_pos_vel(
                orbit_mod.r_GMO, orbit_mod.miu, angles_gmo)
            pos_gmo = [p * SCENE_SCALE for p in pos_gmo_km]
        except Exception:
            theta_gmo = math.radians(250) + THETADOT_GMO * self.sim_time
            pos_gmo   = orbit_point(theta_gmo, 0.0, 0.0, R_GMO_KM)
        self.gmo_craft.setPos(*pos_gmo)

        # ── Communication cone: aim at GMO, recolour based on link status ─────
        try:
            self.comm_cone_node.setPos(0, 0, 0)
            self.comm_cone_node.lookAt(Point3(*pos_gmo))

            angle_sep          = vec_angle_deg(pos_lmo, pos_gmo)
            self.last_angle_sep = angle_sep
            comm_active        = angle_sep < self.comm_threshold_deg
            cone_color         = (0.2, 1.0, 0.2, 1) if comm_active else (0.6, 0.6, 0.6, 1)
            self._rebuild_comm_cone(color=cone_color)

            status = 'ACTIVE' if comm_active else 'INACTIVE'
            self.comm_status_text.setText(
                f'COMM: {status}  Δθ(LMO,GMO)={angle_sep:.1f}°'
                f'  (limit {self.comm_threshold_deg:.0f}°)')
        except Exception:
            pass

        # ── LMO orbit ring (rebuilt every frame to match Orbit module's frame) ─
        try:
            self._update_lmo_ring()
        except Exception:
            pass

        return task.cont


    # ══════════════════════════════════════════════════════════════════════════
    # Per-frame camera update
    # ══════════════════════════════════════════════════════════════════════════

    def update_camera(self, task):
        """Orbit the camera around the focus point; apply smooth focus transitions."""

        # ── Mouse-drag rotation ───────────────────────────────────────────────
        if self.mouse_dragging and self.mouseWatcherNode.hasMouse():
            mx = self.mouseWatcherNode.getMouseX()
            my = self.mouseWatcherNode.getMouseY()
            self.camera_heading -= (mx - self.last_mouse_x) * 50
            self.camera_pitch    = max(-89, min(89,
                self.camera_pitch - (my - self.last_mouse_y) * 50))
            self.last_mouse_x, self.last_mouse_y = mx, my

        # ── Focus-transition lerp ─────────────────────────────────────────────
        target_focus = self._current_focus_point()
        target_dist  = self.camera_distance

        if self.focus_transitioning:
            self.focus_transition_elapsed += task.dt
            t = self.focus_transition_elapsed / self.focus_transition_duration
            s = self._smoothstep(t)

            # Interpolate both the look-at point and the camera distance
            focus = (self.focus_transition_start_pos
                     + (target_focus - self.focus_transition_start_pos) * s)
            dist  = (self.focus_transition_start_dist
                     + (self.focus_transition_target_dist
                        - self.focus_transition_start_dist) * s)

            if t >= 1.0:
                self.focus_transitioning = False
        else:
            focus = target_focus
            dist  = target_dist

        # ── Place camera on a sphere around the focus point ───────────────────
        h = math.radians(self.camera_heading)
        p = math.radians(self.camera_pitch)
        self.camera.setPos(
            focus.x + dist * math.cos(p) * math.sin(h),
            focus.y - dist * math.cos(p) * math.cos(h),
            focus.z + dist * math.sin(p))
        self.camera.lookAt(focus)

        self._update_lmo_subwindow()
        return task.cont


    # ══════════════════════════════════════════════════════════════════════════
    # Attitude integration
    # ══════════════════════════════════════════════════════════════════════════

    def _run_attitude_step(self, sim_index: int):
        """Advance the attitude ODE by one simulated second using RK4.

        Control law:  u = −K · σ_err − P · ω_err   (PD feedback in MRP space)

        The quaternion buffers (last_quat_arr / next_quat_arr) are updated so
        that update_camera can slerp between them at the render frame rate.
        """
        try:
            mode = orbit_mod.get_spacecraft_frame(sim_index)
            mrp_err, omega_err = orbit_mod.attitude_error(
                sim_index, self.att_state[:3], self.att_state[3:], mode)

            # PD control torque
            u = (-orbit_mod.K_controlsun * mrp_err
                 - orbit_mod.P_controlsun * omega_err)

            new_state = orbit_mod.RK4(self.att_state, u, sim_index, 1,
                                      orbit_mod.attitude_derivatives)
            sigma = new_state[:3]
            omega = new_state[3:]

            # Switch to the MRP shadow set when |σ| > 1 to avoid singularity
            if np.linalg.norm(sigma) > 1:
                sigma = -sigma / np.linalg.norm(sigma) ** 2

            # Shift buffers: last ← current next
            self.last_att_state = self.next_att_state.copy()
            try:
                ep = mrp_mod.MRP2EP(self.last_att_state[:3])
                self.last_quat_arr = np.array(ep[:4])
            except Exception:
                self.last_quat_arr = np.array([1.0, 0.0, 0.0, 0.0])

            self.next_att_state = np.concatenate([sigma, omega])
            self.att_state      = self.next_att_state.copy()
            try:
                ep = mrp_mod.MRP2EP(sigma)
                self.next_quat_arr = np.array(ep[:4])
            except Exception:
                self.next_quat_arr = np.array([1.0, 0.0, 0.0, 0.0])

        except Exception:
            pass   # Never crash the render loop on a bad integration step


    # ══════════════════════════════════════════════════════════════════════════
    # Scene geometry helpers  (called on demand or rebuilt each frame)
    # ══════════════════════════════════════════════════════════════════════════

    def _rebuild_comm_cone(self, color=(0.6, 0.6, 0.6, 1)):
        """Destroy and recreate the comm-cone geometry with a new colour."""
        for child in self.comm_cone_node.getChildren():
            child.removeNode()
        cone_np = make_comm_cone_lines(
            height=self.comm_cone_height,
            half_angle_deg=self.comm_threshold_deg,
            segments=32,
            color=color)
        cone_np.reparentTo(self.comm_cone_node)
        cone_np.setLightOff()

    def _update_lmo_ring(self, slices: int = 200):
        """Rebuild the LMO orbit ring using Orbit module's inertial frame functions."""
        try:
            base  = orbit_mod.orbit_angles_LMO_t0.copy()
            omega = float(base[0])
            inc   = float(base[1])
            rkm   = float(orbit_mod.r_LMO)
        except Exception:
            omega, inc, rkm = math.radians(20), math.radians(30), R_LMO_KM

        lines = LineSegs()
        lines.setColor(1, 1, 1, 1)
        lines.setThickness(1.5)

        first = True
        for i in range(slices + 1):
            phi    = 2 * math.pi * i / slices
            angles = np.array([omega, inc, orbit_mod.orbit_angles_LMO_t0[2] + phi])
            try:
                _, pos_km, _ = orbit_mod.inertial_orbit_pos_vel(
                    rkm, orbit_mod.miu, angles)
                pos = [p * SCENE_SCALE for p in pos_km]
            except Exception:
                pos = orbit_point(phi, omega, inc, rkm)

            if first:
                lines.moveTo(*pos)
                first = False
            else:
                lines.drawTo(*pos)

        new_ring = NodePath(lines.create())
        new_ring.reparentTo(self.render)
        new_ring.setLightOff()

        try:
            self.lmo_ring.removeNode()
        except Exception:
            pass
        self.lmo_ring = new_ring

    def _draw_inertial_frame(self, length: float = 125.0):
        """Draw the N1 / N2 / N3 inertial reference frame arrows at the origin."""
        axes = [
            ([1, 0, 0], (1.0, 0.3, 0.3, 1), 'N1'),
            ([0, 1, 0], (0.3, 1.0, 0.3, 1), 'N2'),
            ([0, 0, 1], (0.3, 0.6, 1.0, 1), 'N3'),
        ]
        for direction, color, label in axes:
            tip = [d * length for d in direction]

            # Arrow line
            lines = LineSegs()
            lines.setColor(*color)
            lines.setThickness(2.5)
            lines.moveTo(0, 0, 0)
            lines.drawTo(*tip)
            self.render.attachNewNode(lines.create()).setLightOff()

            # Small sphere at the arrowhead
            tip_np = self.render.attachNewNode(make_uv_sphere(radius=2.0, slices=8, stacks=8))
            tip_np.setPos(*tip)
            tip_np.setColor(*color)
            tip_np.setLightOff()

            # Billboard text label
            text_node = TextNode(label)
            text_node.setText(label)
            text_node.setTextColor(*color)
            text_node.setAlign(TextNode.ACenter)
            text_np = self.render.attachNewNode(text_node)
            text_np.setPos(*[d * (length + 8) for d in direction])
            text_np.setScale(8.0)
            text_np.setBillboardPointEye()
            text_np.setLightOff()


    # ══════════════════════════════════════════════════════════════════════════
    # LMO picture-in-picture subwindow
    # ══════════════════════════════════════════════════════════════════════════

    def _create_lmo_subwindow(self):
        """Create the top-right PiP viewport with border and telemetry plot."""
        if self.lmo_display_region is not None:
            return

        self.lmo_cam = self.makeCamera(self.win, displayRegion=(0.72, 0.98, 0.72, 0.98))
        self.lmo_cam.reparentTo(self.render)
        self.lmo_cam.node().getLens().setFov(15)

        self.lmo_display_region = self.lmo_cam.node().getDisplayRegion(0)
        self.lmo_display_region.setSort(20)
        self.lmo_display_region.setClearColorActive(True)
        self.lmo_display_region.setClearColor(Vec4(0, 0, 0, 1))

        self._create_lmo_border()
        self._create_lmo_plot()
        self._update_lmo_subwindow()

    def _create_lmo_border(self):
        """Draw a white rectangle around the LMO PiP viewport in 2-D overlay space."""
        if self.lmo_border is not None:
            return

        # Convert the display-region fractions (0–1) to render2d coordinates (−1 to +1)
        l, r, b, t = 0.72, 0.98, 0.72, 0.98
        min_x, max_x = l * 2 - 1, r * 2 - 1
        min_y, max_y = b * 2 - 1, t * 2 - 1

        border = LineSegs()
        border.setThickness(3.0)
        border.setColor(1, 1, 1, 1)
        border.moveTo(min_x, 0, min_y)
        border.drawTo(max_x, 0, min_y)
        border.drawTo(max_x, 0, max_y)
        border.drawTo(min_x, 0, max_y)
        border.drawTo(min_x, 0, min_y)

        self.lmo_border = self.render2d.attachNewNode(border.create())
        self.lmo_border.setBin('fixed', 0)
        self.lmo_border.setDepthTest(False)
        self.lmo_border.setDepthWrite(False)

    def _update_lmo_subwindow(self):
        """Reposition the PiP camera each frame based on spacecraft mode and geometry."""
        if self.lmo_display_region is None or not self.lmo_display_region.isActive():
            return

        lmo_pos = self.spacecraft.getPos()
        gmo_pos = self.gmo_craft.getPos()

        try:
            tsec           = int(math.floor(self.sim_time))
            spacecraft_mode = int(orbit_mod.get_spacecraft_frame(tsec))
        except Exception:
            spacecraft_mode = 2   # fallback: nadir pointing

        # Priority 1: near the comm window — wide top-down view of both orbits
        near_comm_window = abs(self.last_angle_sep - self.comm_preview_deg) <= self.comm_preview_band
        if near_comm_window:
            self.lmo_cam.setPos(gmo_pos[0] * 1.825, -500 + gmo_pos[1], gmo_pos[2])
            self.lmo_cam.lookAt(0, 50, 0)

        # Priority 2: sun-pointing mode — top-down view from above Mars (N3)
        elif spacecraft_mode == 1:
            self.lmo_cam.setPos(0, 0, 900)
            self.lmo_cam.lookAt(0, 75, 0)

        # Default: close-up follow cam beside the LMO spacecraft
        else:
            self.lmo_cam.setPos(self._safe_lmo_cam_pos(lmo_pos, scale=2.5))
            self.lmo_cam.lookAt(self.spacecraft)

        try:
            self._update_lmo_plot()
        except Exception:
            pass

    def _safe_lmo_cam_pos(self, target_pos: Point3, scale: float = 2.0) -> Point3:
        """Return a camera position tangent to the orbit plane that avoids Mars.

        The camera is placed to the side of the spacecraft (perpendicular to the
        radial direction in the XY plane) and slightly above the orbit plane.
        If the line of sight would clip through Mars, the camera is raised further.
        """
        mars_r_scene = MARS_RADIUS_KM * SCENE_SCALE

        # Radial unit vector in the XY plane
        radial = Point3(target_pos.x, target_pos.y, 0)
        radial = Point3(0, 1, 0) if radial.lengthSquared() < 0.001 else radial / radial.length()

        # Side vector (90° from radial in XY plane)
        side = Point3(-radial.y, radial.x, 0)
        if side.lengthSquared() < 0.001:
            side = Point3(1, 0, 0)
        else:
            side /= side.length()

        cam_pos = target_pos + side * (80 * scale) + Point3(0, 0, 40 * scale)

        # Check closest approach of the LOS to Mars; raise camera if needed
        los = target_pos - cam_pos
        t   = max(0.0, min(1.0, -(cam_pos.dot(los)) / los.lengthSquared()))
        if (cam_pos + los * t).length() < mars_r_scene + 5.0:
            cam_pos += Point3(0, 0, 40 * scale)

        return cam_pos


    # ══════════════════════════════════════════════════════════════════════════
    # Telemetry plot  (bottom-right corner)
    # ══════════════════════════════════════════════════════════════════════════

    def _create_lmo_plot(self, width_frac: float = 0.28,
                         height_frac: float = 0.28, samples: int = 200):
        """Initialise the telemetry plot area, data buffers, and LineSegs nodes."""
        # Map screen fractions (0–1) to render2d coordinates (−1 to +1)
        min_x = (1.0 - width_frac) * 2 - 1
        max_x = 1.0 * 2 - 1
        min_y = -1.0
        max_y = min_y + height_frac * 2

        self.lmo_plot_area    = (min_x, max_x, min_y, max_y)
        self.lmo_plot_samples = samples
        self.lmo_plot_idx     = 0

        # Circular ring-buffers — one numpy array per component per channel
        self.mrp_plot_buf     = {k: np.zeros(samples) for k in MRP_COLORS}
        self.omega_plot_buf   = {k: np.zeros(samples) for k in OMEGA_COLORS}
        self.control_plot_buf = {k: np.zeros(samples) for k in CONTROL_COLORS}

        # 2-D overlay parent node
        self.lmo_plot_node = self.render2d.attachNewNode('lmo_plot')
        self.lmo_plot_node.setDepthTest(False)
        self.lmo_plot_node.setDepthWrite(False)

        # One LineSegs NodePath per component (recreated each draw call)
        self.mrp_plot_lines     = self._make_line_nodes(MRP_COLORS)
        self.omega_plot_lines   = self._make_line_nodes(OMEGA_COLORS)
        self.control_plot_lines = self._make_line_nodes(CONTROL_COLORS)

        # Plot title
        cx = (min_x + max_x) / 2
        self.plot_title_text = OnscreenText(
            text="Press 'P' to cycle plots",
            pos=(cx - 0.3, max_y + 0.04),
            scale=0.08, fg=(1, 1, 1, 1),
            align=TextNode.ACenter, mayChange=True, font=self.ui_font)

        # Transient elements cleared and recreated each frame
        self.plot_grid_node    = None
        self.plot_border_node  = None
        self.plot_tick_texts   = []
        self.plot_legend_texts = []

    def _make_line_nodes(self, colors: dict) -> dict:
        """Create one empty LineSegs NodePath per key in *colors*."""
        nodes = {}
        for key, rgba in colors.items():
            ls = LineSegs()
            ls.setThickness(2.0)
            ls.setColor(*rgba)
            node = self.lmo_plot_node.attachNewNode(ls.create())
            node.setDepthTest(False)
            node.setDepthWrite(False)
            nodes[key] = node
        return nodes

    def _update_lmo_plot(self):
        """Append latest telemetry samples to ring buffers and redraw active plot."""
        if not hasattr(self, 'lmo_plot_area'):
            return

        tsec = int(math.floor(self.sim_time))
        try:
            mode = int(orbit_mod.get_spacecraft_frame(tsec))
        except Exception:
            mode = 2

        try:
            mrp_err, omega_err = orbit_mod.attitude_error(
                tsec, self.att_state[:3], self.att_state[3:], mode)
            u = (-orbit_mod.K_controlsun * mrp_err
                 - orbit_mod.P_controlsun * omega_err)
        except Exception:
            mrp_err = u = np.zeros(3)

        omega = self.att_state[3:]
        idx   = self.lmo_plot_idx % self.lmo_plot_samples

        for i, key in enumerate(('e1', 'e2', 'e3')):
            self.mrp_plot_buf[key][idx]     = float(mrp_err[i])
        for i, key in enumerate(('w1', 'w2', 'w3')):
            self.omega_plot_buf[key][idx]   = float(omega[i])
        for i, key in enumerate(('u1', 'u2', 'u3')):
            self.control_plot_buf[key][idx] = float(u[i])

        self.lmo_plot_idx += 1
        self._clear_plot_elements()

        min_x, max_x, min_y, max_y = self.lmo_plot_area

        # Dispatch to the generic _draw_plot with the appropriate channel parameters
        if self.current_plot_index == 0:
            self._draw_plot(min_x, max_x, min_y, max_y,
                            self.mrp_plot_buf, self.mrp_plot_lines,
                            MRP_COLORS, MRP_LABELS, -1.0, 1.0,
                            'Attitude Error (MRP)')
        elif self.current_plot_index == 1:
            self._draw_plot(min_x, max_x, min_y, max_y,
                            self.omega_plot_buf, self.omega_plot_lines,
                            OMEGA_COLORS, OMEGA_LABELS, -0.05, 0.05,
                            'Angular Velocity (rad/s)')
        elif self.current_plot_index == 2:
            self._draw_plot(min_x, max_x, min_y, max_y,
                            self.control_plot_buf, self.control_plot_lines,
                            CONTROL_COLORS, CONTROL_LABELS, -0.01, 0.01,
                            'Control Torques (N·m)')
        elif self.current_plot_index == 3:
            self.plot_title_text.setText('')   # hidden — blank plot

    def _clear_plot_elements(self):
        """Remove all transient plot nodes (grid, border, ticks, legends, lines)."""
        for attr in ('plot_grid_node', 'plot_border_node'):
            node = getattr(self, attr, None)
            if node is not None:
                try:
                    node.removeNode()
                except Exception:
                    pass
                setattr(self, attr, None)

        for text_list in (self.plot_tick_texts, self.plot_legend_texts):
            for t in text_list:
                try:
                    t.removeNode()
                except Exception:
                    pass
            text_list.clear()

        # Remove existing line nodes and recreate empty ones for the next draw
        for line_dict, colors in (
            (self.mrp_plot_lines,     MRP_COLORS),
            (self.omega_plot_lines,   OMEGA_COLORS),
            (self.control_plot_lines, CONTROL_COLORS),
        ):
            for node in line_dict.values():
                try:
                    node.removeNode()
                except Exception:
                    pass
            line_dict.update(self._make_line_nodes(colors))

    def _draw_plot(self, min_x: float, max_x: float, min_y: float, max_y: float,
                   buf: dict, line_nodes: dict,
                   colors: dict, labels: dict,
                   val_min: float, val_max: float, title: str):
        """Generic telemetry plot renderer shared by all three data channels.

        Draws a border rectangle, horizontal grid lines with tick labels, the
        scrolling data lines, and a colour-coded legend.

        Args:
            buf:        Ring-buffer dict  {component_key: np.ndarray}.
            line_nodes: LineSegs node dict {component_key: NodePath}.
            colors:     RGBA colour dict  {component_key: (r,g,b,a)}.
            labels:     Legend label dict {component_key: str}.
            val_min/max: Y-axis data range.
            title:      String shown above the plot.
        """
        if self.plot_title_text is not None:
            self.plot_title_text.setText(title)
            self.plot_title_text.setPos(max_x*1.32, max_y*0.80)

        width     = max_x - min_x
        height    = max_y - min_y
        val_range = val_max - val_min

        # Border rectangle
        try:
            b = LineSegs()
            b.setThickness(2.0)
            b.setColor(0.7, 0.7, 0.7, 1)
            b.moveTo(min_x, 0, min_y); b.drawTo(max_x, 0, min_y)
            b.drawTo(max_x, 0, max_y); b.drawTo(min_x, 0, max_y)
            b.drawTo(min_x, 0, min_y)
            self.plot_border_node = self.lmo_plot_node.attachNewNode(b.create())
            self.plot_border_node.setDepthTest(False)
            self.plot_border_node.setDepthWrite(False)
        except Exception:
            pass

        # Horizontal grid lines + numeric tick labels
        try:
            g = LineSegs()
            g.setThickness(1.5)
            g.setColor(0.6, 0.6, 0.6, 0.8)
            for val in np.linspace(val_min, val_max, 5):
                y = min_y + ((val - val_min) / val_range) * height
                g.moveTo(min_x, 0, y)
                g.drawTo(max_x, 0, y)
                tick = OnscreenText(
                    text=f'{val:.3f}',
                    pos=(min_x * 1.75 - 0.02, y + 0.01),
                    scale=0.05, fg=(1, 1, 1, 1),
                    align=TextNode.ARight, mayChange=False, font=self.ui_font)
                self.plot_tick_texts.append(tick)
            self.plot_grid_node = self.lmo_plot_node.attachNewNode(g.create())
            self.plot_grid_node.setDepthTest(False)
            self.plot_grid_node.setDepthWrite(False)
        except Exception:
            pass

        # Scrolling data lines
        N = self.lmo_plot_samples
        for key, node in line_nodes.items():
            ls = LineSegs()
            ls.setThickness(2.0)
            ls.setColor(*colors[key])
            data  = buf[key]
            first = True
            for i in range(N):
                j = (self.lmo_plot_idx - N + i) % N
                x = min_x + (i / (N - 1)) * width if N > 1 else min_x
                y = min_y + ((data[j] - val_min) / val_range) * height
                if first:
                    ls.moveTo(x, 0, y)
                    first = False
                else:
                    ls.drawTo(x, 0, y)
            node.removeNode()
            new_node = self.lmo_plot_node.attachNewNode(ls.create())
            new_node.setDepthTest(False)
            new_node.setDepthWrite(False)
            line_nodes[key] = new_node

        # Colour-coded legend (top-right of plot)
        try:
            legend_y = max_y - 0.03
            for i, (key, rgba) in enumerate(colors.items()):
                leg = OnscreenText(
                    text=labels[key],
                    pos=(max_x * 1.75 - 0.02 - i * 0.075, legend_y - 0.02),
                    scale=0.06, fg=rgba,
                    align=TextNode.ARight, mayChange=False, font=self.ui_font)
                self.plot_legend_texts.append(leg)
        except Exception:
            pass


    # ══════════════════════════════════════════════════════════════════════════
    # Input handlers
    # ══════════════════════════════════════════════════════════════════════════

    def toggle_focus(self):
        """Cycle camera focus between Mars (overview) and LMO spacecraft (close-up)."""
        if self.focus_target == 'mars':
            new_target, new_dist = 'lmo', 20
        else:
            new_target, new_dist = 'mars', 600

        # Snapshot the current focus point as the transition start
        self.focus_transition_start_pos   = self._current_focus_point()
        self.focus_transition_start_dist  = self.camera_distance
        self.focus_transition_target_dist = new_dist
        self.focus_transition_elapsed     = 0.0
        self.focus_transition_duration    = 0.25
        self.focus_transitioning          = True

        self.focus_target    = new_target
        self.camera_distance = new_dist
        self.focus_text.setText(f'Focus: {new_target.upper()}')

    def toggle_comm_cone(self):
        """Show or hide the 35° communication window cone."""
        self.comm_cone_visible = not self.comm_cone_visible
        if self.comm_cone_visible:
            self.comm_cone_node.show()
            self.comm_status_text.show()
        else:
            self.comm_cone_node.hide()
            self.comm_status_text.hide()

    def cycle_plot(self):
        """Advance the active telemetry plot: MRP → ω → torques → hidden → MRP…"""
        self.current_plot_index = (self.current_plot_index + 1) % len(self.plot_names)

    def start_drag(self):
        """Begin mouse-drag camera rotation."""
        self.mouse_dragging = True
        if self.mouseWatcherNode.hasMouse():
            self.last_mouse_x = self.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def stop_drag(self):
        """End mouse-drag camera rotation."""
        self.mouse_dragging = False

    def zoom_in(self):
        """Decrease camera distance by 10 % per scroll tick."""
        self.camera_distance = max(1, self.camera_distance * 0.9)

    def zoom_out(self):
        """Increase camera distance by 10 % per scroll tick."""
        self.camera_distance = min(5_000, self.camera_distance * 1.1)

    def speed_up(self):
        """Double the simulation speed (capped at 50 000×)."""
        self.sim_speed = min(50_000, self.sim_speed * 2)
        self.speed_text.setText(f'Speed: {self.sim_speed:.0f}x')

    def slow_down(self):
        """Halve the simulation speed (floor at 10×)."""
        self.sim_speed = max(10, self.sim_speed / 2)
        self.speed_text.setText(f'Speed: {self.sim_speed:.0f}x')


    # ══════════════════════════════════════════════════════════════════════════
    # Utilities / pure functions
    # ══════════════════════════════════════════════════════════════════════════

    def _current_focus_point(self) -> Point3:
        """Return the world-space point the camera should orbit around."""
        return self.spacecraft.getPos() if self.focus_target == 'lmo' else Point3(0, 0, 0)

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Ease-out cubic easing function: fast start, smooth deceleration.

        Maps t ∈ [0, 1] → s ∈ [0, 1] with zero derivative at t = 1.
        """
        t = max(0.0, min(1.0, t))
        return 1 - (1 - t) ** 3

    @staticmethod
    def _slerp_quat(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions [w, x, y, z].

        Always takes the shortest arc (flips q2 sign when dot < 0).
        Falls back to normalised linear interpolation when the quaternions are
        nearly identical (dot > 0.9995) to avoid numerical issues near θ ≈ 0.

        Args:
            q1: Start quaternion as a numpy array [w, x, y, z].
            q2: End quaternion as a numpy array [w, x, y, z].
            t:  Interpolation parameter in [0, 1].
        """
        q1  = np.asarray(q1, dtype=float)
        q2  = np.asarray(q2, dtype=float)
        dot = np.dot(q1, q2)

        # Ensure shortest-arc interpolation
        if dot < 0.0:
            q2  = -q2
            dot = -dot

        if dot > 0.9995:
            # Nearly identical — use normalised lerp (avoids sin(0) denominator)
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0     = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta       = theta_0 * t

        s1 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s2 = math.sin(theta) / sin_theta_0
        return s1 * q1 + s2 * q2

    def _every_second_task(self, task):
        """Panda3D task: update the elapsed-time HUD label once per real second."""
        self.sec_text.setText(f'Elapsed: {int(self.sim_time)}s')
        return task.again


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = MarsSimulation()
    app.run()