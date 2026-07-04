# Spacecraft_Dynamics_Capstone_Project_Simulation
This repository contains a real-time 3D visualization of the final Capstone Project for the Spacecraft Dynamics and Control Specialization offered by the University of Colorado Boulder on Coursera. The simulation models spacecraft attitude dynamics and orbital motion around Mars using Python and Panda3D.

<img width="800" height="450" alt="Mars_Capstone_Project_Jero" src="https://github.com/user-attachments/assets/11c3776b-13dd-4468-a1c9-60544388b051" />

## Main Features

- Real-time 3D Mars visualization
- Low Mars Orbit (LMO) spacecraft
- Geostationary Mars Orbit (GMO) relay spacecraft
- Communication cone visualization
- Simulation speed controls
- Multiple camera modes
<img width="1912" height="1012" alt="Mars_Sim" src="https://github.com/user-attachments/assets/4cc988df-adb9-462c-900f-7ebb9420573d" />

- Attitude propagation
- MRP attitude representation
- Embedded live telemetry plots
<img width="1282" height="262" alt="Plots" src="https://github.com/user-attachments/assets/52780a2a-2b59-4460-ace6-729e6ba8b157" />

# Main Scripts
Visual_Simulation.py - Real-time 3D visualisation of Mars orbital mechanics using the Panda3D engine.

DCM.py - Direction Cosine Matrix primitives (M1, M3 rotation matrices).

MRP.py - Modified Rodrigues Parameter ↔ Euler Parameter (quaternion) conversion.

Orbit.py - Orbital angles, inertial-frame position/velocity, RK4 integrator, attitude-error functions, and spacecraft-mode selector.

# Installation & Run the Simulation
python Visual_Simulation.py

# Controls

F - Toggle camera focus  (Mars ↔ LMO spacecraft)

↑ / ↓ - Double / halve simulation speed

Scroll - Zoom in / out

Left mouse drag - Orbit camera around focus point

C - Toggle 35° communication cone

P - Cycle telemetry plot  (MRP error → ω → control torques → hidden)
