import numpy as np
from functions import (
    generate_one_circle_point,
    ik_solver,
    joint_velocities
)

# -----------------------------------------
# 1. Define phi = π/2 and compute point
# -----------------------------------------
phi = np.pi / 2
x, y, z = generate_one_circle_point(phi)

# Orientation constraint: horizontal stylus → x4z = 0
c = 0

# -----------------------------------------
# 2. Joint configuration q at φ = π/2
# -----------------------------------------
q = ik_solver(x, y, z, c)

print("Joint angles q at φ = π/2 (3 s.f.):")
print(np.round(q, 3))

# -----------------------------------------
# 3. Desired EE velocity (frame {4})
# v = [vx, vy, vz, wx, wy, wz]
# -----------------------------------------
v04 = np.array([0, -3, 0, 0, 0, 0], dtype=float)

# -----------------------------------------
# 4. Compute joint velocities q_dot
# -----------------------------------------
q_dot = joint_velocities(q, v04)

print("\nRequired joint velocities q_dot (5 s.f.):")
print(np.round(q_dot,10))