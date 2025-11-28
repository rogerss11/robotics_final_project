import numpy as np
from functions import (
    Ixx,
    compute_torques_trajectory,
    ik_solver,
    generate_circle_points,
    geometric_jacobian,
    solve_quintic,
    eval_quintic
)

# ================================================================
# 1. --- PHYSICAL PARAMETERS ---
# ================================================================

m1 = 0.060   # mass link 1  (kg)
m2 = 0.080   # mass link 2
m3 = 0.080   # mass link 3
m4 = 0.040   # mass link 4

masses = np.array([m1, m2, m3, m4])

# Example link dimensions (mm → convert to meters if needed)
dx = 25
dy = 55
dz = 32

# ---------------------------------------------------------------
# Inertia values
# ---------------------------------------------------------------
I0 = Ixx(m1, dz, dy)
print("Ixx =", I0)
print("Calculated Iyy =", Ixx(m1, dx, dz))
print("Real Iyy =",  0.4*I0)
print("Calculated Izz =", Ixx(m1, dx, dy))
print("Real Izz =",  0.9*I0)

D1 = np.diag([I0, 0.4*I0, 0.9*I0])
D2 = np.diag([0.45*I0, 1.4*I0, 1.2*I0])
D3 = np.diag([0.45*I0, 1.4*I0, 1.2*I0])
D4 = np.diag([0.5*I0, 0.5*I0, 0.5*I0])

I_local = [D1, D2, D3, D4]

print("\nD1 =\n", D1)
print("D2 = D3 =\n", D2)
print("D4 =\n", D4)

# ================================================================
# 2. --- COM positions of each link in its own frame ---
#     (Example: midpoint of each link)
# ================================================================

# Adjust based on your actual CAD
r_ci_list = [
    np.array([0, 20, 0]),   # COM of link 1
    np.array([30, 0, 0]),   # link 2
    np.array([30, 0, 0]),   # link 3
    np.array([-25, 15, 0])  # link 4
]

# ================================================================
# 3. --- GRAVITY ---
# ================================================================
g_vec = np.array([0, 0, -9.81])

# =====================================================================
# 4. --- IMPLEMENT THE EXACT SAME TRAJECTORY FROM EX6  ---
# =====================================================================

# 1) Circle path (stylus tip)
N = 37
points = generate_circle_points(N)

# four knot points at 0,2,4,6,8 seconds
idxs  = [0, 9, 18, 27, 36]
times = np.array([0, 2, 4, 6, 8])

# 2) IK at knots (adjust for stylus 50 mm offset)
Q = []
for k in idxs:
    x_t, y_t, z_t = points[k]
    x4 = x_t
    qv = ik_solver(x4, y_t, z_t, c=0.0)
    Q.append(qv)

Q = np.array(Q)      # shape (5,4)

# 3) Compute joint velocities Qdot from Jacobian
# tip (and frame-4 origin) velocities at knot points
V_knots = [
    np.array([0, 0, 0]),
    np.array([0, -27, 0]),
    np.array([0, 0, -27]),
    np.array([0, 27, 0]),
    np.array([0, 0, 0]),
]
OMEGA = np.zeros(3)

Qdot = np.zeros_like(Q)

for k in range(len(times)):
    J = geometric_jacobian(Q[k], 4)
    xdot = np.hstack([V_knots[k], OMEGA])
    qdot = np.linalg.pinv(J) @ xdot
    Qdot[k] = qdot

Qddot = np.zeros_like(Q)

# 4) Quintic polynomial coefficients
Tseg = 2.0

coeffs = [
    [
        solve_quintic(Q[s, j], Q[s+1, j], Qdot[s, j], Qdot[s+1, j])
        for j in range(4)
    ]
    for s in range(4)
]

# 5) Evaluate over full 0–8 s

t_fine = np.linspace(0, 8, 400)
q_vals   = np.zeros((400, 4))
qd_vals  = np.zeros((400, 4))
qdd_vals = np.zeros((400, 4))

for i, t in enumerate(t_fine):
    seg = min(int(t // 2), 3)
    τ = t - seg * 2
    for j in range(4):
        q_vals[i, j], qd_vals[i, j], qdd_vals[i, j] = eval_quintic(coeffs[seg][j], τ)


# ================================================================
# 5. --- COMPUTE TORQUES FOR THE ENTIRE TRAJECTORY ---
# ================================================================

tau = compute_torques_trajectory(
    q_vals, qd_vals, qdd_vals,
    masses, I_local, r_ci_list, g_vec
)

# ================================================================
# 6. --- PLOT TORQUES OVER THE TRAJECTORY ---
# ================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
labels = [r'$\tau_1$', r'$\tau_2$', r'$\tau_3$', r'$\tau_4$']

for j in range(4):
    plt.plot(t_fine, tau[:, j]/1000, label=labels[j], linewidth=2)

plt.title("Joint Torques Over 0–8 s Trajectory")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
