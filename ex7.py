import numpy as np
import matplotlib.pyplot as plt

# === IMPORT HELPERS ===
from functions import (
    ik_solver,
    generate_circle_points,
    compute_T0i
)

# ============================================================
# Helper: Forward kinematics → end-effector position
# ============================================================
def fk_end_effector(q):
    """Return EE {4} position in world frame."""
    T04 = compute_T0i(q, 4)
    return np.array(T04[:3,3]).astype(float).flatten()

# ============================================================
# Quintic interpolation helper (same as in ex6)
# ============================================================
Tseg = 2.0
A = np.array([
    [0,0,0,0,0,1],
    [Tseg**5,Tseg**4,Tseg**3,Tseg**2,Tseg,1],
    [0,0,0,0,1,0],
    [5*Tseg**4,4*Tseg**3,3*Tseg**2,2*Tseg,1,0],
    [0,0,0,2,0,0],
    [20*Tseg**3,12*Tseg**2,6*Tseg,2,0,0],
], float)

def solve_quintic(q0, q1, qd0, qd1):
    b = np.array([q0, q1, qd0, qd1, 0, 0])
    return np.linalg.solve(A, b)

def eval_quintic(a, t):
    a5,a4,a3,a2,a1,a0 = a
    q   = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
    qd  = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
    qdd = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2
    return q, qd, qdd

# ============================================================
# Choose number of knots (increase to improve accuracy!)
# ============================================================
N_knots = 9        # try 9, 13, 17 …
N_circle = 200     # resolution of the true desired path

phi_vals = np.linspace(0, 2*np.pi, N_circle)
desired_points = generate_circle_points(N_circle)

# Knot index selection
idxs = np.linspace(0, N_circle-1, N_knots).astype(int)
times = np.linspace(0, 8, N_knots)

# ============================================================
# IK at knot points
# ============================================================
Q = []
for k in idxs:
    x, y, z = desired_points[k]
    qv = ik_solver(x, y, z, c=0.0)
    Q.append(qv)

Q = np.array(Q)

# ============================================================
# Approximate knot velocities using finite difference
# (more knots → better accuracy)
# ============================================================
Qdot = np.zeros_like(Q)
dt = times[1] - times[0]

for i in range(len(Q)):
    if i == 0:
        Qdot[i] = (Q[i+1] - Q[i]) / dt
    elif i == len(Q)-1:
        Qdot[i] = (Q[i] - Q[i-1]) / dt
    else:
        Qdot[i] = (Q[i+1] - Q[i-1]) / (2*dt)

# ============================================================
# Build ALL quintic segments between knots
# ============================================================
coeffs = []
for s in range(len(times)-1):
    seg_coeffs = []
    for j in range(4):
        a = solve_quintic(Q[s,j], Q[s+1,j], Qdot[s,j], Qdot[s+1,j])
        seg_coeffs.append(a)
    coeffs.append(seg_coeffs)

# ============================================================
# Evaluate 0–8 s trajectory
# ============================================================
t_fine = np.linspace(0, 8, 800)
q_traj = np.zeros((len(t_fine), 4))

for i, t in enumerate(t_fine):
    seg = min(int((t/8)* (len(times)-1)), len(coeffs)-1)
    τ = t - times[seg]
    for j in range(4):
        q_traj[i,j] = eval_quintic(coeffs[seg][j], τ)[0]

# ============================================================
# Compute EE path from FK
# ============================================================
ee_path = np.array([fk_end_effector(q) for q in q_traj])

# ============================================================
# Plot comparison
# ============================================================
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Desired circle
ax.plot(desired_points[:,0], desired_points[:,1], desired_points[:,2],
        'm--', label="Desired circle", linewidth=2)

# Actual interpolated path
ax.plot(ee_path[:,0], ee_path[:,1], ee_path[:,2],
        'b', label="Interpolated EE path", linewidth=2)

ax.scatter(desired_points[idxs,0],
           desired_points[idxs,1],
           desired_points[idxs,2],
           color='k', s=40, label="Knot points")

ax.set_title("End-effector path vs desired circular path")
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.legend()
ax.set_box_aspect([1,1,1])
plt.show()
