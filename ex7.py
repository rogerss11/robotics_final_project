import numpy as np
import matplotlib.pyplot as plt

# === IMPORT HELPERS ===
from functions import (
    ik_solver,
    generate_circle_points,
    compute_T0i,
    eval_quintic,
    solve_quintic,
    geometric_jacobian,
)

from simulation import set_axes_equal

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

# ============================================================
# Choose number of knots
# ============================================================
N_knots = 5
V_knots = [
    np.array([ 0,   0,   0 ]),   # start: stop
    np.array([ 0, -27,   0 ]),   # moving -y
    np.array([ 0,   0, -27 ]),   # moving -z
    np.array([ 0,  27,   0 ]),   # moving +y
    np.array([ 0,   0,   0 ]),   # end: stop
]
OMEGA = np.zeros(3)  # no rotational velocity

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
Qdot = np.zeros_like(Q)

# ============================================================
# 3) Compute joint velocities Qdot from Jacobian (instead of finite difference)
# ============================================================

# Cartesian velocities of the tip at each knot (mm/s)
# These define motion tangential to the circle:

for k in range(len(times)):
    # compute Jacobian at joint configuration Q[k]
    J = geometric_jacobian(Q[k], 4)   # 6×4
    
    # desired Cartesian velocity of the TCP at this knot
    xdot = np.hstack([V_knots[k], OMEGA])   # shape (6,)
    
    # compute qdot using pseudoinverse
    qdot = np.linalg.pinv(J) @ xdot        # shape (4,)
    Qdot[k] = qdot


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
set_axes_equal(ax, equal=True)
plt.show()
