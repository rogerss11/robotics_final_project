import numpy as np
import matplotlib.pyplot as plt

# ===== IMPORT ALL HELPERS =====
from functions import (
    ik_solver,
    generate_circle_points,
    geometric_jacobian,
)

# ============================================================
# 1) Circle path (stylus tip)
# ============================================================
N = 37
phis = np.linspace(0, 2*np.pi, N)
points = generate_circle_points(N)        # → (N, 3)

# four knot points at 0,2,4,6,8 seconds
idxs  = [0, 9, 18, 27, 36]
times = np.array([0, 2, 4, 6, 8])

# ============================================================
# 2) IK at knots (adjust for stylus 50 mm offset)
# ============================================================
Q = []
for k in idxs:
    x_t, y_t, z_t = points[k]
    x4 = x_t
    qv = ik_solver(x4, y_t, z_t, c=0.0)
    Q.append(qv)

Q = np.array(Q)      # shape (5,4)

# ============================================================
# 3) Compute joint velocities Qdot from Jacobian
# ============================================================
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

# ============================================================
# 4) Quintic polynomial coefficients
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

coeffs = [
    [solve_quintic(Q[s,j], Q[s+1,j], Qdot[s,j], Qdot[s+1,j]) for j in range(4)]
    for s in range(4)
]

# ============================================================
# 5) Evaluate over full 0–8 s
# ============================================================
def eval_quintic(a, t):
    a5,a4,a3,a2,a1,a0 = a
    q   = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
    qd  = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
    qdd = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2
    return q, qd, qdd

t_fine = np.linspace(0, 8, 400)
q_vals   = np.zeros((400,4))
qd_vals  = np.zeros((400,4))
qdd_vals = np.zeros((400,4))

for i,t in enumerate(t_fine):
    seg = min(int(t//2), 3)
    τ   = t - seg*2
    for j in range(4):
        q_vals[i,j], qd_vals[i,j], qdd_vals[i,j] = eval_quintic(coeffs[seg][j], τ)

# ============================================================
# 6) Plot
# ============================================================
lbl = [r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$"]

plt.figure(figsize=(10,10))

plt.subplot(3,1,1)
for j in range(4): plt.plot(t_fine, q_vals[:,j])
for j in range(4): plt.scatter(times, Q[:,j], c='k')
plt.title("Joint positions")
plt.ylabel("rad"); plt.grid(True)

plt.subplot(3,1,2)
for j in range(4): plt.plot(t_fine, qd_vals[:,j])
for j in range(4): plt.scatter(times, Qdot[:,j], c='k')
plt.title("Joint velocities"); plt.ylabel("rad/s"); plt.grid(True)

plt.subplot(3,1,3)
for j in range(4): plt.plot(t_fine, qdd_vals[:,j])
plt.title("Joint accelerations"); plt.ylabel("rad/s²"); plt.xlabel("t [s]")
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# Print quintic interpolation coefficients for segments A–D
# ============================================================

seg_names = ["A (0–2s)", "B (2–4s)", "C (4–6s)", "D (6–8s)"]

for s in range(4):
    print(f"\n=== Segment {seg_names[s]} ===")
    for j in range(4):
        a = coeffs[s][j]
        print(f"  Joint q{j+1}:  [{a[0]: .3f}, {a[1]: .3f}, {a[2]: .3f}, "
              f"{a[3]: .3f}, {a[4]: .3f}, {a[5]: .3f}]")
