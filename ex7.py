# ex7.py — Plot actual vs desired end-effector path for interpolated trajectory (Problem 7)
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ============================================================
#  DH, IK, and FK (copied from ex6 for independence)
# ============================================================
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")
dh_params = [
    (0,  sp.pi/2, 50, q1),
    (93, 0,       0,  q2),
    (93, 0,       0,  q3),
    (50, sp.pi/2, 0,  q4)
]

def forward_transforms():
    Ts = [sp.eye(4)]
    T = sp.eye(4)
    for p in dh_params:
        T = T * DH(*p)
        Ts.append(sp.simplify(T))
    return Ts  # T0..T4

Ts_sym = forward_transforms()
T04_sym = Ts_sym[4]

# Inverse kinematics (same as before)
a1, a2, a3, d1 = 0, 93, 93, 50
def ik_solver(x, y, z, c):
    q1v = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) - a1
    s = z - d1
    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c3 = np.clip(c3, -1.0, 1.0)
    q3v = np.arctan2(np.sqrt(1 - c3**2), c3)
    q2v = np.arctan2(s, r) - np.arctan2(a3*np.sin(q3v), a2 + a3*np.cos(q3v))
    c = np.clip(c, -1.0, 1.0)
    q4v = np.arcsin(c) - (q2v + q3v)
    return np.array([q1v, q2v, q3v, q4v])

# ============================================================
#  Load same trajectory parameters from ex6 (hardcode or recompute)
# ============================================================
# Circle definition (desired path)
R = 32.0
p_c = np.array([150.0, 0.0, 120.0])

# Knot configuration generation (same as ex6)
phis = np.linspace(0, 2*np.pi, 37)
idxs = [0, 9, 18, 27, 36]
def circle_point(phi):
    return p_c + R*np.array([0.0, np.cos(phi), np.sin(phi)])

# ------------------------------------------------------------
# Quintic interpolation (reuse from ex6)
# ------------------------------------------------------------
def quintic_coeffs(q0, q1, dq0, dq1, ddq0, ddq1, T=2.0):
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [T5, T4, T3, T2, T, 1],
        [5*T4, 4*T3, 3*T2, 2*T, 1, 0],
        [20*T3, 12*T2, 6*T, 2, 0, 0]
    ])
    b = np.array([q0, dq0, ddq0, q1, dq1, ddq1])
    return np.linalg.solve(M, b)

# Simplify by loading stored data from ex6 or recomputing approximate joint angles.
# We'll recompute knot q values using IK (zero velocities, zero acc)
q_knots = []
for idx in idxs:
    phi = phis[idx]
    x, y, z = circle_point(phi)
    qv = ik_solver(x, y, z, 0.0)
    q_knots.append(qv)
q_knots = np.array(q_knots)
qdot_knots = np.zeros_like(q_knots)
qddot_knots = np.zeros_like(q_knots)

# Segments and coefficient matrices
segments = [('A', 0, 1), ('B', 1, 2), ('C', 2, 3), ('D', 3, 4)]
Tseg = 2.0
seg_mats = []
for name, i0, i1 in segments:
    M = np.zeros((4, 6))
    for j in range(4):
        a = quintic_coeffs(q_knots[i0, j], q_knots[i1, j],
                           qdot_knots[i0, j], qdot_knots[i1, j],
                           qddot_knots[i0, j], qddot_knots[i1, j], Tseg)
        M[j, :] = a
    seg_mats.append(M)

# Evaluate polynomial
def eval_poly_row(a, t):
    t = np.asarray(t)
    q  = a[0]*t**5 + a[1]*t**4 + a[2]*t**3 + a[3]*t**2 + a[4]*t + a[5]
    return q

def concat_segments(mats, T=2.0, dt=0.02):
    times = []
    q_all = []
    t0 = 0.0
    for M in mats:
        t = np.arange(0.0, T+1e-9, dt)
        times.append(t + t0)
        q_seg = []
        for j in range(4):
            q = eval_poly_row(M[j], t)
            q_seg.append(q)
        q_all.append(np.stack(q_seg, axis=0))
        t0 += T
    t_full = np.concatenate(times)
    q_full = np.concatenate(q_all, axis=1)
    return t_full, q_full

t_full, q_traj = concat_segments(seg_mats, T=Tseg, dt=0.02)

# ============================================================
#  Compute actual end-effector path
# ============================================================
def fk_numeric(q_vals):
    subs = {q1: q_vals[0], q2: q_vals[1], q3: q_vals[2], q4: q_vals[3]}
    T_num = np.array(T04_sym.evalf(subs=subs), dtype=float)
    return T_num[:3, 3]

p_actual = np.zeros((len(t_full), 3))
for k in range(len(t_full)):
    qv = q_traj[:, k]
    p_actual[k, :] = fk_numeric(qv)

# Desired exact circular path (Problem 3)
phi_exact = np.linspace(0, 2*np.pi, 400)
circle_exact = np.array([circle_point(phi) for phi in phi_exact])

# ============================================================
#  Plot comparison
# ============================================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot desired circular path
ax.plot(circle_exact[:, 0], circle_exact[:, 1], circle_exact[:, 2],
        'k--', lw=1.5, label="Desired circular path")

# Plot actual end-effector path
ax.plot(p_actual[:, 0], p_actual[:, 1], p_actual[:, 2],
        'r', lw=2, label="Actual end-effector trajectory")

# Style
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_xlim([100, 200])
ax.set_ylim([-50, 50])
ax.set_zlim([80, 160])
ax.set_title("Problem 7 — End-effector trajectory vs desired circular path")
ax.legend()
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()
