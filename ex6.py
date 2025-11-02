# ex6.py — Quintic joint trajectory (Problem 6)
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ---------------------------
# Kinematics (from ex4/ex5)
# ---------------------------
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

def geometric_jacobian(Ts, end_index=4):
    """6x4 geometric Jacobian at frame {end_index}."""
    o_n = Ts[end_index][0:3, 3]
    Jv_cols, Jw_cols = [], []
    for i in range(1, 5):                           # 4 revolute joints
        z_im1 = Ts[i-1][0:3, 2]
        o_im1 = Ts[i-1][0:3, 3]
        Jv_cols.append(sp.Matrix.cross(z_im1, o_n - o_im1))
        Jw_cols.append(z_im1)
    Jv = sp.Matrix.hstack(*Jv_cols)
    Jw = sp.Matrix.hstack(*Jw_cols)
    return sp.simplify(sp.Matrix.vstack(Jv, Jw))

J4_sym = geometric_jacobian(Ts_sym, 4)

# IK (from ex2/ex3)
a1, a2, a3, d1 = 0, 93, 93, 50
def ik_solver(x, y, z, c):
    """Analytic IK. c = x4_z = sin(q2+q3+q4)."""
    q1v = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) - a1
    s = z - d1
    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2*a2*a3)
    c3 = np.clip(c3, -1.0, 1.0)
    q3v = np.arctan2(np.sqrt(1 - c3**2), c3)  # elbow-down
    q2v = np.arctan2(s, r) - np.arctan2(a3*np.sin(q3v), a2 + a3*np.cos(q3v))
    c = np.clip(c, -1.0, 1.0)
    q4v = np.arcsin(c) - (q2v + q3v)
    return np.array([q1v, q2v, q3v, q4v])

# ---------------------------
# Knot points (same circle as ex3)
# ---------------------------
R = 32.0
p_c = np.array([150.0, 0.0, 120.0])
phis = np.linspace(0, 2*np.pi, 37)  # 0..36
idxs = [0, 9, 18, 27, 36]           # φ0, φ9, φ18, φ27, φ36

def circle_point(phi):
    # circle in YZ plane, centered at p_c, radius R
    return p_c + R*np.array([0.0, np.cos(phi), np.sin(phi)])

# End-effector linear velocities at knots (mm/s), ω = 0
# From the statement image:
# v(tA=0)=[0,0,0], v(tA=2)=v(tB=0)=[0,-27,0],
# v(tB=2)=v(tC=0)=[0,0,-27], v(tC=2)=v(tD=0)=[0,27,0], v(tD=2)=[0,0,0]
v_list = [
    np.array([0.,   0.,   0.  ]),   # at φ0          (knot 0)
    np.array([0., -27.,  0.  ]),    # at φ9          (knot 1)
    np.array([0.,  0., -27. ]),     # at φ18         (knot 2)
    np.array([0., 27.,  0.  ]),     # at φ27         (knot 3)
    np.array([0.,  0.,  0.  ])      # at φ36 (=φ0)   (knot 4)
]
omega_list = [np.array([0., 0., 0.]) for _ in range(5)]  # stylus kept horizontal

# Compute q, qdot, qddot (acc=0) at each knot
q_knots   = []
qdot_knots = []
qddot_knots = []

for k, idx in enumerate(idxs):
    phi = phis[idx]
    x, y, z = circle_point(phi)
    c = 0.0                                   # keep x4_z = 0  (horizontal stylus)
    qv = ik_solver(x, y, z, c)                 # joint position
    subs = {q1: qv[0], q2: qv[1], q3: qv[2], q4: qv[3]}
    J4_num = np.array(J4_sym.evalf(subs=subs), dtype=float)

    v = v_list[k].reshape(3, 1)
    omega = omega_list[k].reshape(3, 1)
    xdot = np.vstack((v, omega))               # 6x1

    qdot = np.linalg.pinv(J4_num) @ xdot       # 4x1
    q_knots.append(qv)
    qdot_knots.append(qdot.flatten())
    qddot_knots.append(np.zeros(4))            # per Hint 2

q_knots = np.array(q_knots)         # shape (5,4)
qdot_knots = np.array(qdot_knots)   # shape (5,4)
qddot_knots = np.array(qddot_knots) # zeros

# ---------------------------
# Quintic trajectory per segment/joint
# ---------------------------
def quintic_coeffs(q0, q1, dq0, dq1, ddq0, ddq1, T=2.0):
    """
    Solve for coefficients a5..a0 of:
      q(t) = a5 t^5 + a4 t^4 + a3 t^3 + a2 t^2 + a1 t + a0,   t in [0, T]
    with boundary conditions on q, dq, ddq at t=0 and t=T.
    Returns coefficients [a5, a4, a3, a2, a1, a0].
    """
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [    0,     0,     0,     0,    0, 1],
        [    0,     0,     0,     0,  1.0, 0],
        [    0,     0,     0,   2.0,    0, 0],
        [  T5,   T4,   T3,   T2,    T, 1],
        [5*T4, 4*T3, 3*T2, 2*T,  1.0, 0],
        [20*T3,12*T2, 6*T,  2.0,   0, 0]
    ], dtype=float)
    b = np.array([q0, dq0, ddq0, q1, dq1, ddq1], dtype=float)
    a = np.linalg.solve(M, b)
    return a  # [a5..a0]

# Four segments: A: 0→1, B: 1→2, C: 2→3, D: 3→4  (each T=2 s)
segments = [('A', 0, 1), ('B', 1, 2), ('C', 2, 3), ('D', 3, 4)]
Tseg = 2.0

# Coefficient matrices per segment (4 joints × 6 coeffs)
A_mat = np.zeros((4, 6)); B_mat = np.zeros((4, 6))
C_mat = np.zeros((4, 6)); D_mat = np.zeros((4, 6))
seg_mats = {'A':A_mat, 'B':B_mat, 'C':C_mat, 'D':D_mat}

for name, i0, i1 in segments:
    M = seg_mats[name]
    for j in range(4):  # joint index
        a = quintic_coeffs(q_knots[i0, j], q_knots[i1, j],
                           qdot_knots[i0, j], qdot_knots[i1, j],
                           qddot_knots[i0, j], qddot_knots[i1, j],
                           T=Tseg)
        M[j, :] = a  # a5..a0

# Pretty print coefficients
def print_coeffs(title, M):
    print(f"\n{title} coefficients (rows q1..q4; columns a5 a4 a3 a2 a1 a0):")
    with np.printoptions(precision=6, suppress=True):
        print(M)

print_coeffs("Segment A (q^(0)→q^(9))", A_mat)
print_coeffs("Segment B (q^(9)→q^(18))", B_mat)
print_coeffs("Segment C (q^(18)→q^(27))", C_mat)
print_coeffs("Segment D (q^(27)→q^(36))", D_mat)

# ---------------------------
# Evaluate & plot trajectory
# ---------------------------
def eval_poly_row(a, t):
    """Given coeffs a=[a5..a0], return q, dq, ddq at time array t."""
    t = np.asarray(t)
    q  = (((a[0]*t + a[1])*t + a[2])*t + a[3])*t**2 + a[4]*t + a[5]  # robust Horner-ish
    dq = (5*a[0]*t**4 + 4*a[1]*t**3 + 3*a[2]*t**2 + 2*a[3]*t + a[4])
    ddq = (20*a[0]*t**3 + 12*a[1]*t**2 + 6*a[2]*t + 2*a[3])
    return q, dq, ddq

def concat_segments(mats, T=2.0, dt=0.01):
    """Concatenate A,B,C,D into full 0–8s arrays for q,dq,ddq per joint."""
    times = []
    q_all = []; dq_all = []; ddq_all = []
    t0 = 0.0
    for M in mats:
        t = np.arange(0.0, T+1e-12, dt)
        times.append(t + t0)
        q_seg = []; dq_seg = []; ddq_seg = []
        for j in range(4):
            q, dq, ddq = eval_poly_row(M[j], t)
            q_seg.append(q); dq_seg.append(dq); ddq_seg.append(ddq)
        q_all.append(np.stack(q_seg, axis=0))
        dq_all.append(np.stack(dq_seg, axis=0))
        ddq_all.append(np.stack(ddq_seg, axis=0))
        t0 += T
    t_full = np.concatenate(times)
    q_full = np.concatenate(q_all, axis=1)
    dq_full = np.concatenate(dq_all, axis=1)
    ddq_full = np.concatenate(ddq_all, axis=1)
    return t_full, q_full, dq_full, ddq_full

t, q_traj, dq_traj, ddq_traj = concat_segments([A_mat, B_mat, C_mat, D_mat], T=Tseg, dt=0.01)

# Plot
labels = [r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$"]
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for j in range(4):
    axes[0].plot(t, q_traj[j], label=labels[j])
    axes[1].plot(t, dq_traj[j], label=labels[j])
    axes[2].plot(t, ddq_traj[j], label=labels[j])

axes[0].set_ylabel("q [rad]")
axes[1].set_ylabel("q̇ [rad/s]")
axes[2].set_ylabel("q̈ [rad/s²]")
axes[2].set_xlabel("time [s]")

for ax in axes:
    ax.grid(True, alpha=0.3)
axes[0].legend(ncol=4, fontsize=9, loc="upper right")
fig.suptitle("Problem 6 — Quintic joint trajectories over 4 segments (T=2 s each)")
plt.tight_layout()
plt.show()
