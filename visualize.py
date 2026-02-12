import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ------------------------------------------------------------
# DH, FK, Jacobian, IK definitions (same base as ex5)
# ------------------------------------------------------------
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")
dh_params = [
    (0, sp.pi/2, 50, q1),
    (93, 0, 0, q2),
    (93, 0, 0, q3),
    (50, sp.pi/2, 0, q4)   # stylus link: 50 mm along x4
]

def forward_transforms():
    Ts = [sp.eye(4)]
    T = sp.eye(4)
    for p in dh_params:
        T = T * DH(*p)
        Ts.append(sp.simplify(T))
    return Ts

Ts = forward_transforms()

T04 = Ts[-1]

# numeric FK for frame 4
def fk_numeric(qv):
    subs = {q1: qv[0], q2: qv[1], q3: qv[2], q4: qv[3]}
    Tn = np.array(T04.evalf(subs=subs), dtype=float)
    return Tn

# TIP position = frame 4 + stylus offset along x4 (50 mm)
def tip_position(qv):
    Tn = fk_numeric(qv)
    p4 = Tn[0:3, 3]
    x4 = Tn[0:3, 0]   # x-axis of frame 4
    return p4 + 50.0 * x4


def geometric_jacobian(Ts, end_index):
    # end_index = 4 for frame-4 origin (before the tip offset is implicitly included by DH)
    o_n = Ts[end_index][0:3, 3]
    Jv_cols, Jw_cols = [], []
    for i in range(1, 5):
        z_im1 = Ts[i-1][0:3, 2]
        o_im1 = Ts[i-1][0:3, 3]
        Jv_cols.append(sp.Matrix.cross(z_im1, o_n - o_im1))
        Jw_cols.append(z_im1)
    return sp.simplify(sp.Matrix.vstack(sp.Matrix.hstack(*Jv_cols),
                                        sp.Matrix.hstack(*Jw_cols)))

J4_sym = geometric_jacobian(Ts, 4)

# Analytic IK (for frame-4 base position)
a1, a2, a3, d1 = 0, 93, 93, 50
def ik_solver(x, y, z, c):
    # x,y,z are for frame-4 base position (not the tip). c = sin(tilt) ≈ 0 for horizontal stylus
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

def J4_numeric(qv):
    subs = {q1: qv[0], q2: qv[1], q3: qv[2], q4: qv[3]}
    return np.array(J4_sym.evalf(subs=subs), dtype=float)

# ------------------------------------------------------------
# EE circular path definition (stylus tip path)
# ------------------------------------------------------------
R = 32.0
p_c = np.array([150.0, 0.0, 120.0])
phis = np.linspace(0, 2*np.pi, 37)
points = []
for phi in phis:
    p = p_c + R * np.array([0.0, np.cos(phi), np.sin(phi)])  # tip trajectory in world
    x, y, z = p
    c = 0.0  # horizontal stylus
    points.append([x, y, z, c])
points = np.array(points)

idxs = [0, 9, 18, 27, 36]
times = np.array([0, 2, 4, 6, 8])

# ------------------------------------------------------------
# Joint positions at knots (account for stylus offset)
# Convert desired TIP position -> frame-4 base by subtracting 50 mm along x
# ------------------------------------------------------------
stylus_offset = 50.0  # mm along x4 (≈ world x here since c=0)
Q = []
for i in idxs:
    x_t, y_t, z_t, c_t = points[i]
    x_base = x_t - stylus_offset
    qv = ik_solver(x_base, y_t, z_t, c_t)
    Q.append(qv)
Q = np.array(Q)

# ------------------------------------------------------------
# EE linear velocities at knots (tip path) and joint velocities via Jacobian
# (We approximate using the same linear velocities for the frame-4 origin.)
# ------------------------------------------------------------
V_knots = [
    np.array([0.0,   0.0,  0.0]),
    np.array([0.0, -27.0,  0.0]),
    np.array([0.0,   0.0, -27.0]),
    np.array([0.0,  27.0,  0.0]),
    np.array([0.0,   0.0,  0.0])
]
OMEGA = np.zeros(3)
Qdot = np.zeros_like(Q)
for k in range(len(times)):
    J = J4_numeric(Q[k])
    xdot = np.hstack([V_knots[k], OMEGA]).reshape(6, 1)
    qdot = np.linalg.pinv(J) @ xdot
    Qdot[k] = qdot.flatten()

Qddot = np.zeros_like(Q)  # zero accelerations at knots

# ------------------------------------------------------------
# Quintic interpolation coefficients per segment and joint
# ------------------------------------------------------------
Tseg = 2.0
A = np.array([
    [0, 0, 0, 0, 0, 1],
    [Tseg**5, Tseg**4, Tseg**3, Tseg**2, Tseg, 1],
    [0, 0, 0, 0, 1, 0],
    [5*Tseg**4, 4*Tseg**3, 3*Tseg**2, 2*Tseg, 1, 0],
    [0, 0, 0, 2, 0, 0],
    [20*Tseg**3, 12*Tseg**2, 6*Tseg, 2, 0, 0]
], dtype=float)

def solve_quintic(q0, q1, qd0, qd1):
    b = np.array([q0, q1, qd0, qd1, 0.0, 0.0])
    return np.linalg.solve(A, b)

coeffs = []
for s in range(len(times) - 1):
    seg = [solve_quintic(Q[s, j], Q[s+1, j], Qdot[s, j], Qdot[s+1, j]) for j in range(4)]
    coeffs.append(seg)

# ------------------------------------------------------------
# Print quintic polynomial coefficients per joint and per segment
# Segments: 0-2, 2-4, 4-6, 6-8 (seconds)
# Each coeff array is [a5, a4, a3, a2, a1, a0] for q(t) = a5*t^5 + ... + a0
# ------------------------------------------------------------
# Print only coefficient matrices: one 4x6 matrix per segment (rows = joints 1..4,
# columns = [a5, a4, a3, a2, a1, a0]). No extra text.
np.set_printoptions(precision=3, suppress=False)
for s, seg in enumerate(coeffs):
    t0 = times[s]
    t1 = times[s+1]
    print(f"Segment coefficients for t = {t0} to {t1} s:")
    print("Row = joint (1..4), Columns = [a5, a4, a3, a2, a1, a0]")
    mat = np.array(seg)  # shape (4,6)
    print(mat)
    print()

# ------------------------------------------------------------
# Evaluate q(t), qdot(t), qddot(t) over 0..8 s
# ------------------------------------------------------------
def eval_quintic(a, t):
    a5, a4, a3, a2, a1, a0 = a
    q = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
    qd = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
    qdd = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2
    return q, qd, qdd

t_fine = np.linspace(0, 8, 400)
q_vals = np.zeros((len(t_fine), 4))
qd_vals = np.zeros((len(t_fine), 4))
qdd_vals = np.zeros((len(t_fine), 4))

for k, t in enumerate(t_fine):
    seg_idx = min(int(t // 2), 3)
    t_local = t - 2*seg_idx
    for j in range(4):
        q, qd, qdd = eval_quintic(coeffs[seg_idx][j], t_local)
        q_vals[k, j], qd_vals[k, j], qdd_vals[k, j] = q, qd, qdd

tip_traj = np.array([tip_position(q_vals[k]) for k in range(len(t_fine))])

# ------------------------------------------------------------
#  Full robot animation (same visualization style as Task 3)
# ------------------------------------------------------------

# Reuse FK chain from symbolic transforms
def compute_positions_numeric(q_vals):
    subs = {q1: q_vals[0], q2: q_vals[1], q3: q_vals[2], q4: q_vals[3]}
    Tcurr = sp.eye(4)
    origins = [np.array([0, 0, 0])]
    x_axes, y_axes, z_axes = [], [], []
    for p in dh_params:
        Tcurr = Tcurr * DH(*p)
        Tn = np.array(Tcurr.evalf(subs=subs), dtype=float)
        origins.append(Tn[0:3, 3])
        x_axes.append(Tn[0:3, 0])
        y_axes.append(Tn[0:3, 1])
        z_axes.append(Tn[0:3, 2])
    return origins, x_axes, y_axes, z_axes

# Create figure
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim([0, 250])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 250])
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_title("Task 6: Stylus tip following circular path")

# Draw desired circle
ax.plot(points[:,0], points[:,1], points[:,2], 'r--', label="Desired circle")

# Initial pose
origins, x_axes, y_axes, z_axes = compute_positions_numeric(q_vals[0])

# Draw links
link_lines = []
for i in range(len(origins)-1):
    p1, p2 = origins[i], origins[i+1]
    (line,) = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'k-',lw=2)
    link_lines.append(line)

# Draw frames
frame_quivers = []
for i in range(1,len(origins)):
    o = origins[i]
    qx = ax.quiver(o[0], o[1], o[2], *x_axes[i-1], color='r', length=20)
    qy = ax.quiver(o[0], o[1], o[2], *y_axes[i-1], color='g', length=20)
    qz = ax.quiver(o[0], o[1], o[2], *z_axes[i-1], color='b', length=20)
    frame_quivers.append((qx,qy,qz))

# Stylus tip dot
tip_dot, = ax.plot([],[],[],'bo',markersize=6,label="Stylus tip")
ax.legend()

# Animation loop
for k in range(0, len(t_fine), 4):

    origins, x_axes, y_axes, z_axes = compute_positions_numeric(q_vals[k])

    # Update link positions
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i+1]
        line.set_data([p1[0], p2[0]],[p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])

    # Update frames
    for (qx,qy,qz) in frame_quivers:
        qx.remove(); qy.remove(); qz.remove()
    frame_quivers.clear()

    for i in range(1,len(origins)):
        o = origins[i]
        qx = ax.quiver(o[0], o[1], o[2], *x_axes[i-1], color='r', length=20)
        qy = ax.quiver(o[0], o[1], o[2], *y_axes[i-1], color='g', length=20)
        qz = ax.quiver(o[0], o[1], o[2], *z_axes[i-1], color='b', length=20)
        frame_quivers.append((qx,qy,qz))

    # Update stylus TIP position
    tip_p = origins[-1] + 50.0 * x_axes[-1]
    tip_dot.set_data([tip_p[0]],[tip_p[1]])
    tip_dot.set_3d_properties([tip_p[2]])

    plt.pause(0.02)

plt.show()

