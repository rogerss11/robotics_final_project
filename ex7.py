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

# ------------------------------------------------------------
# FK helper (returns frame origins)
# ------------------------------------------------------------
def compute_positions(q_vec):
    subs = {q1: q_vec[0], q2: q_vec[1], q3: q_vec[2], q4: q_vec[3]}
    Tcurr = sp.eye(4)
    origins = [np.array([0.0, 0.0, 0.0])]
    for p in dh_params:
        Tcurr = Tcurr * DH(*p)
        Tn = np.array(Tcurr.evalf(subs=subs), dtype=float)
        origins.append(Tn[0:3, 3])
    return origins

# ------------------------------------------------------------
# Compare actual EE vs desired circle (both FK-based)
# ------------------------------------------------------------
# Desired (FK-based): convert tip point to frame-4 base for IK, then FK to get tip from DH chain
desired_positions = []
for p in points:
    x_t, y_t, z_t, c_t = p
    q_tmp = ik_solver(x_t - stylus_offset, y_t, z_t, c_t)
    origins_tmp = compute_positions(q_tmp)
    desired_positions.append(origins_tmp[-1])  # tip (thanks to DH last link)
desired_positions = np.array(desired_positions)

# Actual path from interpolated q(t)
actual_positions = []
for q in q_vals:
    origins = compute_positions(q)
    actual_positions.append(origins[-1])
actual_positions = np.array(actual_positions)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(desired_positions[:,0], desired_positions[:,1], desired_positions[:,2],
        'orange', label='Desired EE path (FK-based)', linewidth=2)
ax.plot(actual_positions[:,0], actual_positions[:,1], actual_positions[:,2],
        'b--', label='Interpolated EE path (quintic)', linewidth=2)
ax.scatter(desired_positions[0,0], desired_positions[0,1], desired_positions[0,2],
           c='green', s=50, label='Start')
ax.scatter(desired_positions[-1,0], desired_positions[-1,1], desired_positions[-1,2],
           c='red', s=50, label='End')
ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
ax.set_title("Comparison of End-Effector Path\nDesired (Circle) vs. Interpolated (Quintic)")
ax.legend(); ax.view_init(elev=25, azim=45); ax.set_box_aspect([1,1,1])
# Consistent axes/view
ax.set_xlim([0, 250]); ax.set_ylim([-150, 150]); ax.set_zlim([0, 250])
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=45)
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# Robot animation following interpolated q(t)
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.28)

# Base frame
world_len = 30
ax.quiver(0, 0, 0, 1, 0, 0, color="r", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, color="g", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, color="b", length=world_len, normalize=True)
ax.text(35, 0, 0, "X₀", color="r")
ax.text(0, 35, 0, "Y₀", color="g")
ax.text(0, 0, 35, "Z₀", color="b")

# Show target circle (FK-based)
tip_positions = []
for p in points:
    x_t, y_t, z_t, c_t = p
    q_tmp = ik_solver(x_t - stylus_offset, y_t, z_t, c_t)
    origins_tmp = compute_positions(q_tmp)
    tip_positions.append(origins_tmp[-1])
tip_positions = np.array(tip_positions)
ax.scatter(tip_positions[:,0], tip_positions[:,1], tip_positions[:,2],
           c="orange", s=20, label="Target circle (desired)")

# Initial pose
q_init = q_vals[0]
origins = compute_positions(q_init)

# Draw links
link_lines = []
for i in range(len(origins) - 1):
    p1, p2 = origins[i], origins[i + 1]
    (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      "k--", linewidth=1.2)
    link_lines.append(line)

# Frame axes at each joint (unit directions shown with length=20)
frame_quivers = []
for i in range(1, len(origins)):
    o = origins[i]
    qx = ax.quiver(o[0], o[1], o[2], 1, 0, 0, color="r", length=20, normalize=True)
    qy = ax.quiver(o[0], o[1], o[2], 0, 1, 0, color="g", length=20, normalize=True)
    qz = ax.quiver(o[0], o[1], o[2], 0, 0, 1, color="b", length=20, normalize=True)
    frame_quivers.append((qx, qy, qz))

# Joint values display
joint_display = ax.text2D(0.98, 0.95, "", transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Axis/view
ax.set_xlim([0, 250]); ax.set_ylim([-150, 150]); ax.set_zlim([0, 250])
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
ax.set_title("Ex6: Robot following interpolated (quintic) EE path")
ax.legend(); ax.view_init(elev=25, azim=45); ax.set_box_aspect([1, 1, 1])

# Animate
actual_path = []
n_frames = len(q_vals)
for k in range(n_frames):
    q_now = q_vals[k]
    origins = compute_positions(q_now)
    actual_path.append(origins[-1])

    # Update links
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i + 1]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])

    # Update frames
    for (qx, qy, qz) in frame_quivers:
        qx.remove(); qy.remove(); qz.remove()
    frame_quivers.clear()
    for i in range(1, len(origins)):
        o = origins[i]
        qx = ax.quiver(o[0], o[1], o[2], 1, 0, 0, color="r", length=20, normalize=True)
        qy = ax.quiver(o[0], o[1], o[2], 0, 1, 0, color="g", length=20, normalize=True)
        qz = ax.quiver(o[0], o[1], o[2], 0, 0, 1, color="b", length=20, normalize=True)
        frame_quivers.append((qx, qy, qz))

    # Update joint text
    joint_text = (f"q1 = {q_now[0]:.3f} rad\n"
                  f"q2 = {q_now[1]:.3f} rad\n"
                  f"q3 = {q_now[2]:.3f} rad\n"
                  f"q4 = {q_now[3]:.3f} rad")
    joint_display.set_text(joint_text)

    # Update EE trace
    actual_arr = np.array(actual_path)
    if k == 0:
        path_line, = ax.plot(actual_arr[:,0], actual_arr[:,1], actual_arr[:,2],
                             "b-", linewidth=2, label="Interpolated EE path")
    else:
        path_line.set_data(actual_arr[:,0], actual_arr[:,1])
        path_line.set_3d_properties(actual_arr[:,2])

    plt.pause(0.03)  # ~30 ms per frame

plt.show()

# ------------------------------------------------------------
# Final: Display both circles only (FK-based, no robot)
# ------------------------------------------------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Desired circular path (FK-based)
ax.plot(desired_positions[:,0], desired_positions[:,1], desired_positions[:,2],
        color="orange", linewidth=2, label="Desired EE circle (FK-based)")

# Actual interpolated EE path
actual_arr = np.array(actual_path)
ax.plot(actual_arr[:,0], actual_arr[:,1], actual_arr[:,2],
        "b--", linewidth=2, label="Interpolated EE circle")

# Start/End markers
ax.scatter(desired_positions[0,0], desired_positions[0,1], desired_positions[0,2],
           c='green', s=50, label='Start (t=0)')
ax.scatter(desired_positions[-1,0], desired_positions[-1,1], desired_positions[-1,2],
           c='red', s=50, label='End (t=8 s)')

# Consistent axes/view
ax.set_xlim([0, 250]); ax.set_ylim([-150, 150]); ax.set_zlim([0, 250])
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=45)

ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
ax.set_title("Desired vs. Interpolated EE Circles (No Robot)")
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.show()
