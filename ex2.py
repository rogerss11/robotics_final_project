"""
Inverse Kinematics for the 4-DOF Manipulator
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ------------------------------------------------------------
#  DH helpers and symbolic forward kinematics (same as ex1)
# ------------------------------------------------------------
def DH(a, alpha, d, theta):
    """Standard Denavit–Hartenberg homogeneous transformation."""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH parameters (same as ex1)
dh_params = [
    (0, sp.pi/2, 50, sp.Symbol('q1')),
    (93, 0, 0, sp.Symbol('q2')),
    (93, 0, 0, sp.Symbol('q3')),
    (50, sp.pi/2, 0, sp.Symbol('q4'))
]

# Symbolic FK for base → stylus
T = sp.eye(4)
for p in dh_params:
    T = T * DH(*p)
T04 = sp.simplify(T)
q1, q2, q3, q4 = [p[3] for p in dh_params]

# ------------------------------------------------------------
#  Inverse kinematics solver  (Problem 2)
# ------------------------------------------------------------
a1, a2, a3, d1 = 0, 93, 93, 50  # link lengths and base height [mm]

def ik_solver_tip(x, y, z, c):
    """
    Inverse kinematics when (x, y, z) is the stylus tip (O4),
    not the wrist (O3).
    """
    a2, a3, d1, d4 = 93, 93, 50, 50  # mm
    q1 = np.arctan2(y, x)

    # orientation constraint
    c = np.clip(c, -1.0, 1.0)
    tilt = np.arcsin(c)  # total tilt angle = q2 + q3 + q4

    # we don't yet know q2+q3, so estimate wrist center O3
    # r = horizontal dist, s = vertical dist
    r = np.sqrt(x**2 + y**2)
    s = z - d1

    # subtract wrist offset d4 in direction of stylus x-axis
    rw = r - d4 * np.cos(tilt)
    sw = s - d4 * np.sin(tilt)

    # solve planar 2-link geometry
    c3 = (rw**2 + sw**2 - a2**2 - a3**2) / (2*a2*a3)
    c3 = np.clip(c3, -1.0, 1.0)
    q3 = np.arctan2(np.sqrt(1 - c3**2), c3)   # elbow-down
    q2 = np.arctan2(sw, rw) - np.arctan2(a3*np.sin(q3), a2 + a3*np.cos(q3))
    q4 = tilt - (q2 + q3)

    return np.array([q1, q2, q3, q4])


def ik_solver(x, y, z, c):
    """
    Analytic inverse kinematics for the 4-DOF manipulator.

    Inputs:
        x, y, z : end-effector position in {0} [mm]
        c       : z-component of x₄ axis (orientation constraint)

    Output:
        q = [q1, q2, q3, q4]  (radians)
    """
    # --- 1. Base rotation q1 (azimuth angle around z₀) ---
    q1 = np.arctan2(y, x)

    # --- 2. Planar reduction (r–s plane of the arm) ---
    # r = horizontal distance from joint 2 to wrist target
    # s = vertical distance from joint 2 to wrist target
    r = np.sqrt(x**2 + y**2) - a1
    s = z - d1

    # --- 3. Elbow joint q3 (law of cosines) ---
    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c3 = np.clip(c3, -1.0, 1.0)                     # numerical safety
    q3 = np.arctan2(np.sqrt(1 - c3**2), c3)         # choose elbow-down branch

    # --- 4. Shoulder joint q2 ---
    # From triangle geometry (same as Lecture 3 slides)
    q2 = np.arctan2(s, r) - np.arctan2(a3*np.sin(q3), a2 + a3*np.cos(q3))

    # --- 5. Stylus tilt q4 ---
    # Given only the z-component of the x₄ axis:  x₄z = sin(q2 + q3 + q4) = c
    c = np.clip(c, -1.0, 1.0)
    q4 = np.arcsin(c) - (q2 + q3)

    return np.array([q1, q2, q3, q4])

# ------------------------------------------------------------
#  Utility: compute forward positions for plotting
# ------------------------------------------------------------
def compute_positions(q_vals):
    """Evaluate symbolic transforms numerically and return frame origins and axes."""
    subs = {q1: q_vals[0], q2: q_vals[1], q3: q_vals[2], q4: q_vals[3]}
    Tcurr = sp.eye(4)
    origins = [np.array([0, 0, 0])]  # fixed base origin
    x_axes, y_axes, z_axes = [], [], []
    for p in dh_params:
        Tcurr = Tcurr * DH(*p)
        Tn = np.array(Tcurr.evalf(subs=subs), dtype=float)
        origins.append(Tn[0:3, 3])
        x_axes.append(Tn[0:3, 0])
        y_axes.append(Tn[0:3, 1])
        z_axes.append(Tn[0:3, 2])
    return origins, x_axes, y_axes, z_axes

# ------------------------------------------------------------
#  Visualization setup (same style as ex1)
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.38)

# World axes at origin (base fixed at 0,0,0)
world_len = 30
ax.quiver(0, 0, 0, 1, 0, 0, color="r", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, color="g", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, color="b", length=world_len, normalize=True)
ax.text(35, 0, 0, "X₀", color="r")
ax.text(0, 35, 0, "Y₀", color="g")
ax.text(0, 0, 35, "Z₀", color="b")

# Initial target (x,y,z,c)
target_init = [120, 0, 120, 0.0]
q_init = ik_solver(*target_init)
origins, x_axes, y_axes, z_axes = compute_positions(q_init)

# Draw links and frames
link_lines = []
for i in range(len(origins) - 1):
    p1, p2 = origins[i], origins[i + 1]
    (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k--", linewidth=1)
    link_lines.append(line)

frame_quivers = []
for i in range(1, len(origins)):
    o = origins[i]
    qx = ax.quiver(o[0], o[1], o[2], *x_axes[i-1], color="r", length=20, normalize=True)
    qy = ax.quiver(o[0], o[1], o[2], *y_axes[i-1], color="g", length=20, normalize=True)
    qz = ax.quiver(o[0], o[1], o[2], *z_axes[i-1], color="b", length=20, normalize=True)
    frame_quivers.append((qx, qy, qz))
    ax.text(o[0], o[1], o[2], f"{{{i}}}", fontsize=10, color="k")

# Axis limits and labels
ax.set_xlim([-100, 300])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 300])
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("Ex2 IK: sliders for x, y, z, c")
ax.view_init(elev=25, azim=45)
ax.set_box_aspect([1, 1, 1])

# ------------------------------------------------------------
#  Sliders for (x, y, z, c)
# ------------------------------------------------------------
axcolor = "lightgoldenrodyellow"
slider_labels = ["x", "y", "z", "c (x4z)"]
init_vals = target_init
slider_ranges = [(-150, 250), (-150, 150), (0, 250), (-1.0, 1.0)]
sliders = []

for i, (lbl, rng, init) in enumerate(zip(slider_labels, slider_ranges, init_vals)):
    ax_slider = plt.axes([0.15, 0.28 - i*0.04, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, lbl, rng[0], rng[1], valinit=init)
    sliders.append(slider)

# ------------------------------------------------------------
#  Reset button
# ------------------------------------------------------------
reset_ax = plt.axes([0.8, 0.05, 0.12, 0.05])
button_reset = Button(reset_ax, "Reset", color="lightgray", hovercolor="0.8")

# ------------------------------------------------------------
#  Update function
# ------------------------------------------------------------
def update(val):
    # Read slider values and compute new joint angles
    target = [s.val for s in sliders]
    q_vals = ik_solver(*target)

    # Forward kinematics for visualization
    origins, x_axes, y_axes, z_axes = compute_positions(q_vals)

    # Update link lines
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i + 1]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])

    # Update frame quivers
    for (qx, qy, qz) in frame_quivers:
        qx.remove(); qy.remove(); qz.remove()
    frame_quivers.clear()
    for i in range(1, len(origins)):
        o = origins[i]
        qx = ax.quiver(o[0], o[1], o[2], *x_axes[i-1], color="r", length=20, normalize=True)
        qy = ax.quiver(o[0], o[1], o[2], *y_axes[i-1], color="g", length=20, normalize=True)
        qz = ax.quiver(o[0], o[1], o[2], *z_axes[i-1], color="b", length=20, normalize=True)
        frame_quivers.append((qx, qy, qz))

    plt.draw()

def reset(event):
    for s in sliders:
        s.reset()

for s in sliders:
    s.on_changed(update)
button_reset.on_clicked(reset)

plt.show()
