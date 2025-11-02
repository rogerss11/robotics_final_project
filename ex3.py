import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ------------------------------------------------------------
#  DH helpers and symbolic FK (same as ex2)
# ------------------------------------------------------------
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH chain and symbolic variables
q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")
dh_params = [
    (0, sp.pi/2, 50, q1),
    (93, 0, 0, q2),
    (93, 0, 0, q3),
    (50, sp.pi/2, 0, q4)
]

# ------------------------------------------------------------
#  Inverse kinematics solver from Ex2
# ------------------------------------------------------------
a1, a2, a3, d1 = 0, 93, 93, 50

def ik_solver(x, y, z, c):
    """Analytic IK for the 4-DOF manipulator."""
    q1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) - a1
    s = z - d1
    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c3 = np.clip(c3, -1.0, 1.0)
    q3 = np.arctan2(np.sqrt(1 - c3**2), c3)
    q2 = np.arctan2(s, r) - np.arctan2(a3*np.sin(q3), a2 + a3*np.cos(q3))
    c = np.clip(c, -1.0, 1.0)
    q4 = np.arcsin(c) - (q2 + q3)
    return np.array([q1, q2, q3, q4])

# ------------------------------------------------------------
#  Compute forward frame positions for visualization
# ------------------------------------------------------------
def compute_positions(q_vals):
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

# ------------------------------------------------------------
#  Generate 37 circle points based on Problem 3 definition
# ------------------------------------------------------------
R = 32  # mm
p_c = np.array([150, 0, 120])
phi_vals = np.linspace(0, 2*np.pi, 37)

points = []
for phi in phi_vals:
    p = p_c + R * np.array([0, np.cos(phi), np.sin(phi)])
    x, y, z = p
    c = 0  # horizontal stylus (x4z = 0)
    points.append([x, y, z, c])
points = np.array(points)

# ------------------------------------------------------------
#  Visualization setup
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

# Display all 37 points (smaller markers).
# Compute the actual stylus-tip positions by running IK then FK for each target
# so the plotted points correspond to the tip, not joint 4 origin.
tip_positions = []
for p in points:
    x_t, y_t, z_t, c_t = p
    q_tmp = ik_solver(x_t, y_t, z_t, c_t)
    origins_tmp, _, _, _ = compute_positions(q_tmp)
    tip_positions.append(origins_tmp[-1])
tip_positions = np.array(tip_positions)
ax.scatter(tip_positions[:,0], tip_positions[:,1], tip_positions[:,2], c="orange", s=20, label="Target circle")

# Initial pose
target_init = points[0]
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

# Axis setup
ax.set_xlim([0, 250])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 250])
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("Ex3: Stylus tip tracking 37 circular points (horizontal stylus)")
ax.legend()
ax.view_init(elev=25, azim=45)
ax.set_box_aspect([1, 1, 1])

# Phi display (show current point angle) placed in axes coordinates (top-left).
# For 3D axes use text2D to place text in 2D axes coordinates.
deg0 = np.degrees(phi_vals[0])
phi_display = ax.text2D(0.02, 0.95, f"φ = {phi_vals[0]:.3f} rad ({deg0:.1f}°)", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# ------------------------------------------------------------
#  Slider to choose point index
# ------------------------------------------------------------
axcolor = "lightgoldenrodyellow"
ax_slider = plt.axes([0.15, 0.18, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax_slider, "Point index", 0, len(points)-1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    phi = phi_vals[idx]
    x, y, z, c = points[idx]
    q_vals = ik_solver(x, y, z, c)
    origins, x_axes, y_axes, z_axes = compute_positions(q_vals)

    # Update link lines
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i + 1]
        # set_data requires both x and y sequences for a 3D line; pass x and y, then z
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])

    # Update frames
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

    # update phi display after drawing
    try:
        deg = np.degrees(phi)
        phi_display.set_text(f"φ = {phi:.3f} rad ({deg:.1f}°)")
    except Exception:
        # fallback if phi not available
        phi_display.set_text("")

slider.on_changed(update)

# Reset button
reset_ax = plt.axes([0.8, 0.05, 0.12, 0.05])
button_reset = Button(reset_ax, "Reset", color="lightgray", hovercolor="0.8")

def reset(event):
    slider.reset()

button_reset.on_clicked(reset)

plt.show()
