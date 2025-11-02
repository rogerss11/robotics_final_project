"""
Symbolic Forward Kinematics based on DH Parameters
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Define symbolic variables ---
q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")

def DH(a, alpha, d, theta):
    """Standard DH transformation matrix"""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# --- DH parameters: (a, α, d, θ) ---
dh_params = [
    (0, sp.pi / 2, 50, q1),
    (93, 0, 0, q2),
    (93, 0, 0, q3),
    # (50, sp.pi / 2, 0, q4)
]

# --- Compute T04 symbolically ---
T = sp.eye(4)
for params in dh_params:
    T = T * DH(*params)
T04 = sp.simplify(T)

# --- Print T04 nicely ---
sp.init_printing(use_unicode=True)
print("\nSymbolic transformation T_0_4 (base → stylus):")
sp.pprint(T04)
print("\nPosition of stylus p_0_4 =")
sp.pprint(T04[:3, 3])

# --- Fixed transform from frame {4} to {5} (camera) ---
T_5_4 = sp.Matrix([
    [1, 0, 0, -15],
    [0, 1, 0,  45],
    [0, 0, 1,   0],
    [0, 0, 0,   1]
])

# --- Compute all transformations symbolically for plotting ---
Ts = []
T = sp.eye(4)
for params in dh_params:
    T = T * DH(*params)
    Ts.append(sp.simplify(T))
Ts.append(sp.simplify(T * T_5_4))  # include frame 5

def compute_positions(q_vals):
    subs = {q1: q_vals[0], q2: q_vals[1], q3: q_vals[2], q4: q_vals[3]}
    Ts_num = [np.array(Ti.evalf(subs=subs), dtype=float) for Ti in Ts]
    origins = [np.array([0, 0, 0])]
    x_axes, y_axes, z_axes = [], [], []
    for T in Ts_num:
        origins.append(T[0:3, 3])
        x_axes.append(T[0:3, 0])
        y_axes.append(T[0:3, 1])
        z_axes.append(T[0:3, 2])
    return origins, x_axes, y_axes, z_axes

# --- Create figure ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.38)

# Fixed world axes at origin
world_len = 30
ax.quiver(0, 0, 0, 1, 0, 0, color="r", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, color="g", length=world_len, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, color="b", length=world_len, normalize=True)
ax.text(35, 0, 0, "X₀", color="r")
ax.text(0, 35, 0, "Y₀", color="g")
ax.text(0, 0, 35, "Z₀", color="b")

# Initial pose
q_init = [0, np.pi / 2, 0, 0]
origins, x_axes, y_axes, z_axes = compute_positions(q_init)

# Draw links and frames once
link_lines = []
for i in range(len(origins) - 1):
    p1, p2 = origins[i], origins[i + 1]
    (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k--", linewidth=1)
    link_lines.append(line)

frame_quivers = []
for i in range(1, len(origins)):
    o = origins[i]
    qx = ax.quiver(o[0], o[1], o[2], *x_axes[i - 1], color="r", length=20, normalize=True)
    qy = ax.quiver(o[0], o[1], o[2], *y_axes[i - 1], color="g", length=20, normalize=True)
    qz = ax.quiver(o[0], o[1], o[2], *z_axes[i - 1], color="b", length=20, normalize=True)
    frame_quivers.append((qx, qy, qz))
    ax.text(o[0], o[1], o[2], f"{{{i}}}", fontsize=10, color="k")

# Fixed axis limits
ax.set_xlim([-100, 300])
ax.set_ylim([-150, 150])
ax.set_zlim([0, 350])
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("Robot frames {0}–{5}")
ax.view_init(elev=25, azim=45)
ax.set_box_aspect([1, 1, 1])

# --- Sliders ---
axcolor = "lightgoldenrodyellow"
slider_labels = ["q1", "q2", "q3", "q4"]
sliders = []
for i in range(4):
    ax_slider = plt.axes([0.15, 0.28 - i * 0.04, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, slider_labels[i], -np.pi, np.pi, valinit=q_init[i])
    sliders.append(slider)

# --- Reset button ---
reset_ax = plt.axes([0.8, 0.05, 0.12, 0.05])
button_reset = Button(reset_ax, "Reset", color="lightgray", hovercolor="0.8")

# --- Update function ---
def update(val):
    q_vals = [s.val for s in sliders]
    origins, x_axes, y_axes, z_axes = compute_positions(q_vals)
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i + 1]
        line.set_data([p1[0], p2[0]])
        line.set_3d_properties([p1[2], p2[2]])
    for (qx, qy, qz) in frame_quivers:
        qx.remove(); qy.remove(); qz.remove()
    frame_quivers.clear()
    for i in range(1, len(origins)):
        o = origins[i]
        qx = ax.quiver(o[0], o[1], o[2], *x_axes[i - 1], color="r", length=20, normalize=True)
        qy = ax.quiver(o[0], o[1], o[2], *y_axes[i - 1], color="g", length=20, normalize=True)
        qz = ax.quiver(o[0], o[1], o[2], *z_axes[i - 1], color="b", length=20, normalize=True)
        frame_quivers.append((qx, qy, qz))
    plt.draw()

def reset(event):
    for i, s in enumerate(sliders):
        s.reset()

for s in sliders:
    s.on_changed(update)
button_reset.on_clicked(reset)

plt.show()
