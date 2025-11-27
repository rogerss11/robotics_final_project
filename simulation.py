import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from types import SimpleNamespace

from functions import DH_params, DH, T_45
from functions import ik_solver, generate_circle_points
from functions import generate_one_circle_point


# ===========================================================
# ---- LOW-LEVEL UTILS (NO REPETITION)
# ===========================================================
def compute_positions(q_vals):
    dh = DH_params(*q_vals)
    T = np.eye(4)
    Ts = []

    for params in dh:
        T = T @ DH(*params)
        Ts.append(T)

    Ts.append(T @ T_45())  # frame 5

    origins = [np.zeros(3)]
    x_axes, y_axes, z_axes = [], [], []

    for Ti in Ts:
        origins.append(np.array(Ti[:3, 3], dtype=float).flatten())
        x_axes.append(np.array(Ti[:3, 0], dtype=float).flatten())
        y_axes.append(np.array(Ti[:3, 1], dtype=float).flatten())
        z_axes.append(np.array(Ti[:3, 2], dtype=float).flatten())
    
    # for i, o in enumerate(origins):
    #     print(f"Origin of frame {{{i}}}: {o}")

    return origins, x_axes, y_axes, z_axes



def create_world_axes(ax, L=30):
    """Draw world reference frame."""
    ax.quiver(0,0,0,1,0,0,color="r",length=L,normalize=True)
    ax.quiver(0,0,0,0,1,0,color="g",length=L,normalize=True)
    ax.quiver(0,0,0,0,0,1,color="b",length=L,normalize=True)
    ax.text(35,0,0,"X",color="r")
    ax.text(0,35,0,"Y",color="g")
    ax.text(0,0,35,"Z",color="b")


def draw_robot(ax, origins, x_axes, y_axes, z_axes):
    """Draw link lines and coordinate frames."""
    link_lines = []
    frame_quivers = []

    for i in range(len(origins)-1):
        p1, p2 = origins[i], origins[i+1]
        (line,) = ax.plot(*zip(p1, p2), "k--", linewidth=1)
        link_lines.append(line)

    for i in range(1, len(origins)):
        o = origins[i]
        qx = ax.quiver(*o, *x_axes[i-1], color="r", length=20, normalize=True)
        qy = ax.quiver(*o, *y_axes[i-1], color="g", length=20, normalize=True)
        qz = ax.quiver(*o, *z_axes[i-1], color="b", length=20, normalize=True)
        frame_quivers.append((qx, qy, qz))
        ax.text(*o, f"{{{i}}}", fontsize=10, color="k")

    return link_lines, frame_quivers


def update_robot(ax, q_vals, link_lines, frame_quivers):
    """Update robot shape using new joint angles."""
    origins, x_axes, y_axes, z_axes = compute_positions(q_vals)

    # update links
    for i, line in enumerate(link_lines):
        p1, p2 = origins[i], origins[i+1]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])

    # redraw frames
    for (qx, qy, qz) in frame_quivers:
        qx.remove(); qy.remove(); qz.remove()
    frame_quivers.clear()

    for i in range(1, len(origins)):
        o = origins[i]
        qx = ax.quiver(*o, *x_axes[i-1], color="r", length=20, normalize=True)
        qy = ax.quiver(*o, *y_axes[i-1], color="g", length=20, normalize=True)
        qz = ax.quiver(*o, *z_axes[i-1], color="b", length=20, normalize=True)
        frame_quivers.append((qx, qy, qz))

    plt.draw()


def setup_3d_scene(title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.32)

    create_world_axes(ax)

    ax.set_xlim([-100, 300])
    ax.set_ylim([-150, 150])
    ax.set_zlim([0, 350])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=25, azim=45)

    return fig, ax


def slider_block(labels, ranges, initial):
    sliders = []
    for i, (lab, rng, init) in enumerate(zip(labels, ranges, initial)):
        ax_sl = plt.axes([0.15, 0.27 - i*0.04, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        sliders.append(Slider(ax_sl, lab, rng[0], rng[1], valinit=init))
    return sliders


# ===========================================================
# ---- EXERCISE 1 — JOINT SLIDERS (FK)
# ===========================================================

def launch_simulation():
    fig, ax = setup_3d_scene("Robot frames {0}–{5}")

    q_init = [0, np.pi/2, 0, 0]
    origins, x_axes, y_axes, z_axes = compute_positions(q_init)
    link_lines, frame_quivers = draw_robot(ax, origins, x_axes, y_axes, z_axes)

    sliders = slider_block(
        ["q1","q2","q3","q4"],
        [(-np.pi, np.pi)] * 4,
        q_init
    )

    for s in sliders:
        s.on_changed(lambda _:
            update_robot(ax, [s.val for s in sliders], link_lines, frame_quivers)
        )

    Button(plt.axes([0.8,0.05,0.12,0.05]),
           "Reset", color="lightgray").on_clicked(lambda _:
               [s.reset() for s in sliders])

    plt.show()


# ===========================================================
# ---- EXERCISE 2 — XYZ SLIDERS (IK)
# ===========================================================

def launch_simulation_xyz():
    fig, ax = setup_3d_scene("IK control: sliders for x,y,z,c")

    x0, y0, z0, c0 = 120, 0, 120, 0
    q_init = ik_solver(x0, y0, z0, c0)

    origins, x_axes, y_axes, z_axes = compute_positions(q_init)
    link_lines, frame_quivers = draw_robot(ax, origins, x_axes, y_axes, z_axes)

    sliders = slider_block(
        ["x","y","z","c"],
        [(-150,250), (-150,150), (0,250), (-1,1)],
        [x0, y0, z0, c0]
    )

    def update(_):
        x, y, z, c = [s.val for s in sliders]
        q = ik_solver(x, y, z, c)
        update_robot(ax, q, link_lines, frame_quivers)

    for s in sliders:
        s.on_changed(update)

    plt.show()


# ===========================================================
# ---- EXERCISE 3 — CIRCLE SIMULATION
# ===========================================================

def plot_circle(ax, N=37):
    """
    Draw the generated circle points in the same 3D plot as the robot.

    Parameters:
        ax : the existing 3D axes (from setup_3d_scene)
        N  : number of points around the circle
    """
    pts = generate_circle_points(N)   # (N,3) array
    xs = pts[:,0]
    ys = pts[:,1]
    zs = pts[:,2]

    # Draw the circle curve
    ax.plot(xs, ys, zs, 'm-', linewidth=2)

    # Draw the individual sample points
    ax.scatter(xs, ys, zs, color='m', s=25)

    return pts

def launch_simulation_circle():
    fig, ax = setup_3d_scene("IK control: sliders for phi around circle")

    # optional: show the circle
    plot_circle(ax)

    phi0 = 0
    x0, y0, z0 = generate_one_circle_point(phi0)
    c0 = 0
    q_init = ik_solver(x0, y0, z0, c0)

    origins, x_axes, y_axes, z_axes = compute_positions(q_init)
    link_lines, frame_quivers = draw_robot(ax, origins, x_axes, y_axes, z_axes)

    sliders = slider_block(
        ["phi"],
        [(0, 2*np.pi)],
        [0]
    )

    def update(_):
        phi = [s.val for s in sliders]
        x, y, z = generate_one_circle_point(phi[0])
        c = 0
        q = ik_solver(x, y, z, c)
        update_robot(ax, q, link_lines, frame_quivers)
        plt.draw()   # <--- important

    for slider in sliders:
        slider.on_changed(update)

    plt.show()
