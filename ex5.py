import sympy as sp
import numpy as np

# ------------------------------------------------------------
# Import from Problem 4 (DH, FK, Jacobian)
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
    (50, sp.pi/2, 0, q4)
]

def forward_transforms():
    Ts = [sp.eye(4)]
    T = sp.eye(4)
    for p in dh_params:
        T = T * DH(*p)
        Ts.append(sp.simplify(T))
    return Ts  # up to frame {4}

Ts = forward_transforms()

def geometric_jacobian(Ts, end_index):
    o_n = Ts[end_index][0:3, 3]
    Jv_cols, Jw_cols = [], []
    for i in range(1, 5):
        z_im1 = Ts[i-1][0:3, 2]
        o_im1 = Ts[i-1][0:3, 3]
        Jv_cols.append(sp.Matrix.cross(z_im1, o_n - o_im1))
        Jw_cols.append(z_im1)
    return sp.simplify(sp.Matrix.vstack(sp.Matrix.hstack(*Jv_cols),
                                        sp.Matrix.hstack(*Jw_cols)))

J4 = geometric_jacobian(Ts, 4)

# ------------------------------------------------------------
# Inverse kinematics (same as before)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Evaluate at φ = π/2 (from Problem 3)
# ------------------------------------------------------------
R = 32.0
p_c = np.array([150.0, 0.0, 120.0])
phi = np.pi / 2
p = p_c + R * np.array([0.0, np.cos(phi), np.sin(phi)])
x, y, z = p
c = 0.0
qv = ik_solver(x, y, z, c)
subs = {q1: qv[0], q2: qv[1], q3: qv[2], q4: qv[3]}

J4_num = np.array(J4.evalf(subs=subs), dtype=float)

print("\nConfiguration at φ = π/2:")
print("q = ", np.round(qv, 5))
print("\nJacobian J4 =\n", np.round(J4_num, 4))

# ------------------------------------------------------------
# Desired end-effector velocity
# ------------------------------------------------------------
v4 = np.array([[0.0], [-3.0], [0.0]])  # mm/s
# Assume stylus rotates about x-axis to stay horizontal
omega4 = np.array([[0.0], [0.0], [0.0]])  # rad/s (example value)
xdot = np.vstack((v4, omega4))  # 6x1

# ------------------------------------------------------------
# Compute joint velocities
# ------------------------------------------------------------
qdot = np.linalg.pinv(J4_num) @ xdot

print("\nDesired tip velocity v4 = [0, -3, 0] mm/s and ω4 = [1, 0, 0] rad/s")
print("Joint velocities qdot =\n", np.round(qdot, 6))
