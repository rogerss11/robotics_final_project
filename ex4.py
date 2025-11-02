import sympy as sp
import numpy as np

# ------------------------------------------------------------
#  DH helper and symbolic FK
# ------------------------------------------------------------
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# Symbolic variables and DH parameters
q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")
dh_params = [
    (0, sp.pi/2, 50, q1),
    (93, 0, 0, q2),
    (93, 0, 0, q3),
    (50, sp.pi/2, 0, q4)
]

# Fixed transform from frame {4} to {5} (camera)
T_5_4 = sp.Matrix([
    [1, 0, 0, -15],
    [0, 1, 0,  45],
    [0, 0, 1,   0],
    [0, 0, 0,   1]
])

# Compute all forward transforms T0_i
def forward_transforms():
    Ts = [sp.eye(4)]
    T = sp.eye(4)
    for p in dh_params:
        T = T * DH(*p)
        Ts.append(sp.simplify(T))
    Ts.append(sp.simplify(T * T_5_4))
    return Ts  # [T00, T01, T02, T03, T04, T05]

Ts = forward_transforms()
T04, T05 = Ts[4], Ts[5]

# ------------------------------------------------------------
#  Jacobian construction
# ------------------------------------------------------------
def geometric_jacobian(Ts, end_index):
    o_n = Ts[end_index][0:3, 3]
    Jv_cols, Jw_cols = [], []
    for i in range(1, 5):  # 4 revolute joints
        z_im1 = Ts[i-1][0:3, 2]
        o_im1 = Ts[i-1][0:3, 3]
        Jv_cols.append(sp.Matrix.cross(z_im1, o_n - o_im1))
        Jw_cols.append(z_im1)
    Jv = sp.Matrix.hstack(*Jv_cols)
    Jw = sp.Matrix.hstack(*Jw_cols)
    return sp.simplify(sp.Matrix.vstack(Jv, Jw))

J4 = geometric_jacobian(Ts, 4)
J5 = geometric_jacobian(Ts, 5)

# ------------------------------------------------------------
#  Symbolic results
# ------------------------------------------------------------
print("\n================ SYMBOLIC RESULTS ================\n")
print("T04 (symbolic):")
sp.pprint(T04)
print("\nT05 (symbolic):")
sp.pprint(T05)
print("\nJacobian at frame {4} (symbolic):")
sp.pprint(J4)
print("\nJacobian at frame {5} (symbolic):")
sp.pprint(J5)

# ------------------------------------------------------------
#  Inverse Kinematics for given (x,y,z,c)
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
#  Evaluate on circular path from Problem 3
# ------------------------------------------------------------
R = 32.0
p_c = np.array([150.0, 0.0, 120.0])
phi_list = [0.0, np.pi/2, np.pi, 3*np.pi/2]

print("\n================ NUMERICAL RESULTS ================\n")
for phi in phi_list:
    p = p_c + R * np.array([0.0, np.cos(phi), np.sin(phi)])
    x, y, z = p
    c = 0.0  # stylus horizontal
    qv = ik_solver(x, y, z, c)

    subs = {q1: qv[0], q2: qv[1], q3: qv[2], q4: qv[3]}
    T04_num = np.array(T04.evalf(subs=subs), dtype=float)
    T05_num = np.array(T05.evalf(subs=subs), dtype=float)
    J4_num = np.array(J4.evalf(subs=subs), dtype=float)
    J5_num = np.array(J5.evalf(subs=subs), dtype=float)

    print(f"\n---------------- φ = {phi:.4f} ----------------")
    print(f"Target (x, y, z) = ({x:.1f}, {y:.1f}, {z:.1f})")
    print("Joint values [rad] =", np.round(qv, 5))
    print("\nT04 =\n", np.round(T04_num, 4))
    print("\nT05 =\n", np.round(T05_num, 4))
    print("\nJacobian at frame {4} (6x4):\n", np.round(J4_num, 4))
    print("\nJacobian at frame {5} (6x4):\n", np.round(J5_num, 4))
