import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------
# Denavit–Hartenberg Transformation
# -----------------------------------------------------------
def DH(theta, alpha, a, d):
    """Standard DH homogeneous transformation (angles in radians)."""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# -----------------------------------------------------------
# Symbolic variables
# -----------------------------------------------------------
theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')

# Link parameters (converted to meters)
a1, alpha1, d1 = 0.0, sp.pi/2, 0.050
a2, alpha2, d2 = 0.093, 0.0, 0.0
a3, alpha3, d3 = 0.093, 0.0, 0.0
a4, alpha4, d4 = 0.050, 0.0, 0.0

# -----------------------------------------------------------
# Forward Kinematics (symbolic)
# -----------------------------------------------------------
T01 = DH(theta1, alpha1, a1, d1)
T12 = DH(theta2, alpha2, a2, d2)
T23 = DH(theta3, alpha3, a3, d3)
T34 = DH(theta4, alpha4, a4, d4)

# Tool (stylus tip) offset relative to frame 4
T45 = sp.Matrix([
    [1, 0, 0, -0.015],
    [0, 1, 0,  0.045],
    [0, 0, 1,  0.0],
    [0, 0, 0,  1.0]
])

T04 = sp.simplify(T01 * T12 * T23 * T34)
T05 = sp.simplify(T04 * T45)   # includes the stylus offset

#T04_num = T04.evalf(subs={theta1: 0, theta2: -11.4205295, theta3: 83.35392622, theta4: -60.39644124})
#sp.pprint(T04)

#first we need to find theta3 from theta4

def ik_solver_tip(x, y, z, c):
    """
    Inverse kinematics when (x, y, z) is the stylus tip (O4),
    not the wrist (O3).
    """
    a2, a3, d1, d4 = 93, 93, 50, 50  # mm
    q1 = np.arctan2(y, x) #nope!

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


def inverse_kinematics(x, y, z, x4z):
    a1, a2, a3, d1 = 0.0, 0.093, 0.093, 0.050

    q1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) - a1 #why -a1 i dont know anymore, its 0
    s = z - d1

    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c3 = np.clip(c3, -1.0, 1.0)
    q3 = np.arctan2(np.sqrt(1 - c3**2), c3)

    q2 = np.arctan2(s, r) - np.arctan2(a3*np.sin(q3), a2 + a3*np.cos(q3))

    x4z = np.clip(x4z, -1.0, 1.0)
    q4 = np.arcsin(x4z) - (q2 + q3)

    return np.array([q1, q2, q3, q4])


def ik_solver_right(x, y, z, c):
    """
    Inverse kinematics when (x, y, z) is the stylus tip (O4),
    not the wrist (O3).
    """
    a2, a3, d1, d4 = 93, 93, 50, 50  # mm
    q1 = np.arctan2(yc, xc) #xc og yc for posisjon 

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