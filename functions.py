import numpy as np

# ======= Exercise 1: Forward kinematics =======

# DH parameters function
def DH(a, alpha, d, theta):
    """Standard DH transformation matrix"""
    return np.matrix([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# --- DH parameters: (a, α, d, θ) ---
def DH_params(q1, q2, q3, q4):
    """
    Returns the DH parameters for given joint angles.
    q1, q2, q3, q4: joint angles in radians
    Columns: a, alpha, d, theta
    [
        (a1, α1, d1, q1),
        (a2, α2, d2, q2),
        (a3, α3, d3, q3),
        (a4, α4, d4, q4),
    ]
    """
    return [
        (0, np.pi / 2, 50, q1),
        (93, 0, 0, q2),
        (93, 0, 0, q3),
        (50, 0, 0, q4)
    ]

# --- Compute T0i numerically ---
def compute_T0i(q_vals, i):
    """
    q_vals: [q1, q2, q3, q4] in radians
    i: index of the desired transformation (1 to 4)
    """
    dh_params = DH_params(q_vals[0], q_vals[1], q_vals[2], q_vals[3])
    T = np.eye(4)
    Ts = []
    for params in dh_params:
        T = T @ DH(*params)
        Ts.append(T)
    return Ts[i-1]

# --- T45 fixed transform ---
def T_45():
    """Fixed transform from frame {4} to {5} (camera)"""
    return np.matrix([
        [1, 0, 0, -15],
        [0, 1, 0,  45],
        [0, 0, 1,   0],
        [0, 0, 0,   1]
    ])

# ======= Exercise 2: Inverse kinematics =======
a1, a2, a3, a4, d1 = 0, 93, 93, 50, 50
def ik_solver(x4, y4, z4, c):
    """
    Analytic IK for the 4-DOF arm.
    Solves for the position of FRAME {4}.
    """

    # ---------------------------------------
    # 0. Base rotation
    # ---------------------------------------
    q1 = np.arctan2(y4, x4)

    # ---------------------------------------
    # 1. Project target back by link a4 to get FRAME {3}
    #    The offset is reduced in the planar r–s plane.
    # ---------------------------------------
    r4 = np.sqrt(x4**2 + y4**2)
    s4 = z4 - d1

    # Reduce radius by a4
    r3 = r4 - a4
    s3 = s4

    # ---------------------------------------
    # 2. Law of cosines for q3
    # ---------------------------------------
    c3 = (r3**2 + s3**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c3 = np.clip(c3, -1, 1)
    q3 = np.arctan2(-np.sqrt(1 - c3**2), c3)     # elbow-up

    # ---------------------------------------
    # 3. Shoulder q2
    # ---------------------------------------
    q2 = np.arctan2(s3, r3) - np.arctan2(a3*np.sin(q3), a2 + a3*np.cos(q3))

    # ---------------------------------------
    # 4. Wrist q4 from the z-component of x4
    #    x4z = sin(q2 + q3 + q4)
    # ---------------------------------------
    c = np.clip(c, -1, 1)
    q4 = np.arcsin(c) - (q2 + q3)

    return np.array([q1, q2, q3, q4])


# ======= Exercise 3: Circle positions =======

def generate_one_circle_point(phi):
    """
    Compute point (x,y,z) on circle for given angle phi.
    """
    R = 32  # mm
    p_c = np.array([150, 0, 120])  # circle center
    p = p_c + R * np.array([0, np.cos(phi), np.sin(phi)])
    x, y, z = p
    return x, y, z

def generate_circle_points(N=37):
    """
    Generate N points around the circle.
    Returns an array of shape (N, 3) with (x,y,z) coordinates.
    """
    points = []
    for i in range(N):
        phi = 2 * np.pi * i / N
        x, y, z = generate_one_circle_point(phi)
        points.append([x, y, z])
    return np.array(points)

def compute_joint_positions_circle(phi_vals, N=37):
    """
    Compute all joint positions for points around the circle.
    Returns an array of shape (37, 4) with joint angles.
    """
    Q = []
    for i in range(N):
        phi = phi_vals[i]
        x, y, z = generate_one_circle_point(phi)
        c = 0  # horizontal stylus (x4z = 0)
        qv = ik_solver(x, y, z, c)
        Q.append(qv)
    return np.array(Q)

# ======= Exercise 4: Jacobian =======

def geometric_jacobian(q_vals, i):
    """
    Compute the geometric Jacobian for frame {i}.

    q_vals: [q1, q2, q3, q4] in radians
    i: index of the desired frame (1 to 5)
    """
    # Compute T0i for each frame
    Ts = [np.eye(4)]
    for idx in range(1, 5):
        Ts.append(compute_T0i(q_vals, idx-1))
    # If frame {5} (camera), apply fixed transform from frame 4 to 5
    if i == 5:
        Ts.append(Ts[4] @ T_45())

    Jv_cols = []
    Jw_cols = []
    o_n = Ts[i][0:3, 3].flatten()    # Ensure shape (3,)

    # Loop through each joint up to frame {i}
    for j in range(i):
        z_im1 = Ts[j][0:3, 2].flatten()   # Ensure shape (3,)
        o_im1 = Ts[j][0:3, 3].flatten()   # Ensure shape (3,)

        # Linear part: z_(i-1) × (o_n - o_(i-1))
        Jv_cols.append(np.cross(z_im1, o_n - o_im1))
        # Angular part: z_(i-1)
        Jw_cols.append(z_im1 if z_im1.shape == (3,) else np.asarray(z_im1).flatten())

    # Stack columns to form Jacobian submatrices
    Jv = np.column_stack([np.asarray(col).flatten() for col in Jv_cols])
    Jw = np.column_stack([np.asarray(col).flatten() for col in Jw_cols])
    # Combine linear and angular parts into full Jacobian
    J = np.vstack((Jv, Jw))
    return J

# ======= Exercise 5: Joint velocities =======

def joint_velocities(q_vals, v_e):
    """
    Compute joint velocities given end-effector velocity.

    q_vals: [q1, q2, q3, q4] in radians
    v_e: end-effector velocity vector [vx, vy, vz, wx, wy, wz]
    i: index of the desired frame (1 to 5)
    """
    J = geometric_jacobian(q_vals, 4)  # Jacobian for EE {4}
    J_inv = np.linalg.pinv(J)  # Pseudoinverse in case J is not square
    q_dot = J_inv @ v_e
    return q_dot

# ======= Exercise 6: Trajectory planning =======

# ======= Exercise 7: EE path =======

# ====== Exercise 8: Condition number & Singularities =======

# ===== Exercise 9: Joint Torques =======

def joint_torques(q_vals, F_e):
    """
    Compute joint torques given end-effector force/torque.

    q_vals: [q1, q2, q3, q4] in radians
    F_e: end-effector force/torque vector [fx, fy, fz, tx, ty, tz]
    i: index of the desired frame (1 to 5)
    """
    J = geometric_jacobian(q_vals, 4)  # Jacobian for EE {4}
    tau = J.T @ F_e
    return tau

def joint_torques_circle(phi_vals, F_e, N=37):
    """
    Compute joint torques for points around the circle given a constant end-effector force.

    phi_vals: array of angles around the circle
    N: number of points
    Returns an array of shape (N, 4) with joint torques.
    """
    Taus = []
    for i in range(N):
        phi = phi_vals[i]
        x, y, z = generate_one_circle_point(phi)
        c = 0  # horizontal stylus
        qv = ik_solver(x, y, z, c)
        tau = joint_torques(qv, F_e)
        Taus.append(tau.flatten())  # Convert array to 1D array
    return np.array(Taus)

# ===== Exercise 10: Inertia and Dynamics Simulation =======