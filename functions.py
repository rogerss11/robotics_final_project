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
    x4, y4, z4: target position of frame {4}
    c: tilt angle constraint (x4z = sin(q2 + q3 + q4) = c)
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
    r3 = r4 - a4*np.cos(c)
    s3 = s4 - a4*np.sin(c)

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
    Compute the 6×4 geometric Jacobian for frame {i} (i = 1..5).
    q_vals = [q1, q2, q3, q4]
    """

    # ------------------------------------------------------------
    # Build T0k for k = 0..4 using compute_T0i(q_vals, k)
    # ------------------------------------------------------------
    Ts = [np.eye(4)]                      # T00
    for k in range(1, 5):                 # T01..T04
        Ts.append(compute_T0i(q_vals, k))

    # ------------------------------------------------------------
    # Frame 5 → camera frame: T05 = T04 · T45
    # ------------------------------------------------------------
    if i == 5:
        T05 = Ts[4] @ T_45()
        Ts.append(T05)
    else:
        Ts.append(None)

    # Origin of the target frame i
    o_n = Ts[i][0:3, 3]

    Jv_cols = []
    Jw_cols = []

    # ------------------------------------------------------------
    # Loop through robot joints j = 1..4
    # ------------------------------------------------------------
    for j in range(1, 5):

        # If joint j is AFTER frame i → it cannot affect this frame
        if j > i:
            Jv_cols.append(np.zeros(3))
            Jw_cols.append(np.zeros(3))
            continue

        # Else joint j affects this frame → compute Jacobian normally
        Tjm1 = Ts[j-1]
        z = Tjm1[0:3, 2]
        o = Tjm1[0:3, 3]

        z = np.asarray(z).reshape(3,)
        o = np.asarray(o).reshape(3,)
        o_n = np.asarray(o_n).reshape(3,)

        # Linear part
        Jv_cols.append(np.cross(z, (o_n - o)))

        # Angular part
        Jw_cols.append(z)

    # Combine into 6×4 Jacobian
    Jv = np.column_stack(Jv_cols)
    Jw = np.column_stack(Jw_cols)
    J  = np.vstack((Jv, Jw))

    return J

def jacobian_at_point(q_vals, p_local, frame_index):
    """
    Compute the 6×4 geometric Jacobian for ANY point rigidly attached 
    to ANY frame of the robot.
    """

    # ------------------------------------------------------------
    # Build transforms T0k for k = 0..4
    # ------------------------------------------------------------
    Ts = [np.eye(4)]                      # T00
    for k in range(1, 5):                 # T01..T04
        Ts.append(compute_T0i(q_vals, k))

    # Append camera frame if needed
    if frame_index == 5:
        T05 = Ts[4] @ T_45()
        Ts.append(T05)
    else:
        Ts.append(None)

    # ------------------------------------------------------------
    # Compute world position of the point o_p
    # ------------------------------------------------------------
    T0f = Ts[frame_index]
    p_h = np.hstack((p_local, 1))

    # FIX: flatten output of matrix multiplication
    o_p_full = np.asarray(T0f @ p_h).reshape(4,)
    o_p = o_p_full[0:3]      # final 3D point

    Jv_cols = []
    Jw_cols = []

    # ------------------------------------------------------------
    # Loop through joints j = 1..4
    # ------------------------------------------------------------
    for j in range(1, 5):

        # joint j cannot affect frame above it
        if j > frame_index:
            Jv_cols.append(np.zeros(3))
            Jw_cols.append(np.zeros(3))
            continue

        Tjm1 = Ts[j-1]

        # Convert to proper 1D vectors
        z = np.asarray(Tjm1[0:3, 2]).reshape(3,)
        o = np.asarray(Tjm1[0:3, 3]).reshape(3,)

        # Linear Jacobian component
        Jv_cols.append(np.cross(z, o_p - o))

        # Angular component
        Jw_cols.append(z)

    # Stack into final 6x4 Jacobian
    Jv = np.column_stack(Jv_cols)
    Jw = np.column_stack(Jw_cols)
    J  = np.vstack((Jv, Jw))

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

Tseg = 2.0
A = np.array([
    [0,0,0,0,0,1],
    [Tseg**5,Tseg**4,Tseg**3,Tseg**2,Tseg,1],
    [0,0,0,0,1,0],
    [5*Tseg**4,4*Tseg**3,3*Tseg**2,2*Tseg,1,0],
    [0,0,0,2,0,0],
    [20*Tseg**3,12*Tseg**2,6*Tseg,2,0,0],
], float)

def solve_quintic(q0, q1, qd0, qd1):
    b = np.array([q0, q1, qd0, qd1, 0, 0])
    return np.linalg.solve(A, b)

def eval_quintic(a, t):
    a5,a4,a3,a2,a1,a0 = a
    q   = a5*t**5 + a4*t**4 + a3*t**3 + a2*t**2 + a1*t + a0
    qd  = 5*a5*t**4 + 4*a4*t**3 + 3*a3*t**2 + 2*a2*t + a1
    qdd = 20*a5*t**3 + 12*a4*t**2 + 6*a3*t + 2*a2
    return q, qd, qdd

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

def Ixx(m, dz, dy):
    """
    Inertia tensor of a cuboid about one of its principal axes.
    m: mass
    a, b: dimensions along y and z axes
    returns Ixx
    """
    return (1/12) * m * (dy**2 + dz**2)

def compute_D(q, masses, I_local, r_ci_list):
    """
    Compute the 4×4 inertia matrix D(q) of the robot at joint configuration q.
    """
    D = np.zeros((4,4))

    for i in range(4):
        # Transform from base to link i+1
        T = compute_T0i(q, i+1)
        R = T[:3,:3]

        # inertia in base frame
        I0 = R @ I_local[i] @ R.T
        
        # Jacobian for COM of link
        J = jacobian_at_point(q, r_ci_list[i], frame_index=i+1)
        Jv = J[0:3,:]
        Jw = J[3:6,:]

        D += masses[i] * (Jv.T @ Jv) + Jw.T @ I0 @ Jw

    return D

def compute_g(q, masses, r_ci_list, g_vec):
    """
    Compute the gravity vector g(q) of the robot at joint configuration q.
    """
    g = np.zeros(4)

    for i in range(4):
        J = jacobian_at_point(q, r_ci_list[i], frame_index=i+1)
        Jv = J[0:3,:]
        g += masses[i] * (Jv.T @ g_vec)

    return g

def compute_C(q, qdot, masses, I_local, r_ci_list):
    """
    Compute the Coriolis and centrifugal matrix C(q, qdot) of the robot at joint configuration q and velocity qdot.
    """
    Dq = compute_D(q, masses, I_local, r_ci_list)
    C = np.zeros((4,4))

    eps = 1e-6
    dD_dq = np.zeros((4,4,4))

    # Numerical partial derivatives ∂D/∂qk
    for k in range(4):
        dq = np.zeros(4)
        dq[k] = eps

        D_plus  = compute_D(q + dq, masses, I_local, r_ci_list)
        D_minus = compute_D(q - dq, masses, I_local, r_ci_list)

        dD_dq[:,:,k] = (D_plus - D_minus) / (2*eps)
    # Compute Christoffel symbols and C matrix
    for i in range(4):
        for j in range(4):
            C[i,j] = 0.5 * sum(
                (dD_dq[i,j,k] + dD_dq[i,k,j] - dD_dq[k,j,i]) * qdot[k]
                for k in range(4)
            )

    return C

def compute_tau(q, qdot, qddot, masses, I_local, r_ci_list, g_vec):
    """
    Compute the joint torques tau given joint positions q, velocities qdot, and accelerations qddot.
    """
    D = compute_D(q, masses, I_local, r_ci_list)
    C = compute_C(q, qdot, masses, I_local, r_ci_list)
    g = compute_g(q, masses, r_ci_list, g_vec)
    return D @ qddot + C @ qdot + g

def compute_torques_trajectory(q, qdot, qddot, masses, I_local, r_ci_list, g_vec):
    """
    Compute joint torques for a trajectory of joint positions, velocities, and accelerations.
    """
    N = q.shape[0]
    tau_all = np.zeros((N,4))

    for k in range(N):
        tau_all[k,:] = compute_tau(q[k], qdot[k], qddot[k], masses, I_local, r_ci_list, g_vec)

    return tau_all