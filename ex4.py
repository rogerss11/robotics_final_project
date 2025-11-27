import numpy as np
from functions import (
    generate_one_circle_point,
    ik_solver,
    geometric_jacobian
)

# ----------------------------------------------------
# Angles of interest
# ----------------------------------------------------
phis = [0, np.pi/2, np.pi, 3*np.pi/2]
phi_names = ["0", "π/2", "π", "3π/2"]

# ----------------------------------------------------
# Evaluate Jacobians
# ----------------------------------------------------
for phi, name in zip(phis, phi_names):
    print("\n==========================================")
    print(f" φ = {name}")
    print("==========================================")

    # Circle point
    x, y, z = generate_one_circle_point(phi)

    # Horizontal stylus => x4z = 0
    c = 0

    # Compute joint angles
    q = ik_solver(x, y, z, c)

    print("Joint angles q = ", np.round(q, 2))

    # Jacobian for frame {4}
    J4 = geometric_jacobian(q, 4)
    print("\nJacobian for frame {4}:")
    print(np.round(J4, 2))

    # Jacobian for frame {5}
    J5 = geometric_jacobian(q, 5)
    print("\nJacobian for frame {5}:")
    print(np.round(J5, 2))
