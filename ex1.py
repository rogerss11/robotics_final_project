from simulation import launch_simulation

import sympy as sp

# ============= Symbolic T04 and T05 computation =============
# Joint variables
q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")

# DH matrix (symbolic version matching functions.py)
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH parameters
dh_params = [
    (0,  sp.pi/2, 50, q1),
    (93, 0,       0,  q2),
    (93, 0,       0,  q3),
    (50, 0,       0,  q4)
]

# Compute T04
T = sp.eye(4)
for p in dh_params:
    T = T @ DH(*p)

T04 = sp.simplify(T)

# Fixed transform from frame 4 to 5 (camera)
T45 = sp.Matrix([
    [1, 0, 0, -15],
    [0, 1, 0,  45],
    [0, 0, 1,   0],
    [0, 0, 0,   1]
])

# Compute T05 = T04 * T45
T05 = sp.simplify(T04 * T45)

sp.init_printing()
print("T04 =")
sp.pprint(T04)
print("\nT05 =")
sp.pprint(T05)

# ============= Launch simulation =============



if __name__ == "__main__":
    launch_simulation()
    