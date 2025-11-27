
import matplotlib.pyplot as plt
import numpy as np

from functions import joint_torques, joint_torques_circle

# ===== Exercise 9: Joint Torques =====
F = np.array([0, 0, -1, 0, 0, 0])  # End-effector force/torque vector (N and Nm)
phi_vals = np.linspace(0, 2*np.pi, 37)  # 37 points around the circle
torques = joint_torques_circle(phi_vals, F)

# plot torques

plt.figure()
for i in range(torques.shape[1]):
    plt.plot(phi_vals, torques[:, i], label=f'Joint {i+1}')
plt.xlabel('φ (rad)')
plt.ylabel('Torque (Nm)')
plt.title('Joint Torques around the Circle')
plt.legend()
plt.grid(True)
plt.show()