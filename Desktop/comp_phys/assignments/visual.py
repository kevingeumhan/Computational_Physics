import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the standard axes
ax.plot([-1.5, 1.5], [0, 0], [0, 0], color='gray', linestyle='--') # X-axis
ax.plot([0, 0], [-1.5, 1.5], [0, 0], color='gray', linestyle='--') # Y-axis
ax.plot([0, 0], [0, 0], [-1.5, 1.5], color='gray', linestyle='--') # Z-axis

# Draw the Hadamard "Skewer" (Axis of rotation: x + z)
ax.plot([-1.5, 1.5], [0, 0], [-1.5, 1.5], color='green', linewidth=2, label='Hadamard Skewer (Rotation Axis)')

# Plot Original Y-Vector
ax.quiver(0, 0, 0, 0, 1, 0, color='blue', linewidth=3, arrow_length_ratio=0.1, label='Original +Y')

# Plot Transformed Y-Vector
ax.quiver(0, 0, 0, 0, -1, 0, color='red', linewidth=3, arrow_length_ratio=0.1, label='Transformed -Y (After 180° Spin)')

# Formatting
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('The Hadamard 180° Geometric Spin')
ax.legend()

plt.show()