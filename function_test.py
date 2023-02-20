import torch as th
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import math

# We can simulate at higher resolution with full accuracy
A_space = np.linspace(-3, 3, 40)
r_space = np.linspace(0, 2, 40)
x, y = np.meshgrid(A_space, r_space)
z = np.zeros_like(x)

def simulator(A, r):
    policy_loss_1 = A * r
    policy_loss_2 = A * th.clamp(r, 1 - 0.2, 1 + 0.2)
    return th.min(policy_loss_1, policy_loss_2).mean()

# Loop through the system 21*21 times to collect the control surface
for i in range(40):
    for j in range(40):
        A_i = th.tensor([A_space[i]])
        r_j = th.tensor([r_space[j]])
        output = simulator(A=A_i, r=r_j).float()
        z[j, i] = output

# Plot the result in pretty 3D with alpha blending
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)

print(simulator(A=th.tensor(3), r=th.tensor(1)))
print(simulator(A=th.tensor(3), r=th.tensor(1.2)))
ax.view_init(40, 200)
plt.show()
