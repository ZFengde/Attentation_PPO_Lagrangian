import torch as th
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class gaussmf():
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def ante(self, x):
        return th.exp(-((x - self.mean)**2.) / (2 * self.sigma **2.))
    
class TS_Fuzzy(nn.Module):
    def __init__(self, rules_num):
        super().__init__()

        self.rules_num = rules_num

        self.sub_systems_mat = th.tensor([[0, -0.05, -0.2, -0.1, -0.25, -0.02, -0.2, -0.05, 0], 
                                        [0, -0.0011, -0.0022, -0.0022, 0, -0.00044, -0.0011, -0.0011, 0]])
        self.sub_systems_bias = th.tensor([1, 1, 0.7, 1, 1, 0.22, 0.9, 0.35, 0]) 

        self._init_rules()

    def forward(self, x1, x2): # x1 = 72, 7
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        truth_values = self.ante_process(x1, x2) # 72, 7, 9, as coeffient

        premises = th.stack((x1, x2), dim=2).view(-1, 2).float() # 9, 72*7, 2
        consequence = th.matmul(premises, self.sub_systems_mat) + self.sub_systems_bias
        consequence = consequence.view(x1.shape[0], x1.shape[1], self.rules_num)

        output = th.sum((truth_values * consequence), dim=2) / th.sum(truth_values, dim=2)
        return output

    def ante_process(self, x1, x2):
        # see if here can be batch operations, but not very important
        x1_s_level = self.x1_s.ante(x1)
        x1_m_level = self.x1_m.ante(x1)
        x1_l_level = self.x1_l.ante(x1)

        x2_s_level = self.x2_s.ante(x2)
        x2_m_level = self.x2_m.ante(x2)
        x2_l_level = self.x2_l.ante(x2)

        truth_values = th.stack((th.min(x1_s_level, x2_s_level),
                         th.min(x1_s_level, x2_m_level),
                         th.min(x1_s_level, x2_l_level), 
                         th.min(x1_m_level, x2_s_level), 
                         th.min(x1_m_level, x2_m_level), 
                         th.min(x1_m_level, x2_l_level),
                         th.min(x1_l_level, x2_s_level),
                         th.min(x1_l_level, x2_m_level),
                         th.min(x1_l_level, x2_l_level)), dim=2) 

        return truth_values

    def _init_rules(self):
        self.x1_s = gaussmf(0, 0.75) # mean and sigma
        self.x1_m = gaussmf(2, 0.75)
        self.x1_l = gaussmf(4, 0.75)

        self.x2_s = gaussmf(0, 30) # mean and sigma
        self.x2_m = gaussmf(90, 30) # mean and sigma
        self.x2_l = gaussmf(180, 30) # mean and sigma



import numpy as np
import matplotlib.pyplot as plt
import math

# We can simulate at higher resolution with full accuracy
a = np.linspace(0, 4, 40)
b = np.linspace(0, 180, 40)
x, y = np.meshgrid(a, b)
z = np.zeros_like(x)

fuzzy_simulator = TS_Fuzzy(rules_num=9)

# Loop through the system 21*21 times to collect the control surface
for i in range(40):
    for j in range(40):
        output = fuzzy_simulator(th.tensor([a[i]]), th.tensor([b[j]])).float()
        z[j, i] = output

# Plot the result in pretty 3D with alpha blending
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)

x1 = th.tensor([5]) # should be in range (0, 4)
x2 = th.tensor([180])  # should be in range (0, 180)

output_simu = fuzzy_simulator(x1, x2).float()
print(output_simu)

ax.view_init(40, 200)
plt.show()
