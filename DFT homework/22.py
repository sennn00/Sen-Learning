import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

#设置矩阵
a = 5 / 100
H = np.zeros((100, 100))

# 矩阵元
for i in range(100):
    H[i, i] = 1 / a**2

for i in range(99):
    H[i, i+1] = -1 / (2 * a**2)
    H[i+1, i] = -1 / (2 * a**2)

#设置势阱
for i in range(30, 50):
    H[i, i] = 1 / a**2 - 10

for i in range(90, 100):
    H[i, i] = 1 / a**2 - 8

#势阱
v = np.zeros(100)
for i in range(90, 100):
    v[i] = -8

for i in range(30, 50):
    v[i] = -10

# 解本征值
energy, wf = np.linalg.eig(H)

#按能量排序波函数
energy = np.sort(energy)
wf = wf[:, np.argsort(energy)]
print(energy)
print(wf)
#绘图
plt.plot(wf[:, 0]**2 + wf[:, 1]**2 + wf[:, 2]**2)
plt.plot(v/100)
plt.show()

plt.plot(wf[:, 0]**2 + wf[:, 1]**2 + wf[:, 2]**2 + wf[:, 3]**2)
plt.plot(v/100)
plt.show()

plt.plot(wf[:, 0]**2 + wf[:, 1]**2 + wf[:, 2]**2 + wf[:, 3]**2 + wf[:, 4]**2)
plt.plot(v/100)
plt.show()
