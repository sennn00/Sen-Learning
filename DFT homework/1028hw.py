import numpy as np
import matplotlib.pyplot as plt
from function import solve_schrodinger

def double_softcore_potential(x, x1=1.0, x2=3.0, c=0.0001, v1=1.0, v2=1.0):
    """双峰软核势"""
    return -v1 / np.sqrt((x - x1)**2 + c) - v2 / np.sqrt((x - x2)**2 + c)

# 参数设置
N = 1000
a = 5 / N  # 格点间距
x1, x2 = 1.2, 3.8  # 双峰位置
c = 0.0001        # 软核参数

# 1. 解薛定谔方程（使用双峰软核势）
energies, wavefunctions, xx = solve_schrodinger(
    N=N,
    a=a,
    potential_func=double_softcore_potential,
    potential_args={'x1': x1, 'x2': x2, 'c': c}
)

# 2. 计算电子密度
n = np.sum(wavefunctions[:, :3]**2, axis=1)

# 3. 方法1：傅里叶变换法计算 Hartree 势 (V_H_fft)
nk = np.fft.fft(n)
k = 2 * np.pi * np.fft.fftfreq(N, a)
vhk = -4 * np.pi * nk / (k**2 + 1e-10)  # 避免 k=0 处除零
vhk[0] = 0  # 忽略全局常数势
V_H_fft = np.real(np.fft.ifft(vhk))

# 4. 方法2：直接积分法计算 Hartree 势 (V_H_direct)
normalized_n = n / a  # 归一化电子密度
V_H_direct = np.zeros(N)
for i in range(1000):
    y = np.zeros(1000)
    for j in range(1000):
        y[j] = n[j] / np.sqrt((xx[i] - xx[j]) ** 2 + 0.0001)
    V_H_direct[i] = np.sum(y) * a

# 5. 计算 ESP (使用直接积分法的 V_H)
V_x = double_softcore_potential(xx, x1=x1, x2=x2, c=c)
ESP = -V_x + V_H_direct  # 静电势

# 6. 绘制结果
plt.figure(figsize=(12, 12))

# 子图1：双峰软核势 V_x
plt.subplot(3, 2, 1)
plt.plot(xx, V_x, 'g', label='Softcore Potential ($V_x$)')
plt.xlabel('Position (x)')
plt.ylabel('Potential')
plt.title('(a) Double Softcore Potential')
plt.grid(True)
plt.legend()

# 子图2：傅里叶法计算的 Hartree 势
plt.subplot(3, 2, 3)
plt.plot(xx, V_H_fft, 'b', label='$V_H$ (FFT)')
plt.xlabel('Position (x)')
plt.ylabel('Potential')
plt.title('(b) Hartree Potential (FFT Method)')
plt.grid(True)
plt.legend()

# 子图3：直接积分法计算的 Hartree 势
plt.subplot(3, 2, 4)  # 第4位置
plt.plot(xx, V_H_direct, 'r', label='$V_H$ (Direct)')
plt.xlabel('Position (x)')
plt.ylabel('Potential')
plt.title('(c) Hartree Potential (Direct Method)')
plt.grid(True)
plt.legend()

# 子图4：直接积分法计算的 ESP
plt.subplot(3, 2, 5)  # 第5位置
plt.plot(xx, ESP, 'm', label='ESP ($-V_x + V_H$)')
plt.xlabel('Position (x)')
plt.ylabel('Potential')
plt.title('(d) Electrostatic Potential')
plt.grid(True)
plt.legend()

# 子图5：电子密度 n(x)
plt.subplot(3, 2, 6)  # 第6位置
plt.plot(xx, n, 'k', label='Electron Density ($n$)')
plt.xlabel('Position (x)')
plt.ylabel('Density')
plt.title('(e) Electron Density')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
