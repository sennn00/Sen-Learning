import numpy as np
import matplotlib.pyplot as plt
from function import solve_schrodinger


def double_softcore_potential(x, x1=1.0, x2=3.0, c=0.0001, v1=2.0, v2=1.0):
    """双峰软核势"""
    return -v1 / np.sqrt((x - x1)**2 + c) - v2 / np.sqrt((x - x2)**2 + c)


def calc_hartree_potential(n, xx, a):
    """计算Hartree势（直接积分法）"""
    vh = np.zeros_like(xx)
    for i in range(len(xx)):
        r = np.sqrt((xx[i] - xx) ** 2 + 0.0001)  # 避免除零
        vh[i] = np.sum(n / r) * a
    return vh

def normalize_density(n, a, nelec=3):
    """归一化电子密度"""
    return nelec * n / (np.sum(n) * a)


# 参数设置
N = 1000  # 格点数
a = 5 / N  # 格点间距
nelec = 3  # 电子数
conv_threshold = 1e-3  # 收敛阈值
mix_factor = 0.3  # 密度混合因子

# 初始化
xx = np.arange(N) * a + a / 2  # 格点坐标（与solve_schrodinger一致）
V_nuc = double_softcore_potential(xx)  # 核势

# 初始解：非相互作用系统
energies, wavefunctions, _ = solve_schrodinger(
    N=N, a=a,
    potential_func=double_softcore_potential,
    potential_args={'x1': 1.2, 'x2': 3.8}
)
n = normalize_density(np.sum(wavefunctions[:, :nelec] ** 2, axis=1), a, nelec)

# 自洽迭代
history = []
for it in range(20):
    # 计算Hartree势
    V_H = calc_hartree_potential(n, xx, a)

    # 求解新哈密顿量
    energies, wavefunctions, _ = solve_schrodinger(
        N=N, a=a,
        potential_func=lambda x: double_softcore_potential(x) + V_H[np.argmin(np.abs(xx[:, None] - x), axis=0)]
    )

    # 更新密度
    n_new = normalize_density(np.sum(wavefunctions[:, :nelec] ** 2, axis=1), a, nelec)
    delta_n = np.mean(np.abs(n_new - n))

    # 密度混合
    n = (1 - mix_factor) * n + mix_factor * n_new

    # 记录总能量
    tot_energy = np.sum(energies[:nelec])
    history.append(tot_energy)

    print(f"Iter {it:2d}: E_tot = {tot_energy:.8f}, Δρ = {delta_n:.3e}")

    # 收敛判断
    if delta_n < conv_threshold and it > 5:
        print(f"\nConverged after {it} iterations!")
        break

# 可视化结果
plt.figure(figsize=(14, 10))

# 电子密度
plt.subplot(2, 2, 1)
plt.plot(xx, n, 'b-', linewidth=2)
plt.xlabel('Position (x)', fontsize=12)
plt.ylabel('Electron Density (n)', fontsize=12)
plt.title('Self-consistent Electron Density', fontsize=14)
plt.grid(True)

# 势能曲线
plt.subplot(2, 2, 2)
plt.plot(xx, V_nuc, 'g-', label='Nuclear Potential')
plt.plot(xx, V_H, 'r--', label='Hartree Potential')
plt.plot(xx, V_nuc + V_H, 'k:', label='Total Potential')
plt.xlabel('Position (x)', fontsize=12)
plt.ylabel('Potential Energy', fontsize=12)
plt.title('Potential Energy Landscape', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# 波函数
plt.subplot(2, 2, 3)
for i in range(3):
    plt.plot(xx, wavefunctions[:, i], label=f'State {i}')
plt.xlabel('Position (x)', fontsize=12)
plt.ylabel('Wavefunction', fontsize=12)
plt.title('First Three Wavefunctions', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# 能量收敛
plt.subplot(2, 2, 4)
plt.plot(history, 'ko-')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Total Energy', fontsize=12)
plt.title('Energy Convergence History', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()
