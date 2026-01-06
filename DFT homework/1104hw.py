import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def softcore_potential(x, x0, c=0.0001):
    """软核势函数"""
    return -1.0 / np.sqrt((x - x0) ** 2 + c)


def solve_schrodinger(N, a, potential):
    """求解薛定谔方程"""
    # 构建哈密顿量
    diag = np.ones(N) / a ** 2 + potential
    off_diag = -0.5 * np.ones(N - 1) / a ** 2
    H = diags([diag, off_diag, off_diag], [0, 1, -1])

    # 计算最低3个本征态
    energies, wavefunctions = eigsh(H, k=3, which='SA')
    return energies, wavefunctions, np.linspace(0, N * a, N)


def calc_hartree_potential(n, xx, a):
    """计算Hartree势（直接积分法）"""
    vh = np.zeros_like(xx)
    for i in range(len(xx)):
        r = np.sqrt((xx[i] - xx) ** 2 + 0.0001)
        vh[i] = np.sum(n / r) * a
    return vh


def normalize_density(n, a, nelec=3):
    """归一化电子密度"""
    return nelec * n / (np.sum(n) * a)


# 参数设置
N = 1000  # 格点数
a = 5 / N  # 格点间距
xx = np.linspace(0, N * a, N)

# 初始化外势场（双峰势）
V_nuc = 2 * softcore_potential(xx, 1.2) + softcore_potential(xx, 3.8)

# 初始解：非相互作用电子
energies, wavefunctions, _ = solve_schrodinger(N, a, V_nuc)
n = normalize_density(np.sum(wavefunctions ** 2, axis=1), a)

# 自洽迭代
max_iter = 100
conv_threshold = 1e-4
history = []

for it in range(max_iter):
    # 计算Hartree势
    V_H = calc_hartree_potential(n, xx, a)

    # 更新哈密顿量
    total_potential = V_nuc + V_H
    energies, wavefunctions, _ = solve_schrodinger(N, a, total_potential)

    # 计算新密度并混合
    n_new = normalize_density(np.sum(wavefunctions ** 2, axis=1), a)
    density_change = np.mean(np.abs(n_new - n))
    n = 0.5 * n + 0.5 * n_new  # 线性混合

    # 记录总能量
    tot_energy = np.sum(energies[:3])
    history.append(tot_energy)

    print(f"Iter {it}: E_tot = {tot_energy:.6f}, Δn = {density_change:.6f}")

    # 收敛判断
    if density_change < conv_threshold:
        print(f"Converged after {it} iterations!")
        break

# 可视化结果
plt.figure(figsize=(12, 8))

# 电子密度对比
plt.subplot(2, 2, 1)
plt.plot(xx, n, 'b-', label='Self-consistent')
plt.xlabel('Position')
plt.ylabel('Electron Density')
plt.title('Electron Density Comparison')
plt.grid(True)

# 势能对比
plt.subplot(2, 2, 2)
plt.plot(xx, V_nuc, 'g-', label='Nuclear Potential')
plt.plot(xx, V_H, 'r-', label='Hartree Potential')
plt.xlabel('Position')
plt.ylabel('Potential')
plt.title('Potential Comparison')
plt.legend()
plt.grid(True)

# 波函数
plt.subplot(2, 2, 3)
for i in range(3):
    plt.plot(xx, wavefunctions[:, i], label=f'State {i}')
plt.xlabel('Position')
plt.ylabel('Wavefunction')
plt.title('Wavefunctions')
plt.legend()
plt.grid(True)

# 能量收敛
plt.subplot(2, 2, 4)
plt.plot(history, 'ko-')
plt.xlabel('Iteration')
plt.ylabel('Total Energy')
plt.title('Energy Convergence')
plt.grid(True)

plt.tight_layout()
plt.show()