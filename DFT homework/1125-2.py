import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# 1. 定义甲烷分子的能量数据（来自您提供的7x2矩阵）
energies = np.array([
    [-39.97201539, -39.95301636],
    [-40.10463650, -40.08460426],
    [-40.16348060, -40.14337650],
    [-40.17672672, -40.15724975],
    [-40.16228520, -40.14391060],
    [-40.13166928, -40.11470677],
    [-40.09225703, -40.10182755]
])

# 2. 定义扫描范围（C-H键长和H-C-H角度）
ch_lengths = np.arange(1.00, 1.61, 0.10)  # C-H键长: 1.00-1.60 Å, 步长0.10 Å
angles = np.arange(100, 106, 5)          # H-C-H角度: 100-105度, 步长5度

# 3. 转换能量单位并相对化
energies_kcal = energies * 627.509  # Hartree转kcal/mol
min_energy = np.min(energies_kcal)
energies_kcal = energies_kcal - min_energy  # 使最小能量为0

# 4. 创建网格
CH, ANG = np.meshgrid(ch_lengths, angles)

# 5. 设置全局字体（解决Å显示问题）
rcParams['font.family'] = 'Arial'
rcParams['mathtext.fontset'] = 'stix'

# 6. 创建自定义颜色映射（蓝色=低能量，红色=高能量）
colors = ['#0000ff', '#4444ff', '#8888ff', '#bbbbff', '#ffffff', '#ffbbbb', '#ff8888', '#ff4444', '#ff0000']
cmap = LinearSegmentedColormap.from_list('energy_cmap', colors)

# --------------------------
# 图1: 热力图（Heatmap）
# --------------------------
plt.figure(figsize=(10, 8))

# 绘制热力图
im = plt.imshow(energies_kcal.T, cmap=cmap, origin='lower',
                extent=[ch_lengths[0], ch_lengths[-1], angles[0], angles[-1]],
                aspect='auto')

# 标记最低能量点
min_idx = np.unravel_index(np.argmin(energies_kcal), energies_kcal.shape)
plt.scatter(ch_lengths[min_idx[0]], angles[min_idx[1]],
           color='yellow', s=150, marker='*',
           label=f'Min Energy: {np.min(energies_kcal):.1f} kcal/mol')

# 添加标签和颜色条
cbar = plt.colorbar(im)
cbar.set_label('Relative Energy (kcal/mol)', fontsize=12)
plt.xlabel(r'C-H Bond Length ($\mathrm{\AA}$)', fontsize=12, fontweight='bold')
plt.ylabel('H-C-H Angle (degrees)', fontsize=12, fontweight='bold')
plt.title('CH4 Potential Energy Surface (Heatmap)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# --------------------------
# 图2: 3D势能面
# --------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 创建3D曲面图
surf = ax.plot_surface(CH, ANG, energies_kcal.T, cmap=cmap, alpha=0.85,
                      edgecolor='k', linewidth=0.5)

# 标记最低能量点
ax.scatter(ch_lengths[min_idx[0]], angles[min_idx[1]], np.min(energies_kcal),
          color='yellow', s=200, marker='*',
          label=f'Min Energy: {np.min(energies_kcal):.1f} kcal/mol')

# 自定义3D图
ax.set_xlabel(r'C-H Bond Length ($\mathrm{\AA}$)', fontsize=12, fontweight='bold')
ax.set_ylabel('H-C-H Angle (degrees)', fontsize=12, fontweight='bold')
ax.set_zlabel('Relative Energy (kcal/mol)', fontsize=12, fontweight='bold')
ax.set_title('CH4 Potential Energy Surface (3D View)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('Relative Energy (kcal/mol)', fontsize=10)

# 调整视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# 保存图片（可选）
# plt.savefig('ch4_heatmap.png', dpi=300, bbox_inches='tight')
# fig.savefig('ch4_pes_3d.png', dpi=300, bbox_inches='tight')
