import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# 设置全局字体为 SimHei (黑体) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 1. 基础参数与哈密顿量初始化

a = 5 / 100 # 格点间距 (0.05)
N = 100 # 格点数
xx = np.arange(N) * a + a/2 # 格点坐标（中心在每个格点中间）

# 初始化哈密顿量
H = np.zeros((N, N))
for i in range(N):
    H[i, i] = 1 / a**2 # 对角项（动能）
for i in range(N-1):
    H[i, i+1] = H[i+1, i] = -1 / (2 * a**2) # 近邻跃迁

# 2. 原始系统本征态求解

unsort_e, unsort_wf = np.linalg.eig(H)
energy = np.sort(unsort_e)
wf = unsort_wf[:, np.argsort(unsort_e)]


# 3. 定义双峰软核势场扰动 (中心在x=2.5)
def double_well_soft(X, X0, d, c, V0):
    """
    双峰软核势函数:
        X: 空间坐标数组
        X0: 系统中心位置
        d: 两势阱间距
        c: 软化参数 (避免奇点)
        V0: 势阱深度 (负值)
    """
    term1 = -V0 / np.sqrt((X - (X0 - d/2))**2 + c)
    term2 = -V0 / np.sqrt((X - (X0 + d/2))**2 + c)
    return term1 + term2

# 双峰势参数

X0 = xx[N//2]  # 系统中心 (x=2.5)
d = 1.0        # 两势阱间距
c = 0.01       # 软化参数
V0 = 50        # 势阱深度

# 计算双峰势
vx_soft = double_well_soft(xx, X0, d, c, V0)

H_soft = H.copy()
for i in range(N):
    H_soft[i, i] += vx_soft[i] # 添加软核势

unsort_e_soft, unsort_wf_soft = np.linalg.eig(H_soft)
energy_soft = np.sort(unsort_e_soft)
wf_soft = unsort_wf_soft[:, np.argsort(unsort_e_soft)]

# 绘制软核势场与波函数
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(xx, vx_soft,'k-', linewidth=2)
plt.title("Soft-Core Potential")

plt.subplot(1, 2, 2)
plt.plot(xx, wf_soft[:, 0], label='Ground state')
plt.plot(xx, wf_soft[:, 1], label='1st excited')
plt.plot(xx, wf_soft[:, 2], label='2nd excited')
plt.legend()
plt.title("Wavefunctions")
plt.tight_layout()
plt.show()

# 5. 态密度(DOS)与投影态密度(PDOS)计算
def g(E, E0, sigma):
    return np.exp(-((E - E0)/sigma)**2) # 高斯展宽函数

E = np.linspace(0, 12, 100) # 能量范围0-12

# 计算原始系统的总DOS (前7个能级)
spectrum_total = np.zeros_like(E)
for E0 in energy[:7]:
    spectrum_total += g(E, E0, 0.1)

# 计算PDOS：软核势p带在原始系统前7个态上的投影
c1 = np.sum(wf_soft[:, 3] * wf[:, 0]) # 与原始基态重叠
c2 = np.sum(wf_soft[:, 3] * wf[:, 1]) # 与第一激发态重叠
c3 = np.sum(wf_soft[:, 3] * wf[:, 2])
c4 = np.sum(wf_soft[:, 3] * wf[:, 3])
c5 = np.sum(wf_soft[:, 3] * wf[:, 4])
c6 = np.sum(wf_soft[:, 3] * wf[:, 5])
c7 = np.sum(wf_soft[:, 3] * wf[:, 6])
c = (c1, c2, c3, c4, c5, c6, c7)

spectrum_proj = np.zeros_like(E)
for i in range(7):
    spectrum_proj += c[i] * g(E, energy[i], 0.1)


# 6. 宽峰PDOS计算 (sigma=0.8)

spectrum_total_wide = np.zeros_like(E)
spectrum_proj_wide = np.zeros_like(E)
for i in range(7):
    spectrum_total_wide += g(E, energy[i], 0.8)
    spectrum_proj_wide += c[i] * g(E, energy[i], 0.8)


#绘图
# 创建一个大图包含多个子图
plt.figure(figsize=(14, 10))

# 子图1：窄峰展宽(sigma=0.1)
plt.subplot(2, 2, 1)
plt.plot(E, spectrum_total, 'b-', linewidth=2, label='总态密度(DOS)')
plt.plot(E, spectrum_proj, 'r--', linewidth=2, label='投影态密度(PDOS)')
plt.title('(a) 窄峰展宽(σ=0.1)', fontsize=12, pad=10)
plt.xlabel('能量(E)', fontsize=10)
plt.ylabel('密度', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加文字说明
plt.text(8, 0.8*max(spectrum_total),
         'PDOS计算的是双势阱的\n第三激发态($\psi_3$)在\n原始系统能级上的投影',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# 子图2：宽峰展宽(sigma=0.8)
plt.subplot(2, 2, 2)
plt.plot(E, spectrum_total_wide, 'b-', linewidth=2, label='总态密度(DOS)')
plt.plot(E, spectrum_proj_wide, 'r--', linewidth=2, label='投影态密度(PDOS)')
plt.title('(b) 宽峰展宽(σ=0.8)', fontsize=12, pad=10)
plt.xlabel('能量(E)', fontsize=10)
plt.ylabel('密度', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加文字说明
plt.text(8, 0.8*max(spectrum_total_wide),
         '宽峰展宽使能级峰\n相互重叠，更适合\n展示整体特征',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# 子图3：PDOS不同展宽对比
plt.subplot(2, 2, 3)
plt.plot(E, spectrum_proj, 'g-', linewidth=1.5, label='窄峰PDOS(σ=0.1)')
plt.plot(E, spectrum_proj_wide, 'm--', linewidth=2, label='宽峰PDOS(σ=0.8)')
plt.title('(c) 不同展宽下的PDOS对比', fontsize=12, pad=10)
plt.xlabel('能量(E)', fontsize=10)
plt.ylabel('密度', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# 子图4：能级位置标记
plt.subplot(2, 2, 4)
for i in range(7):
    plt.axvline(x=energy[i], color='gray', linestyle=':', alpha=0.7)
plt.plot(E, spectrum_proj, 'r-', linewidth=1.5, label='PDOS')
plt.title('(d) 原始系统能级位置', fontsize=12, pad=10)
plt.xlabel('能量(E)', fontsize=10)
plt.ylabel('密度', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加能级标记
for i, en in enumerate(energy[:7]):
    plt.text(en, max(spectrum_proj)*0.9, f'E{i}',
             ha='center', va='bottom', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.7))

# 调整布局
plt.tight_layout()
plt.suptitle('双势阱系统的态密度分析', fontsize=14, y=1.02)
plt.show()
