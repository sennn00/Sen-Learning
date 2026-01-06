import numpy as np

def solve_schrodinger(N=100, a=5 / 100, potential_func=None, potential_args=None):
    """
    解一维薛定谔方程（离散格点法），返回本征能量和波函数

    参数：
        N (int): 格点数（默认100）
        a (float): 格点间距（默认0.05）
        potential_func (callable): 势函数，形式为 V(x, **args)（默认None，自由粒子）
        potential_args (dict): 势函数的参数字典（默认None）

    返回：
        tuple: (energies, wavefunctions, xx)
            - energies: 本征能量数组（升序排列）
            - wavefunctions: 波函数数组，每列对应一个本征态
            - xx: 格点坐标数组
    """
    # 初始化格点
    xx = np.arange(N) * a + a / 2  # 格点坐标（中心在格点中间）

    # 初始化哈密顿量（动能项）
    H = np.zeros((N, N))
    np.fill_diagonal(H, 1 / a ** 2)  # 对角项
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -1 / (2 * a ** 2)  # 近邻跃迁

    # 添加势能项
    if potential_func is not None:
        if potential_args is None:
            potential_args = {}
        vx = potential_func(xx, **potential_args)
        np.fill_diagonal(H, H.diagonal() + vx)

    # 解本征问题
    energies, wavefunctions = np.linalg.eigh(H)

    return (energies, wavefunctions, xx)

