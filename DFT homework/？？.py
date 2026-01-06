import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def generate_ethylene_input(bond_length, output_dir="ethylene_scan"):
    """
    生成乙烯分子(C2H4)的Gaussian输入文件
    扫描C=C键长，其他几何参数保持合理值

    参数:
        bond_length (float): C=C键长(Å)
        output_dir (str): 输出目录

    返回:
        input_file (str): 生成的输入文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 乙烯分子几何结构（HCH角度=117.4°，其他CH键长=1.09Å）
    angle = 117.4  # H-C-H角度(度)
    ch_length = 1.09  # C-H键长(Å)

    # 计算原子坐标(C=C沿z轴)
    z_offset = bond_length / 2
    x_offset = ch_length * np.sin(np.radians(angle / 2))
    y_offset = ch_length * np.cos(np.radians(angle / 2))

    # 输入文件内容
    input_content = f"""%nprocshared=4
%mem=4GB
#p rb3lyp/6-31g(d) nosymm

Ethylene C=C scan at {bond_length:.3f} Angstrom

0 1
C       0.000000    0.000000    {z_offset:.6f}
H       {x_offset:.6f}    0.000000    {z_offset + ch_length:.6f}
H       {-x_offset:.6f}    0.000000    {z_offset + ch_length:.6f}
C       0.000000    0.000000    {-z_offset:.6f}
H       0.000000    {y_offset:.6f}    {-z_offset - ch_length:.6f}
H       0.000000    {-y_offset:.6f}    {-z_offset - ch_length:.6f}

"""
    # 写入文件
    input_file = os.path.join(output_dir, f"ethylene_{bond_length:.3f}.com")
    with open(input_file, 'w') as f:
        f.write(input_content)

    return input_file



def run_gaussian(input_file):
    output_file = input_file.replace('.com', '.log')
    gaussian_exe = r"D:\guassian\g16w\g16.exe"  # 替换为你的路径

    # 3. 设置环境变量
    env = os.environ.copy()
    env["GAUSS_EXEDIR"] = os.path.dirname(gaussian_exe)
    env["GAUSS_SCRDIR"] = os.path.abspath("scratch")  # 临时文件目录

    cmd = f'"{gaussian_exe}" < "{input_file}" > "{output_file}"'
    subprocess.run(cmd, shell=True, check=True, env=env)
    return output_file


def extract_energy(output_file):
    """
    从Gaussian输出文件中提取能量(E(RB3LYP))

    参数:
        output_file (str): 输出文件路径

    返回:
        energy (float): 提取的能量值(Hartree)
    """
    energy = None
    with open(output_file, 'r') as f:
        for line in f:
            if "E(RB3LYP)" in line:
                energy = float(line.split()[4])
    return energy


def scan_potential_energy_surface(start=1.20, end=1.50, steps=10):
    """
    扫描乙烯分子的C=C键长，构建势能面

    参数:
        start (float): 起始键长(Å)
        end (float): 终止键长(Å)
        steps (int): 扫描步数

    返回:
        results (dict): 包含键长和对应能量的字典
    """
    bond_lengths = np.linspace(start, end, steps)
    results = {'bond_lengths': [], 'energies': []}

    print("Starting ethylene C=C bond length scan...")
    print(f"Range: {start:.2f} - {end:.2f} Å, Steps: {steps}")
    print("-" * 50)

    for bl in bond_lengths:
        # 生成输入文件
        input_file = generate_ethylene_input(bl)

        # 运行Gaussian计算
        print(f"Running calculation for C=C = {bl:.3f} Å...")
        output_file = run_gaussian(input_file)

        # 提取能量
        energy = extract_energy(output_file)
        if energy is not None:
            results['bond_lengths'].append(bl)
            results['energies'].append(energy)
            print(f"C=C = {bl:.3f} Å: Energy = {energy:.8f} Hartree")
        else:
            print(f"Warning: Failed to extract energy for C=C = {bl:.3f} Å")

        print("-" * 50)

    return results


def plot_potential_energy_surface(results):
    """
    绘制势能面曲线

    参数:
        results (dict): 包含键长和能量的字典
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['bond_lengths'], results['energies'], 'bo-')

    # 找到最小能量点
    min_idx = np.argmin(results['energies'])
    plt.plot(results['bond_lengths'][min_idx], results['energies'][min_idx],
             'ro', markersize=10, label=f'Minimum ({results["bond_lengths"][min_idx]:.3f} Å)')

    plt.xlabel('C=C Bond Length (Å)')
    plt.ylabel('Energy (Hartree)')
    plt.title('Ethylene Potential Energy Surface Scan (RB3LYP/6-31G(d))')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('ethylene_pes_scan.png')
    plt.show()


if __name__ == "__main__":
    # 扫描范围(1.20Å到1.50Å，共10个点)
    scan_results = scan_potential_energy_surface(start=1.20, end=1.50, steps=10)



# 保存结果到文件
with open('ethylene_scan_results.txt', 'w') as f:
    f.write("C=C Bond Length (Å)\tEnergy (Hartree)\n")
    for bl, energy in zip(scan_results['bond_lengths'], scan_results['energies']):
        f.write(f"{bl:.4f}\t{energy:.8f}\n")

print("Scan completed! Results saved to ethylene_scan_results.txt")
