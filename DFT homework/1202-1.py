import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# ======================
# 用户可调参数
# ======================
# 交换和关联比例的扫描范围 (0.0~1.0)
ex_range = np.arange(0.0, 1.1, 0.2)  # 交换部分 local 比例
corr_range = np.arange(0.0, 1.1, 0.2)  # 关联部分 local 比例
gaussian_path = r"D:\guassian\g16w\g16.exe"  # Gaussian 可执行文件路径
output_dir = "hcn_xc_scan_results"  # 结果保存目录
scratch_dir = r"D:\GaussianTemp"  # Gaussian临时目录
# ======================

# 初始化环境变量
os.environ["GAUSS_EXEDIR"] = r"D:\guassian\g16w"
os.environ["GAUSS_SCRDIR"] = scratch_dir
os.makedirs(scratch_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def generate_gaussian_input(ex, corr):
    """生成HCN输入文件"""
    if not (0.0 <= ex <= 1.0) or not (0.0 <= corr <= 1.0):
        raise ValueError("ex and corr must be between 0 and 1")

    # 计算IOp参数（格式化为5位）
    ex_local = min(10000, max(0, int(round(ex * 10000))))
    ex_nonlocal = 10000 - ex_local
    corr_local = min(10000, max(0, int(round(corr * 10000))))
    corr_nonlocal = 10000 - corr_local

    # 确保总长度为10位
    iop_77 = f"{ex_nonlocal:05d}{ex_local:05d}"
    iop_78 = f"{corr_nonlocal:05d}{corr_local:05d}"

    template = f"""%Mem=6GB
%NProcShared=4
#P B3LYP/6-31G(d) SP IOp(3/74=609,3/77={iop_77},3/78={iop_78})

HCN hybrid functional scan: ex={ex:.2f}, corr={corr:.2f}

0 1
H     -1.060000     0.000000     0.000000
C      0.000000     0.000000     0.000000
N      1.160000     0.000000     0.000000

"""
    filename = os.path.join(output_dir, f"hcn_EX{ex:.2f}_CORR{corr:.2f}.com")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(template)
    return filename


def run_gaussian(input_file):
    """运行Gaussian计算"""
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在！")
        return None

    log_file = os.path.splitext(input_file)[0] + ".log"

    try:
        cmd = [gaussian_path, input_file, log_file]
        print(f"运行计算: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=True,
            encoding='gbk',
            errors='ignore'
        )

        time.sleep(2)  # 等待文件写入完成

        if not os.path.exists(log_file) or os.path.getsize(log_file) < 500:
            print(f"警告: 日志文件生成失败或过小")
            return None

        print(f"计算完成: {log_file}")
        return log_file

    except subprocess.CalledProcessError as e:
        print(f"错误: Gaussian计算失败！错误码 {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return None
    except Exception as e:
        print(f"错误: 运行Gaussian时发生异常 - {str(e)}")
        return None


def extract_energy(log_file):
    """提取SCF能量"""
    if not os.path.exists(log_file):
        print(f"错误: 日志文件 {log_file} 不存在！")
        return np.nan

    try:
        with open(log_file, 'r', encoding='gbk') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"无法读取文件 {log_file}: {str(e)}")
            return np.nan

    if "Normal termination" not in content:
        print(f"警告: {os.path.basename(log_file)} 计算未正常终止！")
        return np.nan

    energy_match = re.search(
        r"SCF Done:\s+E\(.*\)\s+=\s+([-+]?\d+\.\d+)",
        content
    )

    if energy_match:
        try:
            energy = float(energy_match.group(1))
            print(f"成功提取能量: {energy:.8f} a.u. ({os.path.basename(log_file)})")
            return energy
        except ValueError:
            pass

    print(f"错误: 未找到SCF能量 in {os.path.basename(log_file)}")
    return np.nan


def skip_existing_calculations(ex, corr):
    """跳过已完成的计算"""
    log_filename = f"hcn_EX{ex:.2f}_CORR{corr:.2f}.log"
    log_file = os.path.join(output_dir, log_filename)

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='gbk', errors='ignore') as f:
                if "Normal termination" in f.read():
                    print(f"跳过已完成计算: {log_file}")
                    return True
        except Exception as e:
            print(f"读取日志文件错误: {str(e)}")
    return False


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    print("=" * 50)
    print("HCN 交换-关联比例势能面扫描程序")
    print("=" * 50)

    # (1) 生成输入文件并提交计算
    print("\n【步骤1：生成输入文件并提交计算】")
    total_calc = len(ex_range) * len(corr_range)
    completed_calc = 0

    energies = np.full((len(ex_range), len(corr_range)), np.nan)

    for i, ex in enumerate(ex_range):
        for j, corr in enumerate(corr_range):
            if skip_existing_calculations(ex, corr):
                completed_calc += 1
                energies[i, j] = extract_energy(os.path.join(output_dir, f"hcn_EX{ex:.2f}_CORR{corr:.2f}.log"))
                continue

            input_file = generate_gaussian_input(ex, corr)
            log_file = run_gaussian(input_file)
            if log_file:
                completed_calc += 1
                energies[i, j] = extract_energy(log_file)

            print(f"进度: {completed_calc}/{total_calc} 计算完成")
            print("-" * 30)

    valid_energy_count = np.count_nonzero(~np.isnan(energies))
    print(f"\n数据提取完成：有效能量 {valid_energy_count}/{total_calc} 个")

    # (3) 保存数据到文本文件
    print("\n【步骤3：保存数据】")
    header = f"交换比例范围: {ex_range[0]:.2f} - {ex_range[-1]:.2f} (步长{np.diff(ex_range)[0]:.2f})\n"
    header += f"关联比例范围: {corr_range[0]:.2f} - {corr_range[-1]:.2f} (步长{np.diff(corr_range)[0]:.2f})\n"
    header += "能量单位：a.u.（Hartree）\n"
    header += "列：关联比例 行：交换比例"

    np.savetxt(
        "hcn_xc_pes_energies.txt",
        energies,
        header=header,
        fmt="%.8f",
        encoding='utf-8'
    )
    print("能量数据已保存至：hcn_xc_pes_energies.txt")

    # (4) 绘制势能面
    print("\n【步骤4：绘制势能面】")
    if valid_energy_count == 0:
        print("错误: 没有有效的能量数据可绘制！")
    else:
        # 转换单位：Hartree -> kcal/mol
        energies_kcal = energies * 627.509
        min_energy = np.nanmin(energies_kcal)
        energies_kcal -= min_energy

        # 创建网格
        ex_grid, corr_grid = np.meshgrid(ex_range, corr_range, indexing='ij')

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(12, 5))

        # 3D曲面图
        ax1 = fig.add_subplot(121, projection='3d')
        mask = ~np.isnan(energies_kcal)
        surf = ax1.plot_surface(
            ex_grid[mask], corr_grid[mask], energies_kcal[mask],
            cmap='viridis', alpha=0.9, linewidth=0.1
        )
        ax1.set_xlabel("交换比例 (ex)", fontsize=10)
        ax1.set_ylabel("关联比例 (corr)", fontsize=10)
        ax1.set_zlabel("相对能量 (kcal/mol)", fontsize=10)
        ax1.set_title("HCN 交换-关联势能面", fontsize=12, pad=20)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='相对能量 (kcal/mol)')

        # 等高线图
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(
            ex_grid, corr_grid, energies_kcal,
            levels=25, cmap='viridis', alpha=0.9,
            extend='max'
        )
        ax2.set_xlabel("交换比例 (ex)", fontsize=10)
        ax2.set_ylabel("关联比例 (corr)", fontsize=10)
        ax2.set_title("势能面等高线图", fontsize=12, pad=20)
        plt.colorbar(contour, ax=ax2, label='相对能量 (kcal/mol)')

        plt.tight_layout()
        plt.savefig("hcn_xc_pes.png", dpi=300, bbox_inches='tight')
        plt.show()

    print("\n" + "=" * 50)
    print("程序完成！")
    print(f"结果文件：")
    print(f"  - 输入文件和日志：{os.path.abspath(output_dir)}")
    print(f"  - 能量数据：{os.path.abspath('hcn_xc_pes_energies.txt')}")
    print(f"  - 势能面图：{os.path.abspath('hcn_xc_pes.png')}")
    print("=" * 50)
