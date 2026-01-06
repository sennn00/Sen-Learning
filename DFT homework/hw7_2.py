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
r_co_range = np.arange(1.0, 1.6, 0.1)  # CO键长扫描范围：1.0 Å 到 1.5 Å，步长 0.1 Å
r_cs_range = np.arange(1.5, 2.1, 0.1)  # CS键长扫描范围：1.5 Å 到 2.0 Å，步长 0.1 Å
gaussian_path = r"D:\guassian\g16w\g16.exe"  # Gaussian 16 可执行文件绝对路径
output_dir = "scan_results"             # 结果保存目录
scratch_dir = r"D:\GaussianTemp"        # Gaussian临时目录（独立设置，避免权限问题）
# ======================

# 初始化环境变量（提前设置，确保所有函数可用）
os.environ["GAUSS_EXEDIR"] = r"D:\guassian\g16w"
os.environ["GAUSS_SCRDIR"] = scratch_dir
os.makedirs(scratch_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def generate_gaussian_input(r_co, r_cs):
    """
    生成 Gaussian 输入文件（移除特殊符号Å，修复编码冲突）
    势能面扫描核心：固定R(CO)和R(CS)，做单点能量计算（不优化键长）
    """
    # 验证键长合理性（避免极端值导致计算失败）
    if not (0.8 <= r_co <= 1.8):
        print(f"警告: C-O键长 {r_co:.2f} 超出合理范围（0.8-1.8 埃）")
    if not (1.4 <= r_cs <= 2.2):
        print(f"警告: C-S键长 {r_cs:.2f} 超出合理范围（1.4-2.2 埃）")
    rco = -r_co
    # 关键修改1：移除模板中的Å符号，改用英文描述（Gaussian输入无需单位，默认是Å）
    template = f"""%Mem=6GB
%NProcShared=4
#P B3LYP/6-31G(d) SP

Generated OCS-like molecule single-point calculation

0 1
C     0.000000     0.000000     0.000000
O     {rco:.6f}     0.000000     0.000000
S     {r_cs:.6f}     0.000000     0.000000

"""
    # 文件名避免特殊字符，使用绝对路径
    filename = os.path.join(output_dir, f"ocs_scan_CO{r_co:.2f}_CS{r_cs:.2f}.com")
    filename_abs = os.path.abspath(filename)
    
    # 关键修改2：改回UTF-8编码（无特殊符号时，Gaussian完全兼容UTF-8）
    with open(filename_abs, 'w', encoding='utf-8', newline='\n') as f:
        f.write(template)
    
    print(f"生成输入文件: {filename_abs}")
    return filename_abs

def run_gaussian(input_file):
    """
    运行Gaussian计算（Windows下稳定调用方式，添加错误捕获和文件验证）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在！")
        return None
    
    # 日志文件（与输入文件同目录，绝对路径）
    log_file = os.path.splitext(input_file)[0] + ".log"
    log_file_abs = os.path.abspath(log_file)
    
    # 检查Gaussian可执行文件是否存在
    if not os.path.exists(gaussian_path):
        print(f"错误: 未找到Gaussian可执行文件 {gaussian_path}")
        return None
    
    # 检查临时目录可写权限
    if not os.access(scratch_dir, os.W_OK):
        print(f"错误: 临时目录 {scratch_dir} 无写入权限！")
        return None
    
    try:
        # Windows下稳定调用格式：g16.exe 输入文件.com 输出文件.log
        cmd = [gaussian_path, input_file, log_file_abs]
        print(f"运行计算: {' '.join(cmd)}")
        
        # 执行命令，捕获错误输出，使用GBK编码
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=True,  # Windows下必须开启shell=True
            encoding='gbk',
            errors='ignore'
        )
        
        # 等待文件写入完成（避免日志未写完就读取）
        time.sleep(2)
        
        # 验证日志文件有效性（大小≥500字节，避免空文件）
        if os.path.getsize(log_file_abs) < 500:
            print(f"警告: 日志文件过小（{os.path.getsize(log_file_abs)} 字节），计算可能未完成")
            return None
        
        print(f"计算完成: {log_file_abs}")
        return log_file_abs
    
    except subprocess.CalledProcessError as e:
        print(f"错误: Gaussian计算失败！错误码 {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return None
    except Exception as e:
        print(f"错误: 运行Gaussian时发生异常 - {str(e)}")
        return None

def extract_energy(log_file):
    """专为RB3LYP优化的能量提取函数"""
    if not os.path.exists(log_file):
        print(f"错误: 日志文件 {log_file} 不存在！")
        return np.nan

    # 优先使用GBK编码（Windows Gaussian默认）
    try:
        with open(log_file, 'r', encoding='gbk') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

    # 严格验证计算是否正常终止
    if "Normal termination" not in content:
        print(f"警告: {os.path.basename(log_file)} 计算未正常终止！")
        return np.nan

    # 精准匹配RB3LYP能量行（示例：SCF Done:  E(RB3LYP) =  -511.272520339     A.U.）
    energy_match = re.search(
        r"SCF Done:\s+E\(RB3LYP\)\s+=\s+([-+]?\d+\.\d+)", 
        content
    )
    
    if energy_match:
        try:
            energy = float(energy_match.group(1))
            print(f"成功提取能量: {energy:.8f} a.u. ({os.path.basename(log_file)})")
            return energy
        except ValueError:
            pass
    
    # 备用匹配模式（应对不同格式）
    fallback_match = re.search(
        r"SCF Done:\s+=\s+([-+]?\d+\.\d+)", 
        content
    )
    
    if fallback_match:
        return float(fallback_match.group(1))
    
    print(f"错误: 未找到SCF能量 in {os.path.basename(log_file)}")
    return np.nan

def skip_existing_calculations(r_co, r_cs):
    """
    跳过已完成的计算（日志文件存在且能量可提取），避免重复计算
    """
    log_filename = f"ocs_scan_CO{r_co:.2f}_CS{r_cs:.2f}.log"
    log_file = os.path.abspath(os.path.join(output_dir, log_filename))
    if os.path.exists(log_file):
        # 快速检查是否有正常终止标记
        with open(log_file, 'r', encoding='gbk', errors='ignore') as f:
            if "Normal termination" in f.read():
                print(f"跳过已完成计算: {log_file}")
                return True
    return False

# ======================
# 主程序（添加进度提示和错误处理）
# ======================
if __name__ == "__main__":
    print("="*50)
    print("OCS 2D 势能面扫描程序")
    print("="*50)
    
    # (1) 生成输入文件并提交计算（跳过已完成的）
    print("\n【步骤1：生成输入文件并提交计算】")
    total_calc = len(r_co_range) * len(r_cs_range)
    completed_calc = 0
    
    for i, r_co in enumerate(r_co_range):
        for j, r_cs in enumerate(r_cs_range):
            # 跳过已完成的计算
            if skip_existing_calculations(r_co, r_cs):
                completed_calc += 1
                continue
            
            # 生成输入文件
            input_file = generate_gaussian_input(r_co, r_cs)
            # 运行计算
            log_file = run_gaussian(input_file)
            if log_file:
                completed_calc += 1
            
            # 打印进度
            print(f"进度: {completed_calc}/{total_calc} 计算完成")
            print("-"*30)
    
    # (2) 提取所有能量数据
    print("\n【步骤2：提取能量数据】")
    energies = np.full((len(r_co_range), len(r_cs_range)), np.nan)  # 初始化为nan
    
    for i, r_co in enumerate(r_co_range):
        for j, r_cs in enumerate(r_cs_range):
            log_filename = f"ocs_scan_CO{r_co:.2f}_CS{r_cs:.2f}.log"
            log_file = os.path.abspath(os.path.join(output_dir, log_filename))
            energy = extract_energy(log_file)
            energies[i, j] = energy
    
    # 检查数据完整性
    valid_energy_count = np.count_nonzero(~np.isnan(energies))
    print(f"\n数据提取完成：有效能量 {valid_energy_count}/{total_calc} 个")
    
    # (3) 保存数据到文本文件（包含键长信息，方便后续分析）
    print("\n【步骤3：保存数据】")
    # 保存能量矩阵+键长范围
    header = f"R(CO)范围: {r_co_range[0]:.2f} - {r_co_range[-1]:.2f} 埃 (步长{np.diff(r_co_range)[0]:.2f} 埃)\n"
    header += f"R(CS)范围: {r_cs_range[0]:.2f} - {r_cs_range[-1]:.2f} 埃 (步长{np.diff(r_cs_range)[0]:.2f} 埃)\n"
    header += "能量单位：a.u.（Hartree）\n"
    header += "列：R(CS) 行：R(CO)"
    
    np.savetxt(
        "ocs_2d_pes_energies.txt",
        energies,
        header=header,
        fmt="%.8f",
        encoding='utf-8'
    )
    print("能量数据已保存至：ocs_2d_pes_energies.txt")
    
    # (4) 绘制3D势能面和等高线图（处理nan值，避免绘图错误）
    print("\n【步骤4：绘制势能面】")
    # 转换单位：Hartree -> kcal/mol（1 Hartree = 627.509 kcal/mol）
    energies_kcal = energies * 627.509
    # 减去最低能量，使势能面以最小值为0（更直观）
    min_energy = np.nanmin(energies_kcal)
    energies_kcal -= min_energy
    
    # 创建网格
    r_co_grid, r_cs_grid = np.meshgrid(r_co_range, r_cs_range, indexing='ij')
    
    # 绘图设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 支持负号
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D曲面图（屏蔽nan值，避免警告）
    ax1 = fig.add_subplot(121, projection='3d')
    mask = ~np.isnan(energies_kcal)
    surf = ax1.plot_surface(
        r_co_grid[mask], r_cs_grid[mask], energies_kcal[mask],
        cmap='viridis', alpha=0.9, linewidth=0.1
    )
    ax1.set_xlabel("R(CO) (Å)", fontsize=10)
    ax1.set_ylabel("R(CS) (Å)", fontsize=10)
    ax1.set_zlabel("相对能量 (kcal/mol)", fontsize=10)
    ax1.set_title("OCS 2D 势能面（B3LYP/6-31G(d)）", fontsize=12, pad=20)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='相对能量 (kcal/mol)')
    
    # 等高线图（处理nan值）
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(
        r_co_grid, r_cs_grid, energies_kcal,
        levels=25, cmap='viridis', alpha=0.9,
        extend='max'  # 处理超出范围的nan值
    )
    ax2.set_xlabel("R(CO) (Å)", fontsize=10)
    ax2.set_ylabel("R(CS) (Å)", fontsize=10)
    ax2.set_title("OCS 势能面等高线图", fontsize=12, pad=20)
    plt.colorbar(contour, ax=ax2, label='相对能量 (kcal/mol)')
    
    # 保存图片（高分辨率）
    plt.tight_layout()
    plt.savefig("ocs_2d_pes.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("程序完成！")
    print(f"结果文件：")
    print(f"  - 输入文件和日志：{os.path.abspath(output_dir)}")
    print(f"  - 能量数据：{os.path.abspath('ocs_2d_pes_energies.txt')}")
    print(f"  - 势能面图：{os.path.abspath('ocs_2d_pes.png')}")
    print("="*50)