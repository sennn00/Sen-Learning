import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import pickle

# ======================
# 用户可调参数
# ======================
ex_range = np.linspace(0.0, 1.0, 5)      # 交换部分local比例扫描：0.0到1.0，步长0.25
corr_range = np.linspace(0.0, 1.0, 5)    # 关联部分local比例扫描：0.0到1.0，步长0.25
gaussian_path = r"D:\App\G16W\g16.exe"   # Gaussian 16 可执行文件绝对路径
output_dir = "hcn_hybrid_scan_results"   # 结果保存目录
scratch_dir = r"D:\GaussianTemp"         # Gaussian临时目录（独立设置，避免权限问题）
# ======================

# 初始化环境变量（提前设置，确保所有函数可用）
os.environ["GAUSS_EXEDIR"] = r"D:\App\G16W"
os.environ["GAUSS_SCRDIR"] = scratch_dir
os.makedirs(scratch_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def generate_inp(ex, corr):
    """
    生成 Gaussian 输入文件 inp.com（HCN分子，固定几何结构）
    参数:
        ex (float): 交换部分 local 比例，范围 [0, 1]
        corr (float): 关联部分 local 比例，范围 [0, 1]
    """
    if not (0.0 <= ex <= 1.0):
        raise ValueError("ex must be between 0 and 1")
    if not (0.0 <= corr <= 1.0):
        raise ValueError("corr must be between 0 and 1")

    # 计算交换部分（限制在0~10000）
    ex_local = int(round(ex * 10000))
    ex_nonlocal = int(round((1.0 - ex) * 10000))
    ex_local = max(0, min(10000, ex_local))
    ex_nonlocal = max(0, min(10000, ex_nonlocal))

    # 计算关联部分
    corr_local = int(round(corr * 10000))
    corr_nonlocal = int(round((1.0 - corr) * 10000))
    corr_local = max(0, min(10000, corr_local))
    corr_nonlocal = max(0, min(10000, corr_nonlocal))

    # 格式化为5位字符串
    iop_77 = f"{ex_nonlocal:05d}{ex_local:05d}"   # 如 "0900001000"
    iop_78 = f"{corr_nonlocal:05d}{corr_local:05d}"

    # HCN分子输入文件（固定键长：CH=1.06 Å, CN=1.16 Å）
    content = f"""%Mem=6GB
%NProcShared=4
#P B3LYP/6-31G(d) SP IOp(3/74=609,3/77={iop_77},3/78={iop_78})

HCN hybrid functional scan: ex={ex:.2f}, corr={corr:.2f}

0 1
H     -1.060000     0.000000     0.000000
C      0.000000     0.000000     0.000000
N      1.160000     0.000000     0.000000

"""
    filename = os.path.join(output_dir, f"hcn_ex{ex:.2f}_corr{corr:.2f}.com")
    filename_abs = os.path.abspath(filename)
    
    with open(filename_abs, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print(f"生成输入文件: {filename_abs}")
    return filename_abs

def run_gaussian(input_file):
    """
    运行Gaussian计算（Windows下稳定调用方式）
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
    """从Gaussian输出文件中提取能量"""
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

    # 精准匹配SCF能量行
    energy_match = re.search(
        r"SCF Done:\s+E\(.+\)\s+=\s+([-+]?\d+\.\d+)", 
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

def skip_existing(ex, corr):
    """检查是否已存在计算结果"""
    log_filename = f"hcn_ex{ex:.2f}_corr{corr:.2f}.log"
    log_file = os.path.abspath(os.path.join(output_dir, log_filename))
    if os.path.exists(log_file):
        # 快速检查是否有正常终止标记
        with open(log_file, 'r', encoding='gbk', errors='ignore') as f:
            if "Normal termination" in f.read():
                print(f"跳过已完成计算: {log_file}")
                return True
    return False

def plot_pes(energies):
    """绘制双参数势能面"""
    # 转换为kcal/mol并相对最小值归一化
    energies_kcal = energies * 627.509
    energies_kcal -= np.nanmin(energies_kcal)
    
    # 创建网格
    ex_grid, corr_grid = np.meshgrid(ex_range, corr_range, indexing='ij')
    
    # 绘图设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 支持负号
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D曲面图（屏蔽nan值）
    ax1 = fig.add_subplot(121, projection='3d')
    mask = ~np.isnan(energies_kcal)
    surf = ax1.plot_surface(
        ex_grid[mask], corr_grid[mask], energies_kcal[mask],
        cmap='viridis', alpha=0.9, linewidth=0.1
    )
    ax1.set_xlabel("交换局部比例", fontsize=10)
    ax1.set_ylabel("关联局部比例", fontsize=10)
    ax1.set_zlabel("相对能量 (kcal/mol)", fontsize=10)
    ax1.set_title("HCN 双参数势能面", fontsize=12, pad=20)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='相对能量 (kcal/mol)')
    
    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(
        ex_grid, corr_grid, energies_kcal,
        levels=25, cmap='viridis', alpha=0.9,
        extend='max'  # 处理超出范围的nan值
    )
    ax2.set_xlabel("交换局部比例", fontsize=10)
    ax2.set_ylabel("关联局部比例", fontsize=10)
    ax2.set_title("HCN 参数空间等高线", fontsize=12, pad=20)
    plt.colorbar(contour, ax=ax2, label='相对能量 (kcal/mol)')
    
    # 保存图片（高分辨率）
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hcn_hybrid_pes.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("="*50)
    print("HCN 双参数（交换-关联）扫描程序")
    print("="*50)
    
    # (1) 生成输入文件并提交计算（跳过已完成的）
    print("\n【步骤1：生成输入文件并提交计算】")
    total_calc = len(ex_range) * len(corr_range)
    completed_calc = 0
    
    energies = np.full((len(ex_range), len(corr_range)), np.nan)  # 初始化为nan
    
    for i, ex in enumerate(ex_range):
        for j, corr in enumerate(corr_range):
            # 跳过已完成的计算
            if skip_existing(ex, corr):
                completed_calc += 1
                continue
            
            # 生成输入文件
            input_file = generate_inp(ex, corr)
            # 运行计算
            log_file = run_gaussian(input_file)
            if log_file:
                completed_calc += 1
            
            # 打印进度
            print(f"进度: {completed_calc}/{total_calc} 计算完成")
            print("-"*30)
    
    # (2) 提取所有能量数据
    print("\n【步骤2：提取能量数据】")
    for i, ex in enumerate(ex_range):
        for j, corr in enumerate(corr_range):
            log_filename = f"hcn_ex{ex:.2f}_corr{corr:.2f}.log"
            log_file = os.path.abspath(os.path.join(output_dir, log_filename))
            energy = extract_energy(log_file)
            energies[i, j] = energy
    
    # 检查数据完整性
    valid_energy_count = np.count_nonzero(~np.isnan(energies))
    print(f"\n数据提取完成：有效能量 {valid_energy_count}/{total_calc} 个")
    
    # (3) 保存数据到文本文件
    print("\n【步骤3：保存数据】")
    header = f"交换比例范围: {ex_range[0]:.2f} - {ex_range[-1]:.2f} (步长{np.diff(ex_range)[0]:.2f})\n"
    header += f"关联比例范围: {corr_range[0]:.2f} - {corr_range[-1]:.2f} (步长{np.diff(corr_range)[0]:.2f})\n"
    header += "能量单位：a.u.（Hartree）\n"
    header += "列：关联比例 行：交换比例"
    
    np.savetxt(
        os.path.join(output_dir, "hcn_hybrid_energies.txt"),
        energies,
        header=header,
        fmt="%.8f",
        encoding='utf-8'
    )
    print(f"能量数据已保存至：{os.path.join(output_dir, 'hcn_hybrid_energies.txt')}")
    
    # 同时保存二进制格式（便于后续读取）
    with open(os.path.join(output_dir, "hcn_hybrid_energies.pkl"), 'wb') as f:
        pickle.dump({
            'ex_range': ex_range,
            'corr_range': corr_range,
            'energies': energies
        }, f)
    
    # (4) 绘制3D势能面和等高线图
    print("\n【步骤4：绘制势能面】")
    plot_pes(energies)
    
    print("\n" + "="*50)
    print("程序完成！")
    print(f"结果文件：")
    print(f"  - 输入文件和日志：{os.path.abspath(output_dir)}")
    print(f"  - 能量数据：{os.path.join(os.path.abspath(output_dir), 'hcn_hybrid_energies.txt')}")
    print(f"  - 势能面图：{os.path.join(os.path.abspath(output_dir), 'hcn_hybrid_pes.png')}")
    print("="*50)
