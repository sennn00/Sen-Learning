import numpy as np 
from matplotlib import pyplot as plt 

# ==================== 1. 系统初始化 ==================== 
ngrid = 500 
L = 2.0 
a = L/ngrid 
xx = np.arange(ngrid)*a + a/2 

# 动能矩阵 K 
K = np.zeros((ngrid, ngrid)) 
for i in range(ngrid): 
    K[i, i] = 1./a**2 
for i in range(ngrid-1): 
    K[i, i+1] = -1./(a**2*2) 
    K[i+1, i] = -1./(a**2*2) 

# 求解本征态 
e1, v1 = np.linalg.eigh(K) 
wf = v1[:, np.argsort(e1)] 
energies = np.sort(e1) 

# ==================== 2. 吸收谱计算 ==================== 
def calculate_absorption_spectrum(omega_range): 
    absorption = np.zeros_like(omega_range) 
    psi_0 = wf[:, 0].astype(complex) 
    dt, t_total = 0.05, 100.0 
    n_steps = int(t_total / dt) 

    for idx, omega in enumerate(omega_range): 
        print(f"计算频率 {idx+1}/{len(omega_range)}: ω = {omega:.3f}") 

        A, Vx = 0.1, 0.1 * (xx - 1.0) 
        psi = psi_0.copy() 
        dipole_moment = np.zeros(n_steps, dtype=complex) 
        I = np.eye(ngrid) 

        for step in range(n_steps): 
            t = step * dt 
            V_t = Vx * np.sin(omega * t) 
            H = K + np.diag(V_t) 
            CN_min = I + 1j * dt / 2 * H 
            CN_plu = I - 1j * dt / 2 * H 

            try: 
                psi = np.linalg.solve(CN_min, CN_plu @ psi) 
            except np.linalg.LinAlgError: 
                absorption[idx] = 0 
                break 

            # 稳健的偶极矩计算 
            dipole_moment[step] = np.sum(np.conj(psi) * Vx * psi) 

        if step == n_steps - 1: 
            start_idx = n_steps // 2 
            if len(dipole_moment[start_idx:]) > 10: 
                spectrum = np.abs(np.fft.fft(dipole_moment[start_idx:]))**2 
                freqs = np.fft.fftfreq(len(dipole_moment[start_idx:]), dt) 
                omega_idx = np.argmin(np.abs(freqs - omega)) 
                absorption[idx] = spectrum[omega_idx] 

    return absorption 

# ==================== 3. 修正的能级分析 ==================== 
def analyze_levels(energies, wf, omega_range, absorption): 
    """修正后的能级分析 - 避免数组形状错误""" 
    # 电场算符 
    Vx = 0.1 * (xx - 1.0) 
    n_levels = min(10, len(energies)) 

    # 方法1：直接计算矩阵元（最稳健） 
    transition_strength = np.zeros((n_levels, n_levels)) 
    for i in range(n_levels): 
        for j in range(n_levels): 
            if i != j: 
                # ⟨i|V|j⟩ = ∫ ψ_i* V ψ_j dx ≈ sum(ψ_i* V ψ_j * a) 
                # 由于是离散点，直接求和即可 
                matrix_element = np.sum(np.conj(wf[:, i]) * Vx * wf[:, j]) 
                transition_strength[i, j] = np.abs(matrix_element) 

    # 计算能级差 
    level_diffs = [] 
    diff_pairs = [] 
    for i in range(n_levels): 
        for j in range(i+1, n_levels): 
            level_diffs.append(energies[j] - energies[i]) 
            diff_pairs.append((i, j)) 

    return transition_strength, np.array(level_diffs), diff_pairs 

# ==================== 4. 主程序 ==================== 
# 计算吸收谱 
print("开始计算吸收谱...") 
omega_range = np.arange(0.5, 5.0, 0.2) 
absorption = calculate_absorption_spectrum(omega_range) 
print("计算完成！\n") 

# 能级分析 
transition_strength, level_diffs, diff_pairs = analyze_levels(energies, wf, omega_range, absorption) 

# ==================== 5. 增强可视化 ==================== 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10)) 

# (1) 吸收谱 
ax1.plot(omega_range, absorption, 'b-o', linewidth=2, markersize=4) 
ax1.set_xlabel('Driving Frequency ω') 
ax1.set_ylabel('Absorption Intensity') 
ax1.set_title('Absorption Spectrum') 
ax1.grid(True, alpha=0.3) 

peaks = np.where(absorption > 0.1 * np.max(absorption))[0] 
for peak in peaks: 
    ax1.axvline(omega_range[peak], color='r', linestyle='--', alpha=0.5) 
ax1.text(0.05, 0.95, f'{len(peaks)} peaks found', transform=ax1.transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) 
# (2) 能级结构 
n_display = min(10, len(energies)) 
for i in range(n_display): 
    ax2.hlines(1.0, energies[i], energies[i]+0.01, color='k', linewidth=3) 
    ax2.text(energies[i], 1.03, f'E{i}', fontsize=9, ha='center', rotation=45) 

ax2.set_xlabel('Energy') 
ax2.set_ylabel('Level') 
ax2.set_title('Energy Levels (E0 to E9)') 
ax2.set_ylim(0.5, 1.5) 
ax2.set_xlim(energies[0]-0.2, energies[n_display-1]+0.2) 
ax2.grid(True, alpha=0.3) 

# (3) 能级差 vs 共振频率 
ax3.plot(level_diffs, np.ones_like(level_diffs), 'g^', markersize=10, label='ΔE Levels') 
ax3.plot(omega_range[peaks], np.ones_like(peaks), 'ro', markersize=12, label='Resonance ω') 
ax3.set_xlabel('Energy / Frequency') 
ax3.set_ylabel('Reference Line') 
ax3.set_title('Level Differences vs Resonance') 
ax3.legend() 
ax3.grid(True, alpha=0.3) 

# 标记匹配 
for i, omega in enumerate(omega_range[peaks]): 
    matches = np.where(np.abs(level_diffs - omega) < 0.2)[0] 
    if len(matches) > 0: 
        for m in matches: 
            d, (ei, ej) = level_diffs[m], diff_pairs[m] 
            ax3.text(d, 1.1, f'ΔE{ei}{ej}', fontsize=8, ha='center', color='blue') 

# (4) 跃迁矩阵元热图 
im = ax4.imshow(transition_strength[:n_display, :n_display], cmap='hot', interpolation='nearest') 
ax4.set_xlabel('Final State j') 
ax4.set_ylabel('Initial State i') 
ax4.set_title('Transition Strength |⟨i|V|j⟩|') 
ax4.set_xticks(range(n_display)) 
ax4.set_yticks(range(n_display)) 
plt.colorbar(im, ax=ax4, label='Strength') 

plt.tight_layout() 
plt.show() 
# ==================== 6. 文字分析 ==================== 
print("\n" + "="*50) 
print("共振分析报告") 
print("="*50) 

print(f"\n1. 共振频率: {omega_range[peaks]}") 
print(f"2. 前10个能级: {energies[:10]}") 

print("\n3. 可能的跃迁:") 
found_transitions = [] 
for i in range(min(5, len(energies)-1)): 
    for j in range(i+1, min(8, len(energies))): 
        diff = energies[j] - energies[i] 
        for peak in peaks: 
            if abs(diff - omega_range[peak]) < 0.2: 
                strength = transition_strength[i, j] if i < 10 and j < 10 else 0 
                found_transitions.append((i, j, diff, omega_range[peak], strength)) 
                print(f"  E{i} → E{j}: ΔE = {diff:.3f}, ω = {omega_range[peak]:.3f}, 强度 = {strength:.4f}") 

print("\n4. 选择定则 (强跃迁 > 0.01):") 
n_display = min(10, len(energies)) 
for i in range(n_display): 
    for j in range(n_display): 
        if transition_strength[i, j] > 0.01 and i != j: 
            print(f"  E{i} ↔ E{j}: 强度 = {transition_strength[i, j]:.4f}") 

if len(found_transitions) == 0: 
    print("\n  未发现明显共振匹配，可能原因:") 
    print("  - 频率分辨率不足") 
    print("  - 电场强度A=0.1太弱") 
    print("  - 能级差超出扫描范围")