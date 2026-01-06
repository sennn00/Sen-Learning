import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


energy_data = """
-81.89229990 -82.15968013 -82.42721589 -82.69490675 -82.96275232 -83.23075217
-83.72073172 -83.98963742 -84.25869711 -84.52791033 -84.79727661 -85.06679547
-85.56832091 -85.83876435 -86.10935971 -86.38010647 -86.65100411 -86.92205211
-87.43540450 -87.70739439 -87.97953357 -88.25182149 -88.52425757 -88.79684127
-89.32232130 -89.59586227 -89.86954938 -90.14338204 -90.41735964 -90.69148158
-91.22940535 -91.50449748 -91.77973209 -92.05510854 -92.33062621 -92.60628450
"""

# Convert the energy data to a numpy array
energies = np.array([float(x) for x in energy_data.split()]).reshape(6, 6)

# Exchange and correlation ratios (0.0-1.0 in 0.2 steps)
ex_ratios = np.linspace(0.0, 1.0, 6)
corr_ratios = np.linspace(0.0, 1.0, 6)

# Find the lowest energy point
min_energy = np.min(energies)
min_indices = np.unravel_index(np.argmin(energies), energies.shape)
min_ex = ex_ratios[min_indices[0]]
min_corr = corr_ratios[min_indices[1]]

# Convert to kcal/mol and relative to minimum
energies_kcal = (energies - min_energy) * 627.509

# Create meshgrid for plotting
ex_grid, corr_grid = np.meshgrid(ex_ratios, corr_ratios, indexing='ij')

# Set up the figure
plt.figure(figsize=(14, 6))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 3D Surface Plot
ax1 = plt.subplot(121, projection='3d')
surf = ax1.plot_surface(ex_grid, corr_grid, energies_kcal, 
                       cmap='viridis', alpha=0.9)

# Mark the lowest energy point
ax1.scatter([min_ex], [min_corr], [0], 
            color='red', s=100, label=f'Minimum: {min_energy:.6f} a.u.')

ax1.set_xlabel('Exchange Ratio', fontsize=10)
ax1.set_ylabel('Correlation Ratio', fontsize=10)
ax1.set_zlabel('Relative Energy (kcal/mol)', fontsize=10)
ax1.set_title('Potential Energy Surface (3D View)')
ax1.legend()

# Contour Plot
ax2 = plt.subplot(122)
contour = ax2.contourf(ex_grid, corr_grid, energies_kcal, 
                      levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='Relative Energy (kcal/mol)')

# Mark the lowest energy point
ax2.scatter(min_ex, min_corr, color='red', s=100, 
           label=f'Minimum at ({min_ex:.1f}, {min_corr:.1f})')

ax2.set_xlabel('Exchange Ratio', fontsize=10)
ax2.set_ylabel('Correlation Ratio', fontsize=10)
ax2.set_title('Potential Energy Surface (Contour View)')
ax2.legend()

plt.tight_layout()
plt.savefig('potential_energy_surface.png', dpi=300)
plt.show()

print(f"Minimum energy found at:")
print(f"Exchange ratio = {min_ex:.2f}")
print(f"Correlation ratio = {min_corr:.2f}")
print(f"Energy = {min_energy:.6f} Hartrees")
print(f"       = {(min_energy * 627.509):.2f} kcal/mol")
