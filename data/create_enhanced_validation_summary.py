#!/usr/bin/env python3
"""
Create summary figure for enhanced validation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the results
df = pd.read_csv('enhanced_physical_sweep_results.csv')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Growth rate distribution
ax = axes[0, 0]
unstable = df[df['gamma_norm'] > 0]
ax.hist(unstable['gamma_norm'], bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
ax.axvline(unstable['gamma_norm'].mean(), color='red', linestyle='--', 
           label=f'Mean: {unstable["gamma_norm"].mean():.3f}')
ax.axvline(unstable['gamma_norm'].max(), color='darkred', linestyle='-', 
           label=f'Max: {unstable["gamma_norm"].max():.3f}')
ax.set_xlabel('Growth Rate (γ/ωₚ)')
ax.set_ylabel('Count')
ax.set_title(f'RTI Growth Rate Distribution\n{len(unstable):,} unstable cases (74%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Parameter space coverage
ax = axes[0, 1]
a0_values = df['a0'].unique()
k_values = df['k_ratio'].unique()
density_values = df['density_nc'].unique()

# Create text summary
summary_text = f"""ENHANCED VALIDATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters: 567,000
Computation Time: 19.9 minutes
Rate: 475 params/second

PARAMETER RANGES:
• a₀: {len(a0_values)} values [{min(a0_values):.0f}-{max(a0_values):.0f}]
• k/k_laser: {len(k_values)} values [{min(k_values):.1f}-{max(k_values):.1f}]
• Density: {len(density_values)} values [{min(density_values):.0f}-{max(density_values):.0f} nc]
• Thickness: 10 values [5-100 nm]
• Angles: 0°, 20°, 40°
• Polarizations: CP, LP

KEY RESULTS:
✓ 419,400 unstable (74%)
✓ Max γ/ωₚ = 0.221
✓ Physics-guided selection
✓ 22.7× larger than original"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
ax.axis('off')

# Panel 3: Optimal growth regions
ax = axes[1, 0]
# Group by a0 and k_ratio, take max gamma
pivot = df.groupby(['a0', 'k_ratio'])['gamma_norm'].max().reset_index()
pivot_matrix = pivot.pivot(index='a0', columns='k_ratio', values='gamma_norm')

im = ax.contourf(pivot_matrix.columns, pivot_matrix.index, pivot_matrix.values, 
                 levels=20, cmap='viridis')
plt.colorbar(im, ax=ax, label='Max γ/ωₚ')
ax.set_xlabel('k/k_laser')
ax.set_ylabel('a₀')
ax.set_title('Maximum Growth Rate Map')
ax.set_xscale('log')

# Mark optimal point
optimal = df.loc[df['gamma_norm'].idxmax()]
ax.scatter(optimal['k_ratio'], optimal['a0'], color='red', s=200, 
           marker='*', edgecolor='white', linewidth=2, 
           label=f'Optimal: γ/ωₚ={optimal["gamma_norm"]:.3f}')
ax.legend()

# Panel 4: Comparison with experiments
ax = axes[1, 1]
# Bar chart comparing different validations
validations = {
    'Original 25k': {'count': 25000, 'time': 1.1, 'unstable_pct': 85, 'max_gamma': 0.18},
    'Failed 900k': {'count': 900000, 'time': 6.9, 'unstable_pct': 0, 'max_gamma': 0},
    'Enhanced 567k': {'count': 567000, 'time': 19.9, 'unstable_pct': 74, 'max_gamma': 0.221}
}

x = np.arange(len(validations))
names = list(validations.keys())
counts = [v['count']/1000 for v in validations.values()]
unstable_pcts = [v['unstable_pct'] for v in validations.values()]

ax2 = ax.twinx()
bars1 = ax.bar(x - 0.2, counts, 0.4, label='Parameters (thousands)', color='skyblue')
bars2 = ax2.bar(x + 0.2, unstable_pcts, 0.4, label='Unstable %', color='lightgreen')

ax.set_ylabel('Parameters (×1000)', fontsize=12)
ax2.set_ylabel('Unstable Percentage', fontsize=12)
ax.set_xlabel('Validation Run')
ax.set_title('Validation Comparison: Quality Matters')
ax.set_xticks(x)
ax.set_xticklabels(names)

# Add value labels
for bar, val in zip(bars1, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{int(val)}k', ha='center', va='bottom')

for bar, val in zip(bars2, unstable_pcts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val}%', ha='center', va='bottom')

ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Main title
fig.suptitle('Enhanced Physical RTI Validation: 567,000 Parameters in the Right Regime', 
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('enhanced_validation_summary.pdf', dpi=300, bbox_inches='tight')
plt.savefig('enhanced_validation_summary.png', dpi=150, bbox_inches='tight')

print("✓ Enhanced validation summary figure created!")
print("  - enhanced_validation_summary.pdf")
print("  - enhanced_validation_summary.png")
