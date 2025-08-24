#!/usr/bin/env python3
"""
Enhanced Physical Parameter Sweep - 200k meaningful parameters
Staying within RTI-relevant regime
"""

import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime
import os

# Physical constants
C = 2.99792458e8  # m/s
M_E = 9.10938356e-31  # kg
Q_E = 1.602176634e-19  # C
EPSILON_0 = 8.854187817e-12  # F/m

def compute_rti_physical(params):
    """Compute RTI with proper physical scaling"""
    a0, k_ratio, density_nc, thickness_nm, polarization, angle_deg = params
    
    # Laser parameters
    wavelength = 1.0e-6
    omega = 2 * np.pi * C / wavelength
    k_laser = 2 * np.pi / wavelength
    
    # Angle effect
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    
    # Critical density
    n_crit = EPSILON_0 * M_E * omega**2 / Q_E**2
    density = density_nc * n_crit
    omega_p = np.sqrt(density * Q_E**2 / (EPSILON_0 * M_E))
    
    # Skip if overdense for this a0
    relativistic_factor = np.sqrt(1 + a0**2/2)
    n_crit_rel = n_crit * relativistic_factor
    if density > 10 * n_crit_rel:  # Way overdense
        return {
            'a0': a0, 'k_ratio': k_ratio, 'density_nc': density_nc,
            'thickness_nm': thickness_nm, 'polarization': polarization,
            'angle_deg': angle_deg, 'gamma_norm': 0.0, 'regime': 'overdense'
        }
    
    # Ponderomotive acceleration (angle-dependent)
    pol_factor = 1.0 if polarization == 'circular' else 0.5
    a = pol_factor * (omega * C * a0**2) / (4 * (1 + a0**2/2)) * cos_theta
    
    # Thickness effects
    d = thickness_nm * 1e-9
    T = d**2 * a / 12
    
    # Physical viscosity scaling (reduced from buggy version)
    C_RR = 1e-4  # Realistic RR coefficient
    C_IF = 1e-5  # Realistic IFE coefficient
    
    nu_RR = C_RR * (C**2/omega) * (a0/10)**4
    B_norm = min(0.1 * a0, 5.0)
    nu_IF = C_IF * (C**2/omega) * B_norm**2
    nu_eff = nu_RR + nu_IF
    
    # Key scales
    k_T = np.sqrt(a/T) if T > 0 else 1e10
    k_nu3 = (a/nu_eff**2)**(1/3) if nu_eff > 0 else 1e10
    Phi_3 = k_T / k_nu3
    
    # Growth rate
    k = k_ratio * k_laser
    kd = k * d
    
    # Skip very thick targets
    if kd > 3:  # Finite thickness kills RTI
        return {
            'a0': a0, 'k_ratio': k_ratio, 'density_nc': density_nc,
            'thickness_nm': thickness_nm, 'polarization': polarization,
            'angle_deg': angle_deg, 'gamma_norm': 0.0, 'regime': 'thick'
        }
    
    g_finite = np.tanh(kd) / kd if kd > 0 else 1.0
    a_eff = a * g_finite
    
    # Solve dispersion relation
    B = 2 * nu_eff * k**2
    C_disp = -(a_eff * k - T * k**3)
    
    if C_disp >= 0:
        gamma = 0.0
        regime = 'stable'
    else:
        discriminant = B**2 - 4*C_disp
        gamma = (-B + np.sqrt(discriminant)) / 2
        
        # Classify regime
        if gamma/omega_p > 0.01:  # Significant growth
            regime = 'unstable'
        else:
            regime = 'marginal'
    
    gamma_norm = gamma / omega_p
    
    return {
        'a0': a0,
        'k_ratio': k_ratio,
        'density_nc': density_nc,
        'thickness_nm': thickness_nm,
        'polarization': polarization,
        'angle_deg': angle_deg,
        'gamma_norm': gamma_norm,
        'Phi_3': Phi_3,
        'regime': regime,
        'a': a,
        'nu_eff': nu_eff
    }

def run_enhanced_sweep():
    """Run enhanced physical parameter sweep"""
    print("="*70)
    print("ENHANCED PHYSICAL RTI PARAMETER SWEEP")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Enhanced but still physical ranges
    a0_range = np.logspace(np.log10(8), np.log10(60), 35)  # Focus on RTI-active range
    k_ratio_range = np.logspace(np.log10(0.3), np.log10(3.0), 30)  # Near k_laser
    
    # Smart density selection based on a0
    density_list = []
    for a0 in [10, 20, 30, 40, 50]:
        # Add densities around relativistic critical density
        n_rel = a0  # Approximate
        density_list.extend([
            n_rel * 0.3,   # Underdense
            n_rel * 0.5,   # Moderately underdense
            n_rel * 1.0,   # Critical
            n_rel * 2.0,   # Moderately overdense
            n_rel * 5.0,   # Overdense
        ])
    density_range = sorted(set([d for d in density_list if 30 <= d <= 500]))[:20]
    
    thickness_range = np.array([5, 8, 10, 15, 20, 30, 40, 50, 70, 100])  # nm
    angle_range = [0, 20, 40]  # degrees
    polarizations = ['circular', 'linear']
    
    print(f"\nParameter ranges (physical regime):")
    print(f"  a₀: {len(a0_range)} values [{a0_range[0]:.1f} - {a0_range[-1]:.1f}]")
    print(f"  k/k_laser: {len(k_ratio_range)} values [{k_ratio_range[0]:.2f} - {k_ratio_range[-1]:.2f}]")
    print(f"  Density: {len(density_range)} values [{min(density_range):.0f} - {max(density_range):.0f} nc]")
    print(f"  Thickness: {len(thickness_range)} values [{thickness_range[0]} - {thickness_range[-1]} nm]")
    print(f"  Angles: {angle_range} degrees")
    print(f"  Polarizations: {polarizations}")
    
    # Generate parameter combinations
    all_params = []
    for a0 in a0_range:
        for k_ratio in k_ratio_range:
            for density in density_range:
                for thickness in thickness_range:
                    for pol in polarizations:
                        for angle in angle_range:
                            all_params.append((a0, k_ratio, density, thickness, pol, angle))
    
    total_params = len(all_params)
    print(f"\nTotal parameters: {total_params:,}")
    
    # Process with parallel execution
    n_workers = mp.cpu_count()
    results = []
    start_time = time.time()
    
    print(f"\nProcessing with {n_workers} workers...")
    
    # Process in chunks
    chunk_size = 1000
    n_chunks = (total_params + chunk_size - 1) // chunk_size
    
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, total_params)
        chunk = all_params[chunk_start:chunk_end]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(compute_rti_physical, chunk))
            results.extend(chunk_results)
        
        # Progress
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            elapsed = time.time() - start_time
            rate = len(results) / elapsed
            eta = (total_params - len(results)) / rate
            print(f"Progress: {len(results):,}/{total_params:,} ({len(results)/total_params*100:.1f}%) | "
                  f"Rate: {rate:.0f}/s | ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"Average rate: {total_params/total_time:.0f} parameters/second")
    
    # Growth statistics
    growing = df[df['gamma_norm'] > 0]
    print(f"\nGrowth statistics:")
    print(f"  Total unstable: {len(growing):,} ({len(growing)/len(df)*100:.1f}%)")
    print(f"  Max γ/ωₚ: {df['gamma_norm'].max():.4f}")
    print(f"  Mean γ/ωₚ (unstable): {growing['gamma_norm'].mean():.4f}")
    
    # Regime breakdown
    print(f"\nRegime distribution:")
    for regime, count in df['regime'].value_counts().items():
        print(f"  {regime}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Find optimal parameters
    if len(growing) > 0:
        optimal = df.loc[df['gamma_norm'].idxmax()]
        print(f"\nOptimal parameters:")
        print(f"  a₀ = {optimal['a0']:.1f}")
        print(f"  k/k_laser = {optimal['k_ratio']:.2f}")
        print(f"  Density = {optimal['density_nc']:.0f} nc")
        print(f"  Thickness = {optimal['thickness_nm']:.0f} nm")
        print(f"  Angle = {optimal['angle_deg']:.0f}°")
        print(f"  Polarization = {optimal['polarization']}")
        print(f"  γ/ωₚ = {optimal['gamma_norm']:.4f}")
    
    # Save results
    df.to_csv('enhanced_physical_sweep_results.csv', index=False)
    print(f"\n✓ Results saved to enhanced_physical_sweep_results.csv")
    
    # Save summary
    with open('enhanced_sweep_summary.txt', 'w') as f:
        f.write(f"Enhanced Physical RTI Parameter Sweep Summary\n")
        f.write(f"="*50 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Computation time: {total_time/60:.1f} minutes\n")
        f.write(f"Rate: {total_params/total_time:.0f} parameters/second\n")
        f.write(f"Unstable cases: {len(growing):,} ({len(growing)/len(df)*100:.1f}%)\n")
        f.write(f"Max growth rate: γ/ωₚ = {df['gamma_norm'].max():.4f}\n")
    
    return df

if __name__ == "__main__":
    df_results = run_enhanced_sweep()
    print("\n✓ Enhanced physical parameter sweep complete!")
