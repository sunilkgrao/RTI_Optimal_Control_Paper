#!/usr/bin/env python3
"""
Edge-of-Transparency Tracker Validator
Validates critical surface tracking for RTI control in laser-plasma interactions
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EdgeTransparency')

class EdgeTransparencyValidator:
    def __init__(self, output_dir='../analysis'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Physical constants
        self.c = 3e8  # Speed of light (m/s)
        self.e = 1.602e-19  # Electron charge (C)
        self.m_e = 9.109e-31  # Electron mass (kg)
        self.epsilon_0 = 8.854e-12  # Permittivity of free space
        
        # Transparency threshold (1/e transmission)
        self.transparency_threshold = 1/np.e
        
        # Test parameters
        self.intensities = np.logspace(14, 18, 5)  # W/cm^2
        self.wavelengths = [0.35e-6, 0.53e-6, 1.06e-6]  # meters (UV, green, IR)
        
        self.results = {}
        
    def calculate_critical_density(self, wavelength):
        """Calculate critical plasma density for given wavelength"""
        
        omega = 2 * np.pi * self.c / wavelength
        n_critical = self.m_e * self.epsilon_0 * omega**2 / self.e**2
        
        return n_critical
    
    def calculate_ponderomotive_force(self, intensity, wavelength):
        """Calculate ponderomotive force for given laser parameters"""
        
        # Convert intensity to SI units (W/m^2)
        I_SI = intensity * 1e4
        
        # Laser frequency
        omega = 2 * np.pi * self.c / wavelength
        
        # Quiver velocity
        E_0 = np.sqrt(2 * I_SI / (self.c * self.epsilon_0))
        v_osc = self.e * E_0 / (self.m_e * omega)
        
        # Ponderomotive potential
        U_p = 0.5 * self.m_e * v_osc**2
        
        # Ponderomotive force (gradient of potential)
        # Assuming Gaussian profile with characteristic length L
        L_char = wavelength * 10  # Typical focal spot size
        F_pond = U_p / L_char
        
        return F_pond, U_p
    
    def simulate_critical_surface_evolution(self, intensity, wavelength, time_duration):
        """Simulate evolution of critical density surface"""
        
        # Critical density
        n_crit = self.calculate_critical_density(wavelength)
        
        # Ponderomotive force
        F_pond, U_p = self.calculate_ponderomotive_force(intensity, wavelength)
        
        # Time array
        dt = 1e-15  # femtosecond resolution
        times = np.arange(0, time_duration, dt)
        
        # Initial density profile (exponential scale length)
        L_scale = 10e-6  # 10 micron scale length
        x = np.linspace(-100e-6, 100e-6, 1000)
        
        # Track critical surface position
        critical_positions = []
        transmission_values = []
        
        for t in times[::100]:  # Sample every 100 timesteps
            # Density profile evolution (simplified model)
            # Includes ponderomotive steepening and hole boring
            
            # Hole boring velocity
            v_hb = np.sqrt(2 * U_p / (self.m_e * n_crit))
            
            # Critical surface recession
            x_crit_0 = L_scale * np.log(1e23 / n_crit)  # Initial position
            x_crit = x_crit_0 - v_hb * t
            
            # Modified density profile
            density = 1e23 * np.exp(-(x - x_crit) / L_scale)
            density[x < x_crit] = n_crit * np.exp((x[x < x_crit] - x_crit) / (L_scale * 0.1))
            
            # Calculate transmission (simplified)
            optical_depth = np.cumsum(density * np.sqrt(1 - density/n_crit + 0j)) * (x[1] - x[0])
            transmission = np.exp(-np.real(optical_depth))
            
            # Find edge-of-transparency position
            trans_norm = transmission / transmission[0] if transmission[0] > 0 else transmission
            edge_indices = np.where(trans_norm < self.transparency_threshold)[0]
            
            if len(edge_indices) > 0:
                edge_position = x[edge_indices[0]]
                critical_positions.append(edge_position)
                transmission_values.append(trans_norm[edge_indices[0]])
            else:
                critical_positions.append(x_crit)
                transmission_values.append(0)
        
        return {
            'times': times[::100],
            'positions': np.array(critical_positions),
            'transmissions': np.array(transmission_values),
            'recession_velocity': v_hb,
            'critical_density': n_crit,
            'ponderomotive_potential': U_p
        }
    
    def analyze_tracking_stability(self, tracking_data):
        """Analyze stability of critical surface tracking"""
        
        times = tracking_data['times']
        positions = tracking_data['positions']
        
        # Calculate velocity
        if len(positions) > 1:
            velocities = np.gradient(positions, times)
        else:
            velocities = np.array([0])
        
        # Find oscillations
        peaks, properties = find_peaks(np.abs(velocities - np.mean(velocities)),
                                      prominence=0.1*np.std(velocities))
        
        # Calculate stability metrics
        mean_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # Oscillation frequency
        if len(peaks) > 1 and len(times) > 0:
            oscillation_period = np.mean(np.diff(times[peaks]))
            oscillation_freq = 1 / oscillation_period if oscillation_period > 0 else 0
        else:
            oscillation_freq = 0
        
        # Tracking quality metric
        stability_ratio = velocity_std / abs(mean_velocity) if abs(mean_velocity) > 0 else np.inf
        
        return {
            'mean_recession_velocity': mean_velocity,
            'velocity_std': velocity_std,
            'oscillation_frequency': oscillation_freq,
            'n_oscillations': len(peaks),
            'stability_ratio': stability_ratio,
            'stable': stability_ratio < 0.2,  # Less than 20% variation
            'tracking_quality': 'good' if stability_ratio < 0.1 else 
                              'moderate' if stability_ratio < 0.3 else 'poor'
        }
    
    def validate_rti_suppression(self, tracking_data, atwood):
        """Validate RTI suppression at tracked critical surface"""
        
        # Growth rate at critical surface
        g_eff = tracking_data['recession_velocity']**2 / (2 * tracking_data['positions'][-1])
        
        # Perturbation wavelength
        lambda_pert = 20e-6  # 20 micron typical
        k = 2 * np.pi / lambda_pert
        
        # Classical RTI growth rate
        gamma_classical = np.sqrt(atwood * abs(g_eff) * k)
        
        # Suppression due to tracking (oscillatory acceleration)
        osc_freq = self.analyze_tracking_stability(tracking_data)['oscillation_frequency']
        
        if osc_freq > 0:
            # Stabilization parameter
            omega_osc = 2 * np.pi * osc_freq
            stabilization = omega_osc**2 / (4 * gamma_classical**2)
            
            # Effective growth rate (reduced by oscillations)
            gamma_eff = gamma_classical * np.exp(-stabilization)
            suppression_factor = gamma_classical / gamma_eff if gamma_eff > 0 else np.inf
        else:
            gamma_eff = gamma_classical
            suppression_factor = 1.0
        
        return {
            'classical_growth_rate': gamma_classical,
            'effective_growth_rate': gamma_eff,
            'suppression_factor': suppression_factor,
            'rti_suppressed': suppression_factor > 2,  # At least 2x suppression
            'effective_acceleration': g_eff
        }
    
    def plot_validation_results(self, all_results):
        """Create comprehensive validation plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Critical surface evolution for different intensities
        ax1 = axes[0, 0]
        
        for intensity in self.intensities[:3]:  # Plot first 3 intensities
            if f'I_{intensity:.2e}_λ_{self.wavelengths[0]:.2e}' in all_results:
                data = all_results[f'I_{intensity:.2e}_λ_{self.wavelengths[0]:.2e}']
                tracking = data['tracking']
                
                ax1.plot(tracking['times'] * 1e12,  # Convert to ps
                        tracking['positions'] * 1e6,  # Convert to μm
                        label=f'I={intensity:.1e} W/cm²')
        
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Critical Surface Position (μm)')
        ax1.set_title('Critical Surface Evolution')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transmission profiles
        ax2 = axes[0, 1]
        
        for wavelength in self.wavelengths:
            intensities_plot = []
            transmissions_plot = []
            
            for key, data in all_results.items():
                if f'λ_{wavelength:.2e}' in key:
                    intensities_plot.append(data['intensity'])
                    transmissions_plot.append(np.mean(data['tracking']['transmissions']))
            
            if intensities_plot:
                ax2.semilogx(intensities_plot, transmissions_plot,
                           'o-', label=f'λ={wavelength*1e6:.2f} μm')
        
        ax2.axhline(y=self.transparency_threshold, color='r', linestyle='--',
                   label='1/e threshold')
        ax2.set_xlabel('Intensity (W/cm²)')
        ax2.set_ylabel('Mean Transmission')
        ax2.set_title('Edge-of-Transparency Tracking')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recession velocity vs intensity
        ax3 = axes[0, 2]
        
        for wavelength in self.wavelengths:
            intensities_plot = []
            velocities_plot = []
            
            for key, data in all_results.items():
                if f'λ_{wavelength:.2e}' in key:
                    intensities_plot.append(data['intensity'])
                    velocities_plot.append(data['tracking']['recession_velocity'])
            
            if intensities_plot:
                ax3.loglog(intensities_plot, velocities_plot,
                          'o-', label=f'λ={wavelength*1e6:.2f} μm')
        
        # Theoretical scaling: v_hb ~ sqrt(I)
        I_theory = np.logspace(14, 18, 50)
        v_theory = 1e5 * np.sqrt(I_theory / 1e16)  # Normalized
        ax3.loglog(I_theory, v_theory, 'k--', alpha=0.5, label='√I scaling')
        
        ax3.set_xlabel('Intensity (W/cm²)')
        ax3.set_ylabel('Recession Velocity (m/s)')
        ax3.set_title('Hole Boring Velocity Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Stability analysis
        ax4 = axes[1, 0]
        
        wavelength_labels = []
        stability_ratios = []
        
        for key, data in all_results.items():
            if 'stability' in data:
                wavelength_labels.append(key.split('_')[2])
                stability_ratios.append(data['stability']['stability_ratio'])
        
        if stability_ratios:
            bars = ax4.bar(range(len(stability_ratios)), stability_ratios, alpha=0.7)
            
            # Color code by quality
            for i, ratio in enumerate(stability_ratios):
                if ratio < 0.1:
                    bars[i].set_color('green')
                elif ratio < 0.3:
                    bars[i].set_color('yellow')
                else:
                    bars[i].set_color('red')
            
            ax4.axhline(y=0.2, color='r', linestyle='--', label='Stability threshold')
            ax4.set_ylabel('Velocity Variation Ratio')
            ax4.set_title('Tracking Stability Analysis')
            ax4.legend()
        
        # Plot 5: RTI suppression factors
        ax5 = axes[1, 1]
        
        atwood_values = [0.3, 0.5, 0.7]  # Test Atwood numbers
        suppression_data = {A: [] for A in atwood_values}
        
        for A in atwood_values:
            for key, data in all_results.items():
                if 'rti_suppression' in data:
                    if abs(data['rti_suppression']['atwood'] - A) < 0.01:
                        suppression_data[A].append(
                            data['rti_suppression']['suppression_factor']
                        )
        
        positions = np.arange(len(atwood_values))
        width = 0.35
        
        for i, A in enumerate(atwood_values):
            if suppression_data[A]:
                ax5.bar(positions[i], np.mean(suppression_data[A]),
                       width, label=f'A={A}', alpha=0.7)
        
        ax5.axhline(y=2, color='g', linestyle='--', label='2× suppression')
        ax5.set_ylabel('RTI Suppression Factor')
        ax5.set_xlabel('Atwood Number')
        ax5.set_title('RTI Growth Suppression')
        ax5.set_xticks(positions)
        ax5.set_xticklabels([f'{A}' for A in atwood_values])
        ax5.legend()
        
        # Plot 6: Summary metrics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate summary statistics
        n_stable = sum(1 for d in all_results.values() 
                      if 'stability' in d and d['stability']['stable'])
        n_suppressed = sum(1 for d in all_results.values()
                          if 'rti_suppression' in d and d['rti_suppression']['rti_suppressed'])
        n_total = len(all_results)
        
        summary_text = f"""Edge-of-Transparency Validation Summary
        
Total simulations: {n_total}
Stable tracking: {n_stable}/{n_total} ({n_stable/n_total*100:.1f}%)
RTI suppressed: {n_suppressed}/{n_total} ({n_suppressed/n_total*100:.1f}%)

Wavelengths tested: {len(self.wavelengths)}
Intensities tested: {len(self.intensities)}

Validation: {'PASSED' if n_stable/n_total > 0.7 else 'FAILED'}
"""
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10,
                verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'edge_transparency_validation.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {output_file}")
        
        return output_file
    
    def run_validation(self):
        """Execute complete edge-of-transparency validation"""
        
        logger.info("Starting Edge-of-Transparency validation...")
        print("\n=== Edge-of-Transparency Tracking Validation ===")
        print(f"Testing {len(self.intensities)} intensities")
        print(f"Testing {len(self.wavelengths)} wavelengths")
        
        all_results = {}
        
        for wavelength in self.wavelengths:
            print(f"\nWavelength: {wavelength*1e6:.2f} μm")
            
            for intensity in self.intensities:
                print(f"  Intensity: {intensity:.2e} W/cm²")
                
                # Simulate critical surface evolution
                tracking_data = self.simulate_critical_surface_evolution(
                    intensity, wavelength, time_duration=10e-12  # 10 ps
                )
                
                # Analyze stability
                stability = self.analyze_tracking_stability(tracking_data)
                
                # Test RTI suppression
                atwood = 0.5  # Representative Atwood number
                rti_suppression = self.validate_rti_suppression(tracking_data, atwood)
                rti_suppression['atwood'] = atwood
                
                # Store results
                key = f'I_{intensity:.2e}_λ_{wavelength:.2e}'
                all_results[key] = {
                    'intensity': intensity,
                    'wavelength': wavelength,
                    'tracking': tracking_data,
                    'stability': stability,
                    'rti_suppression': rti_suppression
                }
                
                print(f"    ✓ Tracking: {stability['tracking_quality']}")
                print(f"    ✓ RTI suppression: {rti_suppression['suppression_factor']:.2f}×")
        
        # Create visualization
        plot_file = self.plot_validation_results(all_results)
        
        # Calculate overall validation metrics
        n_stable = sum(1 for d in all_results.values() if d['stability']['stable'])
        n_suppressed = sum(1 for d in all_results.values() 
                          if d['rti_suppression']['rti_suppressed'])
        n_total = len(all_results)
        
        # Save validation summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'theorem': 'Edge-of-Transparency Tracking',
            'intensities_W/cm2': self.intensities.tolist(),
            'wavelengths_m': self.wavelengths,
            'n_simulations': n_total,
            'n_stable_tracking': n_stable,
            'n_rti_suppressed': n_suppressed,
            'validation_passed': n_stable / n_total > 0.7,
            'plot_file': plot_file,
            'detailed_results': {
                key: {
                    'stability': data['stability']['tracking_quality'],
                    'suppression_factor': data['rti_suppression']['suppression_factor'],
                    'recession_velocity_m/s': data['tracking']['recession_velocity']
                }
                for key, data in all_results.items()
            }
        }
        
        output_file = os.path.join(self.output_dir, 'edge_transparency_validation.json')
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Validation complete!")
        print(f"Results saved to: {output_file}")
        print(f"Visualization saved to: {plot_file}")
        
        return summary

def main():
    """Main execution"""
    validator = EdgeTransparencyValidator()
    results = validator.run_validation()
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Theorem validated: {results['validation_passed']}")
    print(f"Stable tracking: {results['n_stable_tracking']}/{results['n_simulations']}")
    print(f"RTI suppressed: {results['n_rti_suppressed']}/{results['n_simulations']}")
    
    if results['validation_passed']:
        print("✓ Edge-of-transparency tracking validated for RTI control!")
    else:
        print("✗ Edge-of-transparency validation incomplete - review results")

if __name__ == "__main__":
    main()