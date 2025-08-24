#!/usr/bin/env python3
"""
Generate Realistic RTI Test Data Based on Physical Scaling Laws
Creates synthetic but physics-based data that follows actual RT instability physics
"""

import numpy as np
import json
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class RealisticRTIGenerator:
    def __init__(self, output_dir='../test_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_linear_growth_data(self, atwood, viscosity, k_mode):
        """Generate early-time linear growth data using dispersion relation"""
        
        # Dispersion relation: γ² + 2νk²γ - (Ag k - σk³/ρ) = 0
        # For RT without surface tension: γ² + 2νk²γ - Agk = 0
        
        A = atwood
        g = 9.81  # m/s²
        nu = viscosity  # kinematic viscosity
        k = k_mode
        
        # Solve quadratic: γ² + 2νk²γ - Agk = 0
        # γ = (-2νk² ± √(4ν²k⁴ + 4Agk))/2
        discriminant = 4*nu**2*k**4 + 4*A*g*k
        if discriminant < 0:
            return None  # Stable mode
            
        gamma_plus = (-2*nu*k**2 + np.sqrt(discriminant))/2
        gamma_minus = (-2*nu*k**2 - np.sqrt(discriminant))/2
        
        # Physical growth rate (positive root)
        gamma = gamma_plus if gamma_plus > 0 else 0
        
        return gamma
    
    def generate_nonlinear_evolution(self, atwood, viscosity, times):
        """Generate full nonlinear evolution including viscous transition"""
        
        # Wavelength of most unstable mode
        lambda_optimal = 2*np.pi/np.sqrt(atwood*9.81/(3*viscosity**2))
        k_optimal = 2*np.pi/lambda_optimal
        
        # Linear growth rate
        gamma = self.generate_linear_growth_data(atwood, viscosity, k_optimal)
        if gamma is None or gamma <= 0:
            return np.ones_like(times) * 1e-6  # Tiny constant for stable case
            
        # Initial amplitude
        a0 = 1e-5  # 10 micron initial perturbation
        
        mixing_widths = []
        for t in times:
            if t <= 0:
                width = 2*a0
            else:
                # Viscous time scale
                t_visc = 1.0/(viscosity * k_optimal**2)
                
                if t < 0.1 * t_visc:
                    # Pure linear growth
                    width = 2*a0 * np.exp(gamma * t)
                elif t < 2.0 * t_visc:
                    # Transition regime - reduced growth due to viscous effects
                    linear_growth = 2*a0 * np.exp(gamma * 0.1 * t_visc)
                    t_trans = t - 0.1*t_visc
                    # Exponential decay of growth rate
                    effective_gamma = gamma * np.exp(-t_trans/t_visc)
                    width = linear_growth * np.exp(effective_gamma * t_trans)
                else:
                    # Viscous regime - cubic scaling emerges
                    # h(t) = C * (Ag)^(1/2) * (νt³)^(1/3)
                    C_empirical = 1.8 + 0.7*atwood  # Empirical scaling constant
                    viscous_scale = (viscosity * t**3)**(1/3)
                    width = C_empirical * np.sqrt(atwood * 9.81) * viscous_scale
                    
            mixing_widths.append(width)
            
        # Add realistic noise (measurement uncertainty)
        noise_level = 0.05  # 5% measurement uncertainty
        np.random.seed(int(atwood*1000 + viscosity*1e6))
        noise = np.random.normal(1.0, noise_level, len(mixing_widths))
        mixing_widths = np.array(mixing_widths) * noise
        
        return np.array(mixing_widths)
    
    def generate_experimental_dataset(self):
        """Generate comprehensive experimental dataset"""
        
        # Experimental parameters (realistic ranges)
        atwood_numbers = [0.1, 0.25, 0.5, 0.75]  # Realistic experimental range
        viscosities = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]  # m²/s
        
        # Time array (typical experiment duration)
        times = np.linspace(0, 0.5, 100)  # 0.5 second experiment
        
        experimental_data = {}
        
        for A in atwood_numbers:
            for nu in viscosities:
                # Generate mixing width evolution
                mixing_widths = self.generate_nonlinear_evolution(A, nu, times)
                
                # Extract growth rate from linear regime
                linear_mask = (times > 0.001) & (times < 0.05)  # Linear regime
                if np.sum(linear_mask) > 10:
                    t_lin = times[linear_mask]
                    w_lin = mixing_widths[linear_mask]
                    
                    # Log-linear fit
                    valid_mask = w_lin > 0
                    if np.sum(valid_mask) > 5:
                        coeffs = np.polyfit(t_lin[valid_mask], 
                                          np.log(w_lin[valid_mask]), 1)
                        extracted_gamma = coeffs[0]
                    else:
                        extracted_gamma = 0
                else:
                    extracted_gamma = 0
                
                # Theoretical growth rate for comparison
                k_optimal = np.sqrt(A*9.81/(3*nu**2))
                theoretical_gamma = self.generate_linear_growth_data(A, nu, k_optimal)
                
                key = f"A{A:.2f}_nu{nu:.1e}"
                experimental_data[key] = {
                    'times': times.tolist(),
                    'mixing_widths': mixing_widths.tolist(),
                    'atwood': A,
                    'viscosity': nu,
                    'extracted_growth_rate': extracted_gamma,
                    'theoretical_growth_rate': theoretical_gamma,
                    'wavelength': 2*np.pi/k_optimal if theoretical_gamma else None
                }
        
        return experimental_data
    
    def save_experimental_data(self):
        """Save realistic experimental data"""
        
        data = self.generate_experimental_dataset()
        
        # Save as JSON
        output_file = os.path.join(self.output_dir, 'realistic_rti_experimental_data.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Generated realistic experimental data: {output_file}")
        print(f"Total test cases: {len(data)}")
        
        # Create validation summary
        growth_rate_errors = []
        for key, test_case in data.items():
            extracted = test_case['extracted_growth_rate']
            theoretical = test_case['theoretical_growth_rate']
            
            if theoretical and theoretical > 0:
                error = abs(extracted - theoretical) / theoretical
                growth_rate_errors.append(error)
        
        if growth_rate_errors:
            mean_error = np.mean(growth_rate_errors)
            print(f"Mean growth rate extraction error: {mean_error:.1%}")
        
        return output_file
    
    def create_validation_plots(self, data_file):
        """Create validation plots"""
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Growth rate comparison
        ax1 = axes[0, 0]
        theoretical_rates = []
        extracted_rates = []
        
        for key, case in data.items():
            if case['theoretical_growth_rate'] and case['theoretical_growth_rate'] > 0:
                theoretical_rates.append(case['theoretical_growth_rate'])
                extracted_rates.append(case['extracted_growth_rate'])
        
        ax1.scatter(theoretical_rates, extracted_rates, alpha=0.7)
        ax1.plot([0, max(theoretical_rates)], [0, max(theoretical_rates)], 'r--', 
                label='Perfect Agreement')
        ax1.set_xlabel('Theoretical Growth Rate (1/s)')
        ax1.set_ylabel('Extracted Growth Rate (1/s)')
        ax1.set_title('Growth Rate Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample time evolution
        ax2 = axes[0, 1]
        sample_cases = list(data.keys())[:5]
        for key in sample_cases:
            case = data[key]
            times = np.array(case['times'])
            widths = np.array(case['mixing_widths'])
            ax2.semilogy(times, widths, label=f"A={case['atwood']}, ν={case['viscosity']:.0e}")
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mixing Width (m)')
        ax2.set_title('Sample Mixing Width Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Viscous scaling test
        ax3 = axes[1, 0]
        for key in sample_cases:
            case = data[key]
            times = np.array(case['times'])
            widths = np.array(case['mixing_widths'])
            nu = case['viscosity']
            
            # Late time cubic scaling
            late_mask = times > 0.2
            if np.sum(late_mask) > 10:
                t_late = times[late_mask]
                w_late = widths[late_mask]
                viscous_scale = (nu * t_late**3)**(1/3)
                xi = w_late / viscous_scale
                
                ax3.plot(t_late, xi, label=f"A={case['atwood']}")
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('ξ = h/(νt³)^(1/3)')
        ax3.set_title('Viscous Scaling Check')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Atwood number dependence
        ax4 = axes[1, 1]
        atwood_vals = []
        final_xi_vals = []
        
        for key, case in data.items():
            if case['viscosity'] == 1e-5:  # Fixed viscosity
                times = np.array(case['times'])
                widths = np.array(case['mixing_widths'])
                nu = case['viscosity']
                
                # Final time xi value
                t_final = times[-1]
                w_final = widths[-1]
                viscous_scale = (nu * t_final**3)**(1/3)
                xi_final = w_final / viscous_scale
                
                atwood_vals.append(case['atwood'])
                final_xi_vals.append(xi_final)
        
        if atwood_vals:
            ax4.scatter(atwood_vals, final_xi_vals)
            ax4.set_xlabel('Atwood Number')
            ax4.set_ylabel('Final ξ Value')
            ax4.set_title('Atwood Dependence (ν=1e-5)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = data_file.replace('.json', '_validation.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Created validation plots: {plot_file}")
        
        return plot_file

if __name__ == "__main__":
    generator = RealisticRTIGenerator()
    data_file = generator.save_experimental_data()
    plot_file = generator.create_validation_plots(data_file)