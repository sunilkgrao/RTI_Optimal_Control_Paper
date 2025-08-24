#!/usr/bin/env python3
"""
Generate synthetic experimental data based on Zhou 2017 typical values
for RTI validation studies.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

def generate_growth_rate_data():
    """Generate growth rate vs Atwood number data following γ = √(A*g*k)"""
    
    # Zhou 2017 typical parameters
    g = 9.81  # gravity acceleration (m/s^2)
    k_values = [1.0, 2.0, 5.0, 10.0]  # wave numbers (1/m)
    
    # Atwood numbers from 0.1 to 0.9
    A_values = np.linspace(0.1, 0.9, 9)
    
    data = []
    
    for k in k_values:
        for A in A_values:
            # Theoretical growth rate
            gamma_theory = np.sqrt(A * g * k)
            
            # Add 5-10% experimental scatter
            scatter = np.random.uniform(0.05, 0.10) * np.random.choice([-1, 1])
            gamma_exp = gamma_theory * (1 + scatter)
            
            # Experimental uncertainty (typically 3-5%)
            uncertainty = gamma_exp * np.random.uniform(0.03, 0.05)
            
            data.append({
                'atwood_number': A,
                'wave_number': k,
                'growth_rate_theory': gamma_theory,
                'growth_rate_experimental': gamma_exp,
                'uncertainty': uncertainty,
                'source': 'Zhou2017_synthetic'
            })
    
    return pd.DataFrame(data)

def generate_mixing_width_data():
    """Generate mixing width evolution data"""
    
    # Time evolution parameters
    t_max = 2.0  # seconds
    t_values = np.linspace(0.1, t_max, 20)
    
    A_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    g = 9.81
    k = 2.0  # representative wave number
    
    data = []
    
    for A in A_values:
        gamma = np.sqrt(A * g * k)
        
        for t in t_values:
            # Mixing width follows h = 2 * γ * t^2 in early stages
            h_theory = 2 * gamma * t**2
            
            # Add experimental scatter
            scatter = np.random.uniform(0.05, 0.15) * np.random.choice([-1, 1])
            h_exp = h_theory * (1 + scatter)
            
            uncertainty = h_exp * np.random.uniform(0.04, 0.08)
            
            data.append({
                'time': t,
                'atwood_number': A,
                'mixing_width_theory': h_theory,
                'mixing_width_experimental': h_exp,
                'uncertainty': uncertainty,
                'source': 'Zhou2017_synthetic'
            })
    
    return pd.DataFrame(data)

def generate_bubble_spike_data():
    """Generate bubble and spike velocity data"""
    
    A_values = np.linspace(0.1, 0.9, 9)
    g = 9.81
    k = 2.0
    
    data = []
    
    for A in A_values:
        # Zhou 2017 empirical relations
        v_bubble_theory = 0.25 * np.sqrt(A * g / k)  # Bubble velocity
        v_spike_theory = -0.30 * np.sqrt(A * g / k)  # Spike velocity (negative)
        
        # Add experimental scatter
        v_bubble_exp = v_bubble_theory * (1 + np.random.uniform(-0.08, 0.08))
        v_spike_exp = v_spike_theory * (1 + np.random.uniform(-0.08, 0.08))
        
        data.append({
            'atwood_number': A,
            'bubble_velocity_theory': v_bubble_theory,
            'bubble_velocity_experimental': v_bubble_exp,
            'spike_velocity_theory': v_spike_theory,
            'spike_velocity_experimental': v_spike_exp,
            'bubble_uncertainty': abs(v_bubble_exp * 0.05),
            'spike_uncertainty': abs(v_spike_exp * 0.05),
            'source': 'Zhou2017_synthetic'
        })
    
    return pd.DataFrame(data)

def main():
    """Generate all synthetic datasets"""
    
    print("Generating synthetic experimental data based on Zhou 2017...")
    
    # Create output directory
    output_dir = Path("/Users/sunilrao/Downloads/RTI_Optimal_Control_Paper/rti_validation/data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate datasets
    growth_data = generate_growth_rate_data()
    mixing_data = generate_mixing_width_data()
    bubble_spike_data = generate_bubble_spike_data()
    
    # Save to CSV files
    growth_data.to_csv(output_dir / "growth_rate_data.csv", index=False)
    mixing_data.to_csv(output_dir / "mixing_width_data.csv", index=False)
    bubble_spike_data.to_csv(output_dir / "bubble_spike_data.csv", index=False)
    
    # Save metadata
    metadata = {
        "description": "Synthetic experimental data generated based on Zhou 2017 typical values",
        "datasets": {
            "growth_rate_data.csv": {
                "description": "Growth rate vs Atwood number following γ = √(A*g*k)",
                "rows": len(growth_data),
                "scatter": "5-10% experimental scatter added",
                "uncertainty": "3-5% experimental uncertainty"
            },
            "mixing_width_data.csv": {
                "description": "Mixing width evolution h = 2*γ*t²",
                "rows": len(mixing_data),
                "scatter": "5-15% experimental scatter added",
                "uncertainty": "4-8% experimental uncertainty"
            },
            "bubble_spike_data.csv": {
                "description": "Bubble and spike velocities from Zhou 2017 relations",
                "rows": len(bubble_spike_data),
                "scatter": "±8% experimental scatter added",
                "uncertainty": "5% experimental uncertainty"
            }
        },
        "parameters": {
            "gravity": 9.81,
            "wave_numbers": [1.0, 2.0, 5.0, 10.0],
            "atwood_range": [0.1, 0.9],
            "time_range": [0.1, 2.0]
        },
        "reference": "Zhou, Y. (2017). Rayleigh-Taylor and Richtmyer-Meshkov instability induced flow, turbulence, and mixing. Physics Reports, 720, 1-136."
    }
    
    with open(output_dir / "synthetic_data_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(growth_data)} growth rate data points")
    print(f"Generated {len(mixing_data)} mixing width data points")
    print(f"Generated {len(bubble_spike_data)} bubble/spike velocity data points")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    main()