#!/usr/bin/env python3
"""
Run Section III.F Analysis on Real Simulation Data
Processes LP/CP PIC outputs through growth rate extractor
"""

import sys
from pathlib import Path
import json
from section_iiif_extractor import SectionIIIFExtractor

def analyze_simulation_outputs():
    """Process both LP and CP simulation outputs"""
    
    extractor = SectionIIIFExtractor()
    
    # Simulation parameters (from synthetic PIC simulation)
    sim_params = {
        'lambda0_um': 0.8,
        'a0': 15.0,
        'thickness_nm': 50.0,
        'k_seed': 2.1e6,
        'ne_over_nc': 100.0,
        'Te_eV': 10.0,
        'Ti_eV': 1.0,
        'target_material': 'DLC',
        'facility': 'Synthetic_PIC_Validation'
    }
    
    results = {}
    
    # Process LP simulation
    print("=== ANALYZING LP SIMULATION ===")
    lp_probe_file = "lp_simulation/Probes0000.h5"
    if Path(lp_probe_file).exists():
        try:
            # Load HDF5 data
            import h5py
            with h5py.File(lp_probe_file, 'r') as f:
                times = f['probes/0/times'][:]
                interface_data = f['probes/0/interface_position'][:]
            
            # Process through Section III.F workflow
            perturbation_data = extractor.extract_interface_perturbation({
                'times': times,
                'interface_evolution': interface_data
            })
            
            lp_results = extractor.extract_spectral_growth_rates(perturbation_data, 
                                                               k_modes=[sim_params['k_seed']])
            results['LP'] = lp_results
            print(f"✓ LP Growth Rate: {lp_results['linear_fit']['slope']:.3e} ± {lp_results['linear_fit']['slope_err']:.3e} s⁻¹")
            print(f"✓ LP R²: {lp_results['linear_fit']['r_squared']:.4f}")
            print(f"✓ LP Confidence: {lp_results['confidence']:.3f}")
        except Exception as e:
            print(f"✗ LP Analysis failed: {e}")
            results['LP'] = None
    
    # Process CP simulation
    print("\n=== ANALYZING CP SIMULATION ===")
    cp_probe_file = "cp_simulation/Probes0000.h5"
    if Path(cp_probe_file).exists():
        try:
            # Load HDF5 data
            import h5py
            with h5py.File(cp_probe_file, 'r') as f:
                times = f['probes/0/times'][:]
                interface_data = f['probes/0/interface_position'][:]
            
            # Process through Section III.F workflow
            perturbation_data = extractor.extract_interface_perturbation({
                'times': times,
                'interface_evolution': interface_data
            })
            
            cp_results = extractor.extract_spectral_growth_rates(perturbation_data, 
                                                               k_modes=[sim_params['k_seed']])
            results['CP'] = cp_results
            print(f"✓ CP Growth Rate: {cp_results['linear_fit']['slope']:.3e} ± {cp_results['linear_fit']['slope_err']:.3e} s⁻¹")
            print(f"✓ CP R²: {cp_results['linear_fit']['r_squared']:.4f}")
            print(f"✓ CP Confidence: {cp_results['confidence']:.3f}")
        except Exception as e:
            print(f"✗ CP Analysis failed: {e}")
            results['CP'] = None
    
    # Save combined results
    output_file = Path("section_iiif_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'simulation_parameters': sim_params,
            'lp_analysis': results.get('LP'),
            'cp_analysis': results.get('CP'),
            'analysis_metadata': {
                'extractor': 'SectionIIIFExtractor',
                'methodology': 'RANSAC_fitting_with_linear_window_detection',
                'analysis_date': '2025-08-24T12:08:00Z'
            }
        }, f, indent=2, default=_json_converter)
    
    print(f"\n✓ Results saved to {output_file}")
    return results

def _json_converter(obj):
    """Convert numpy types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

if __name__ == "__main__":
    results = analyze_simulation_outputs()
    
    if results.get('LP') and results.get('CP'):
        lp_gamma = results['LP']['linear_fit']['slope']
        cp_gamma = results['CP']['linear_fit']['slope']
        ratio = cp_gamma / lp_gamma
        print(f"\n=== GROWTH RATE COMPARISON ===")
        print(f"LP γ = {lp_gamma:.3e} s⁻¹")
        print(f"CP γ = {cp_gamma:.3e} s⁻¹")  
        print(f"r_γ = γ_CP/γ_LP = {ratio:.4f}")
        print("✓ Section III.F analysis complete!")
    else:
        print("✗ Analysis incomplete - check simulation outputs")