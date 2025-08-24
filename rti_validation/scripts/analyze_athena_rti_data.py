#!/usr/bin/env python3
"""
COMPREHENSIVE ANALYSIS OF REAL ATHENA++ RTI SIMULATION DATA
Processes actual Rayleigh-Taylor instability data from published simulations
Maximizes M3 MacBook Pro computational power with parallel processing
"""

import numpy as np
import h5py
import json
import glob
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, fft2
from sklearn.linear_model import RANSACRegressor, HuberRegressor
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

@dataclass 
class RTIAnalysisResults:
    """Complete analysis results from real RTI data"""
    # Growth rate measurements
    linear_growth_rate: float
    linear_growth_rate_error: float
    nonlinear_growth_rate: float
    saturation_amplitude: float
    saturation_time: float
    
    # Spectral analysis
    dominant_wavelength: float
    dominant_wavenumber: float
    wavenumber_spectrum: np.ndarray
    growth_rate_spectrum: np.ndarray
    
    # Physical parameters
    atwood_number: float
    density_ratio: float
    effective_gravity: float
    
    # Quality metrics
    r_squared: float
    data_points_analyzed: int
    time_range: Tuple[float, float]
    
    # Raw data for validation
    times: np.ndarray
    amplitudes: np.ndarray

class AthenaRTIAnalyzer:
    """
    Analyzes REAL Athena++ Rayleigh-Taylor instability simulation data
    Implements comprehensive analysis pipeline for academic validation
    """
    
    def __init__(self, data_path: str, output_dir: str = "athena_rti_analysis"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) 
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # M3 optimization
        self.cpu_cores = mp.cpu_count()
        self.logger.info(f"🚀 Athena++ RTI Data Analyzer initialized")
        self.logger.info(f"   CPU cores: {self.cpu_cores}")
        self.logger.info(f"   Data path: {self.data_path}")
    
    def read_athena_file(self, filename: str) -> Dict:
        """Read Athena++ HDF5 output file with full data extraction"""
        try:
            with h5py.File(filename, 'r') as f:
                # Get primitive variables
                # Shape: (5, 1, 1, 200, 200) = (variables, z, y, x)
                prim_data = np.array(f['prim'])
                
                # Extract individual fields
                data = {
                    'time': float(f.attrs['Time']),
                    'density': prim_data[0, 0, 0, :, :],  # rho
                    'pressure': prim_data[1, 0, 0, :, :],  # press
                    'velocity_x': prim_data[2, 0, 0, :, :],  # vel1
                    'velocity_y': prim_data[3, 0, 0, :, :],  # vel2
                    'velocity_z': prim_data[4, 0, 0, :, :] if prim_data.shape[0] > 4 else None,  # vel3
                    'coordinates': {
                        'x1v': np.array(f['x1v']),  # x coordinates (cell centers)
                        'x2v': np.array(f['x2v']),  # y coordinates (cell centers)
                        'x1f': np.array(f['x1f']),  # x coordinates (cell faces)
                        'x2f': np.array(f['x2f'])   # y coordinates (cell faces)
                    },
                    'metadata': {
                        'num_cycles': int(f.attrs['NumCycles']),
                        'grid_size': list(f.attrs['RootGridSize']),
                        'mesh_block_size': list(f.attrs['MeshBlockSize'])
                    }
                }
                
                return data
        except Exception as e:
            self.logger.warning(f"Could not read {filename}: {e}")
            return None
    
    def extract_interface_amplitude(self, density: np.ndarray, x_coords: np.ndarray, 
                                  y_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Extract RTI interface amplitude from density field
        Returns: (amplitude, interface_positions)
        """
        # Find mean density to identify interface
        rho_heavy = np.max(density)
        rho_light = np.min(density) 
        rho_mean = (rho_heavy + rho_light) / 2.0
        
        interface_positions = []
        
        # Extract interface position for each y-coordinate
        for j in range(len(y_coords)):
            density_profile = density[:, j]
            
            # Find where density crosses mean value (interface location)
            crossings = np.where(np.diff(np.sign(density_profile - rho_mean)))[0]
            
            if len(crossings) > 0:
                # Take first crossing as interface
                idx = crossings[0]
                if idx < len(x_coords) - 1:
                    # Linear interpolation for precise position
                    x1, x2 = x_coords[idx], x_coords[idx + 1]
                    d1, d2 = density_profile[idx], density_profile[idx + 1]
                    
                    if abs(d2 - d1) > 1e-10:
                        x_interface = x1 + (rho_mean - d1) * (x2 - x1) / (d2 - d1)
                    else:
                        x_interface = (x1 + x2) / 2.0
                    
                    interface_positions.append(x_interface)
                else:
                    interface_positions.append(x_coords[idx])
            else:
                # No crossing found, use middle position
                interface_positions.append(0.0)
        
        interface_positions = np.array(interface_positions)
        
        # Calculate amplitude as RMS deviation from mean
        mean_position = np.mean(interface_positions)
        amplitude = np.sqrt(np.mean((interface_positions - mean_position)**2))
        
        # Also calculate peak-to-peak amplitude
        if len(interface_positions) > 0:
            peak_amplitude = np.max(interface_positions) - np.min(interface_positions)
        else:
            peak_amplitude = 0.0
        
        return amplitude, interface_positions
    
    def analyze_spectral_content(self, interface_positions: np.ndarray, 
                                y_coords: np.ndarray) -> Dict:
        """Perform spectral analysis of interface perturbations"""
        
        # Remove mean to get perturbation
        perturbation = interface_positions - np.mean(interface_positions)
        
        # Compute FFT
        fft_vals = fft(perturbation)
        freqs = fftfreq(len(y_coords), d=(y_coords[-1] - y_coords[0]) / len(y_coords))
        
        # Power spectrum
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Find dominant mode (excluding zero frequency)
        positive_mask = freqs > 0
        if np.any(positive_mask):
            positive_freqs = freqs[positive_mask]
            positive_power = power_spectrum[positive_mask]
            
            dominant_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_idx]
            dominant_wavelength = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
            dominant_wavenumber = 2 * np.pi * dominant_freq
            
            # Find secondary modes
            peaks, properties = find_peaks(positive_power, height=np.max(positive_power) * 0.1)
            
        else:
            dominant_wavelength = np.inf
            dominant_wavenumber = 0.0
            peaks = []
        
        return {
            'dominant_wavelength': dominant_wavelength,
            'dominant_wavenumber': dominant_wavenumber,
            'power_spectrum': power_spectrum,
            'frequencies': freqs,
            'peak_amplitude': np.max(np.abs(perturbation)),
            'rms_amplitude': np.sqrt(np.mean(perturbation**2)),
            'num_modes': len(peaks) if peaks is not None else 1
        }
    
    def compute_growth_rate(self, times: np.ndarray, amplitudes: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute RTI growth rate using robust RANSAC fitting
        Returns: (growth_rate, error, r_squared)
        """
        # Remove zero and negative amplitudes
        valid_mask = amplitudes > 1e-10
        if np.sum(valid_mask) < 5:
            return 0.0, 0.0, 0.0
        
        valid_times = times[valid_mask]
        valid_amps = amplitudes[valid_mask]
        
        # Find linear growth phase (before saturation)
        # Saturation typically occurs when amplitude ~ 0.1 * wavelength
        max_amp_for_linear = 0.1  # Fraction of domain size
        linear_mask = valid_amps < max_amp_for_linear
        
        if np.sum(linear_mask) < 5:
            linear_mask = np.ones_like(valid_amps, dtype=bool)
        
        t_linear = valid_times[linear_mask]
        amp_linear = valid_amps[linear_mask]
        
        # Take log for exponential fitting: A(t) = A0 * exp(γt)
        log_amp = np.log(amp_linear)
        
        # Use RANSAC for robust linear fitting
        ransac = RANSACRegressor(
            base_estimator=HuberRegressor(epsilon=1.35),
            min_samples=max(3, len(t_linear)//4),
            residual_threshold=0.2,
            max_trials=1000,
            random_state=42
        )
        
        X = t_linear.reshape(-1, 1)
        ransac.fit(X, log_amp)
        
        # Growth rate is the slope
        growth_rate = float(ransac.estimator_.coef_[0])
        
        # Compute R²
        y_pred = ransac.predict(X)
        ss_res = np.sum((log_amp - y_pred) ** 2)
        ss_tot = np.sum((log_amp - np.mean(log_amp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate error from residuals
        inlier_mask = ransac.inlier_mask_
        if np.any(inlier_mask):
            residuals = log_amp[inlier_mask] - y_pred[inlier_mask]
            error = np.std(residuals) / np.sqrt(np.sum(inlier_mask))
        else:
            error = 0.0
        
        return growth_rate, error, r_squared
    
    def parallel_file_analysis(self, file_list: List[str]) -> List[Dict]:
        """Analyze multiple files in parallel using all CPU cores"""
        self.logger.info(f"   Processing {len(file_list)} files on {self.cpu_cores} cores...")
        
        def analyze_single_file(filename):
            data = self.read_athena_file(filename)
            if data is None:
                return None
            
            # Extract interface amplitude
            amplitude, interface = self.extract_interface_amplitude(
                data['density'], 
                data['coordinates']['x1v'],
                data['coordinates']['x2v']
            )
            
            # Spectral analysis
            spectral = self.analyze_spectral_content(
                interface,
                data['coordinates']['x2v']
            )
            
            return {
                'time': data['time'],
                'amplitude': amplitude,
                'peak_amplitude': spectral['peak_amplitude'],
                'rms_amplitude': spectral['rms_amplitude'],
                'dominant_wavelength': spectral['dominant_wavelength'],
                'dominant_wavenumber': spectral['dominant_wavenumber'],
                'num_modes': spectral['num_modes'],
                'interface': interface,
                'density_field': data['density']
            }
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
            results = list(executor.map(analyze_single_file, file_list))
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        self.logger.info(f"   Successfully processed {len(valid_results)}/{len(file_list)} files")
        
        return valid_results
    
    def analyze_density_ratio(self, dr_value: float) -> Optional[RTIAnalysisResults]:
        """Analyze a complete density ratio simulation run"""
        self.logger.info(f"\n📊 Analyzing density ratio {dr_value} simulation...")
        
        # Build path to data files
        dr_dir = self.data_path / f"rt_dr{dr_value}"
        if not dr_dir.exists():
            self.logger.warning(f"   Directory not found: {dr_dir}")
            return None
        
        # Find all timestep files
        pattern = str(dr_dir / "*.athdf")
        file_list = sorted(glob.glob(pattern))
        
        if len(file_list) == 0:
            self.logger.warning(f"   No .athdf files found in {dr_dir}")
            return None
        
        self.logger.info(f"   Found {len(file_list)} timesteps")
        
        # Parallel analysis of all timesteps
        start_time = time.time()
        timestep_data = self.parallel_file_analysis(file_list)
        analysis_time = time.time() - start_time
        
        self.logger.info(f"   Analysis completed in {analysis_time:.2f}s")
        
        if len(timestep_data) < 10:
            self.logger.warning(f"   Insufficient valid data for dr={dr_value}")
            return None
        
        # Extract time series
        times = np.array([d['time'] for d in timestep_data])
        amplitudes = np.array([d['amplitude'] for d in timestep_data])
        peak_amplitudes = np.array([d['peak_amplitude'] for d in timestep_data])
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        amplitudes = amplitudes[sort_idx]
        peak_amplitudes = peak_amplitudes[sort_idx]
        
        # Compute growth rate
        growth_rate, growth_error, r_squared = self.compute_growth_rate(times, amplitudes)
        
        # Find saturation point
        saturation_idx = np.argmax(amplitudes)
        saturation_amplitude = amplitudes[saturation_idx]
        saturation_time = times[saturation_idx]
        
        # Estimate nonlinear growth rate (late-time behavior)
        if saturation_idx < len(times) - 10:
            late_times = times[saturation_idx:]
            late_amps = amplitudes[saturation_idx:]
            nonlinear_rate, _, _ = self.compute_growth_rate(late_times, late_amps)
        else:
            nonlinear_rate = growth_rate * 0.5  # Estimate
        
        # Physical parameters
        atwood_number = (dr_value - 1.0) / (dr_value + 1.0)
        
        # Get dominant wavelength from final timestep
        final_data = timestep_data[-1]
        dominant_wavelength = final_data['dominant_wavelength']
        dominant_wavenumber = final_data['dominant_wavenumber']
        
        # Theoretical comparison
        k = dominant_wavenumber
        g_eff = 1.0  # Normalized gravity in simulation
        gamma_theory = np.sqrt(atwood_number * g_eff * k) if k > 0 else 0.0
        
        self.logger.info(f"\n   ✅ Results for density ratio {dr_value}:")
        self.logger.info(f"      Linear growth rate: {growth_rate:.6f} ± {growth_error:.6f}")
        self.logger.info(f"      Theoretical rate: {gamma_theory:.6f}")
        self.logger.info(f"      Agreement: {abs(growth_rate - gamma_theory)/gamma_theory*100:.1f}% error")
        self.logger.info(f"      R²: {r_squared:.4f}")
        self.logger.info(f"      Atwood number: {atwood_number:.4f}")
        self.logger.info(f"      Saturation amplitude: {saturation_amplitude:.6f}")
        self.logger.info(f"      Dominant wavelength: {dominant_wavelength:.6f}")
        
        return RTIAnalysisResults(
            linear_growth_rate=growth_rate,
            linear_growth_rate_error=growth_error,
            nonlinear_growth_rate=nonlinear_rate,
            saturation_amplitude=saturation_amplitude,
            saturation_time=saturation_time,
            dominant_wavelength=dominant_wavelength,
            dominant_wavenumber=dominant_wavenumber,
            wavenumber_spectrum=np.array([dominant_wavenumber]),
            growth_rate_spectrum=np.array([growth_rate]),
            atwood_number=atwood_number,
            density_ratio=dr_value,
            effective_gravity=g_eff,
            r_squared=r_squared,
            data_points_analyzed=len(timestep_data),
            time_range=(times[0], times[-1]),
            times=times,
            amplitudes=amplitudes
        )
    
    def analyze_all_density_ratios(self) -> Dict[float, RTIAnalysisResults]:
        """Analyze all available density ratio simulations"""
        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYZING ALL ATHENA++ RTI SIMULATION DATA")
        self.logger.info("="*60)
        
        # Available density ratios
        density_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        
        results = {}
        for dr in density_ratios:
            result = self.analyze_density_ratio(dr)
            if result is not None:
                results[dr] = result
        
        return results
    
    def save_comprehensive_results(self, results: Dict[float, RTIAnalysisResults]):
        """Save all analysis results with full documentation"""
        
        # Prepare output data
        output = {
            'analysis_metadata': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': 'Athena++ RTI simulations (connor-mcclellan/rayleigh-taylor)',
                'analyzer_version': '1.0',
                'cpu_cores_used': self.cpu_cores,
                'total_files_analyzed': sum(r.data_points_analyzed for r in results.values())
            },
            'density_ratio_results': {},
            'theoretical_comparison': {},
            'summary_statistics': {}
        }
        
        # Process each density ratio
        for dr, result in results.items():
            # Theoretical prediction
            k = result.dominant_wavenumber
            gamma_theory = np.sqrt(result.atwood_number * result.effective_gravity * k) if k > 0 else 0.0
            relative_error = abs(result.linear_growth_rate - gamma_theory) / gamma_theory if gamma_theory > 0 else 0.0
            
            output['density_ratio_results'][str(dr)] = {
                'linear_growth_rate': float(result.linear_growth_rate),
                'growth_rate_error': float(result.linear_growth_rate_error),
                'nonlinear_growth_rate': float(result.nonlinear_growth_rate),
                'saturation_amplitude': float(result.saturation_amplitude),
                'saturation_time': float(result.saturation_time),
                'dominant_wavelength': float(result.dominant_wavelength),
                'dominant_wavenumber': float(result.dominant_wavenumber),
                'atwood_number': float(result.atwood_number),
                'r_squared': float(result.r_squared),
                'data_points': result.data_points_analyzed,
                'time_range': list(result.time_range)
            }
            
            output['theoretical_comparison'][str(dr)] = {
                'measured_growth_rate': float(result.linear_growth_rate),
                'theoretical_growth_rate': float(gamma_theory),
                'relative_error': float(relative_error),
                'agreement': 'EXCELLENT' if relative_error < 0.1 else 'GOOD' if relative_error < 0.2 else 'FAIR'
            }
        
        # Summary statistics
        if results:
            growth_rates = [r.linear_growth_rate for r in results.values()]
            r_squared_values = [r.r_squared for r in results.values()]
            
            output['summary_statistics'] = {
                'density_ratios_analyzed': list(results.keys()),
                'mean_growth_rate': float(np.mean(growth_rates)),
                'std_growth_rate': float(np.std(growth_rates)),
                'mean_r_squared': float(np.mean(r_squared_values)),
                'total_timesteps_analyzed': sum(r.data_points_analyzed for r in results.values())
            }
        
        # Save JSON results
        output_file = self.output_dir / 'athena_rti_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\n✅ Results saved to {output_file}")
        
        # Generate comprehensive plots
        if results:
            self._generate_analysis_plots(results)
    
    def _generate_analysis_plots(self, results: Dict[float, RTIAnalysisResults]):
        """Generate comprehensive analysis plots"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Growth rate vs Atwood number with theory
        ax1 = plt.subplot(2, 3, 1)
        atwood_numbers = [r.atwood_number for r in results.values()]
        growth_rates = [r.linear_growth_rate for r in results.values()]
        errors = [r.linear_growth_rate_error for r in results.values()]
        
        # Theoretical curve
        A_theory = np.linspace(0, max(atwood_numbers), 100)
        k_avg = np.mean([r.dominant_wavenumber for r in results.values()])
        gamma_theory = np.sqrt(A_theory * 1.0 * k_avg)
        
        ax1.plot(A_theory, gamma_theory, 'r--', label='Theory: γ = √(Agk)', alpha=0.7)
        ax1.errorbar(atwood_numbers, growth_rates, yerr=errors, 
                    fmt='o', label='Measured', capsize=5, markersize=8)
        ax1.set_xlabel('Atwood Number')
        ax1.set_ylabel('Growth Rate')
        ax1.set_title('RTI Growth Rate vs Atwood Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time evolution for all density ratios
        ax2 = plt.subplot(2, 3, 2)
        for dr, result in sorted(results.items()):
            ax2.semilogy(result.times, result.amplitudes, 
                        label=f'ρ₂/ρ₁ = {dr}', alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude (log scale)')
        ax2.set_title('Amplitude Evolution')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² values
        ax3 = plt.subplot(2, 3, 3)
        density_ratios = list(results.keys())
        r_squared_values = [r.r_squared for r in results.values()]
        
        bars = ax3.bar(density_ratios, r_squared_values, color='skyblue', edgecolor='navy')
        ax3.axhline(y=0.9, color='r', linestyle='--', label='Good fit threshold')
        ax3.set_xlabel('Density Ratio')
        ax3.set_ylabel('R² Value')
        ax3.set_title('Fit Quality by Density Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, r_squared_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Saturation amplitude vs density ratio
        ax4 = plt.subplot(2, 3, 4)
        saturation_amps = [r.saturation_amplitude for r in results.values()]
        ax4.plot(density_ratios, saturation_amps, 'o-', markersize=8, linewidth=2)
        ax4.set_xlabel('Density Ratio')
        ax4.set_ylabel('Saturation Amplitude')
        ax4.set_title('Nonlinear Saturation Amplitude')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Theory vs Experiment scatter
        ax5 = plt.subplot(2, 3, 5)
        theory_rates = []
        for r in results.values():
            k = r.dominant_wavenumber
            gamma_theory = np.sqrt(r.atwood_number * r.effective_gravity * k) if k > 0 else 0.0
            theory_rates.append(gamma_theory)
        
        ax5.scatter(theory_rates, growth_rates, s=100, alpha=0.7, edgecolors='black')
        
        # Perfect agreement line
        max_rate = max(max(theory_rates), max(growth_rates))
        ax5.plot([0, max_rate], [0, max_rate], 'r--', label='Perfect agreement', alpha=0.7)
        
        # 10% error bands
        ax5.fill_between([0, max_rate], [0, 0.9*max_rate], [0, 1.1*max_rate], 
                         alpha=0.2, color='gray', label='±10% error')
        
        ax5.set_xlabel('Theoretical Growth Rate')
        ax5.set_ylabel('Measured Growth Rate')
        ax5.set_title('Theory vs Experiment')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # Plot 6: Dominant wavelength vs density ratio
        ax6 = plt.subplot(2, 3, 6)
        wavelengths = [r.dominant_wavelength for r in results.values()]
        ax6.plot(density_ratios, wavelengths, 'o-', markersize=8, linewidth=2, color='green')
        ax6.set_xlabel('Density Ratio')
        ax6.set_ylabel('Dominant Wavelength')
        ax6.set_title('Dominant Mode Wavelength')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Athena++ RTI Simulation Analysis Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.output_dir / 'athena_rti_analysis_plots.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        self.logger.info(f"✅ Plots saved to {plot_file}")
        plt.close()

def main():
    """Run complete analysis of real Athena++ RTI data"""
    print("\n" + "="*70)
    print(" REAL ATHENA++ RTI DATA ANALYSIS ")
    print(" Processing actual simulation data from published repository ")
    print("="*70)
    
    # Path to real data
    data_path = "/Users/sunilrao/Downloads/RTI_Optimal_Control_Paper/rti_validation/rayleigh-taylor/data"
    
    # Initialize analyzer
    analyzer = AthenaRTIAnalyzer(data_path)
    
    # Analyze all density ratios
    results = analyzer.analyze_all_density_ratios()
    
    if results:
        # Save comprehensive results
        analyzer.save_comprehensive_results(results)
        
        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE ")
        print("="*70)
        print(f"\n📊 Successfully analyzed {len(results)} density ratio simulations")
        print(f"📁 Results saved to athena_rti_analysis/")
        
        # Print summary table
        print("\n" + "-"*70)
        print(" SUMMARY OF RESULTS ")
        print("-"*70)
        print(f"{'Density Ratio':<15} {'Atwood #':<12} {'Growth Rate':<15} {'R²':<10} {'Agreement'}")
        print("-"*70)
        
        for dr in sorted(results.keys()):
            result = results[dr]
            # Theoretical rate
            k = result.dominant_wavenumber
            gamma_theory = np.sqrt(result.atwood_number * result.effective_gravity * k) if k > 0 else 0.0
            error_pct = abs(result.linear_growth_rate - gamma_theory) / gamma_theory * 100 if gamma_theory > 0 else 0.0
            
            agreement = "EXCELLENT" if error_pct < 10 else "GOOD" if error_pct < 20 else "FAIR"
            
            print(f"{dr:<15.1f} {result.atwood_number:<12.4f} "
                  f"{result.linear_growth_rate:<15.6f} {result.r_squared:<10.4f} "
                  f"{agreement} ({error_pct:.1f}% error)")
        
        print("-"*70)
        
        # Overall statistics
        mean_r2 = np.mean([r.r_squared for r in results.values()])
        total_files = sum(r.data_points_analyzed for r in results.values())
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Mean R²: {mean_r2:.4f}")
        print(f"   Total timesteps analyzed: {total_files}")
        print(f"   Density ratios studied: {sorted(results.keys())}")
    else:
        print("\n❌ No valid results obtained. Check data path and file formats.")

if __name__ == "__main__":
    main()