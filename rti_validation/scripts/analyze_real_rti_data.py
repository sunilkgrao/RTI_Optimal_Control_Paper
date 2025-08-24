#!/usr/bin/env python3
"""
Analyze Real Athena++ RTI Simulation Data
=========================================
This script processes REAL RTI simulation data from the Athena++ code.
"""

import h5py
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.fft import fft, fftfreq
from sklearn.linear_model import RANSACRegressor, HuberRegressor
import logging
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealRTIAnalyzer:
    """Analyzer for REAL Athena++ RTI simulation data."""
    
    def __init__(self, data_path: str):
        """Initialize the analyzer."""
        self.data_path = Path(data_path)
        self.results = {}
        self.cpu_cores = multiprocessing.cpu_count()
        self.g = 1.0  # Gravity acceleration
        
        logger.info(f"🚀 Real RTI Data Analyzer initialized")
        logger.info(f"   CPU cores: {self.cpu_cores}")
        logger.info(f"   Data path: {self.data_path}")
        
    def read_athena_file(self, filename: str) -> Dict:
        """Read a single Athena++ HDF5 output file."""
        with h5py.File(filename, 'r') as f:
            # Get primitive variables
            prim_data = np.array(f['prim'])
            
            # Get coordinates (shape: (nmeshblocks, n))
            x1v = np.array(f['x1v'])[0]  # Extract first meshblock
            x2v = np.array(f['x2v'])[0]  
            
            # Extract fields
            data = {
                'time': float(f.attrs['Time']),
                'density': prim_data[0, 0, 0, :, :],
                'pressure': prim_data[1, 0, 0, :, :],
                'velocity_x': prim_data[2, 0, 0, :, :],
                'velocity_y': prim_data[3, 0, 0, :, :],
                'x1v': x1v,
                'x2v': x2v,
            }
            
        return data
    
    def extract_interface_position(self, density: np.ndarray, x_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """Extract RTI interface position from density field."""
        ny, nx = density.shape
        interface_positions = []
        
        # For each horizontal position, find the interface
        for j in range(ny):
            density_profile = density[j, :]
            
            # Find interface as density midpoint
            rho_heavy = np.max(density_profile)
            rho_light = np.min(density_profile)
            rho_mean = (rho_heavy + rho_light) / 2.0
            
            # Find crossings
            crossings = np.where(np.diff(np.sign(density_profile - rho_mean)))[0]
            
            if len(crossings) > 0:
                idx = crossings[0]
                if idx < len(x_coords) - 1:
                    # Linear interpolation
                    x1, x2 = x_coords[idx], x_coords[idx + 1]
                    d1, d2 = density_profile[idx], density_profile[idx + 1]
                    
                    if abs(d2 - d1) > 1e-10:
                        x_interface = x1 + (rho_mean - d1) * (x2 - x1) / (d2 - d1)
                    else:
                        x_interface = (x1 + x2) / 2.0
                    
                    interface_positions.append(x_interface)
                else:
                    interface_positions.append(x_coords[min(idx, len(x_coords)-1)])
            else:
                interface_positions.append(0.0)
        
        interface_positions = np.array(interface_positions)
        
        # Calculate amplitude as RMS deviation
        mean_position = np.mean(interface_positions)
        amplitude = np.sqrt(np.mean((interface_positions - mean_position)**2))
        
        return amplitude, interface_positions
    
    def compute_growth_rate(self, times: np.ndarray, amplitudes: np.ndarray) -> Dict:
        """Compute RTI growth rate using RANSAC fitting."""
        # Filter out zero or negative amplitudes
        valid_mask = amplitudes > 1e-10
        if np.sum(valid_mask) < 3:
            return {'growth_rate': 0.0, 'r_squared': 0.0}
        
        times_valid = times[valid_mask]
        amplitudes_valid = amplitudes[valid_mask]
        log_amp = np.log(amplitudes_valid)
        t_fit = times_valid.reshape(-1, 1)
        
        try:
            ransac = RANSACRegressor(
                estimator=HuberRegressor(epsilon=1.35),
                min_samples=max(2, len(times_valid)//3),
                residual_threshold=0.1,
                random_state=42
            )
            ransac.fit(t_fit, log_amp)
            
            growth_rate = float(ransac.estimator_.coef_[0])
            y_pred = ransac.predict(t_fit)
            ss_res = np.sum((log_amp - y_pred)**2)
            ss_tot = np.sum((log_amp - np.mean(log_amp))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {'growth_rate': growth_rate, 'r_squared': r_squared}
            
        except:
            return {'growth_rate': 0.0, 'r_squared': 0.0}
    
    def analyze_single_file(self, filepath: Path) -> Dict:
        """Analyze a single Athena++ output file."""
        try:
            data = self.read_athena_file(str(filepath))
            amplitude, interface = self.extract_interface_position(
                data['density'], data['x1v']
            )
            
            # Compute spectral content
            if len(interface) > 1:
                fft_vals = fft(interface - np.mean(interface))
                freqs = fftfreq(len(interface), d=data['x2v'][1] - data['x2v'][0])
                positive_freqs = freqs[freqs > 0]
                positive_fft = np.abs(fft_vals[freqs > 0])
                if len(positive_fft) > 0:
                    dominant_idx = np.argmax(positive_fft)
                    dominant_k = 2 * np.pi * positive_freqs[dominant_idx]
                else:
                    dominant_k = 0.0
            else:
                dominant_k = 0.0
            
            kinetic_energy = 0.5 * np.mean(
                data['density'] * (data['velocity_x']**2 + data['velocity_y']**2)
            )
            
            return {
                'time': data['time'],
                'amplitude': amplitude,
                'dominant_k': dominant_k,
                'kinetic_energy': kinetic_energy,
            }
        except:
            return None
    
    def analyze_density_ratio(self, density_ratio: float) -> Dict:
        """Analyze all timesteps for a given density ratio."""
        dr_dir = self.data_path / f"rt_dr{density_ratio}"
        
        if not dr_dir.exists():
            logger.warning(f"   Directory not found: {dr_dir}")
            return None
        
        file_pattern = f"rt_dr{density_ratio}.out2.*.athdf"
        file_list = sorted(dr_dir.glob(file_pattern))
        
        if not file_list:
            return None
        
        logger.info(f"   Found {len(file_list)} timesteps")
        logger.info(f"   Processing files...")
        
        start_time = time.time()
        
        # Analyze files
        results = []
        for f in file_list[:50]:  # Process first 50 files for speed
            result = self.analyze_single_file(f)
            if result:
                results.append(result)
        
        if not results:
            return None
        
        # Extract time series
        times = np.array([d['time'] for d in results])
        amplitudes = np.array([d['amplitude'] for d in results])
        
        # Compute growth rate
        growth_analysis = self.compute_growth_rate(times, amplitudes)
        
        # Calculate Atwood number
        atwood = (density_ratio - 1.0) / (density_ratio + 1.0)
        
        # Theoretical growth rate
        k_values = [d['dominant_k'] for d in results if d['dominant_k'] > 0]
        if k_values:
            avg_k = np.mean(k_values)
            gamma_theory = np.sqrt(atwood * self.g * avg_k)
        else:
            gamma_theory = 0.0
        
        processing_time = time.time() - start_time
        
        result = {
            'density_ratio': density_ratio,
            'atwood_number': atwood,
            'num_timesteps': len(results),
            'growth_rate_measured': growth_analysis['growth_rate'],
            'growth_rate_theory': gamma_theory,
            'fit_r_squared': growth_analysis['r_squared'],
            'processing_time_seconds': processing_time
        }
        
        logger.info(f"   ✅ Complete in {processing_time:.2f}s")
        logger.info(f"   Growth rate: {growth_analysis['growth_rate']:.4f}")
        
        return result
    
    def analyze_all_density_ratios(self) -> Dict:
        """Analyze all available density ratio simulations."""
        density_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        all_results = {}
        
        for dr in density_ratios:
            logger.info(f"\n📊 Analyzing density ratio {dr} simulation...")
            result = self.analyze_density_ratio(dr)
            if result:
                all_results[f"dr_{dr}"] = result
        
        return all_results

def main():
    """Main execution function."""
    base_path = Path("/Users/sunilrao/Downloads/RTI_Optimal_Control_Paper/rti_validation")
    data_path = base_path / "rayleigh-taylor" / "data"
    
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        return
    
    analyzer = RealRTIAnalyzer(str(data_path))
    
    logger.info("\n" + "="*60)
    logger.info("ANALYZING REAL ATHENA++ RTI SIMULATION DATA")
    logger.info("="*60)
    
    results = analyzer.analyze_all_density_ratios()
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    report = {
        "validation_summary": {
            "timestamp": datetime.now().isoformat(),
            "data_source": "Athena++ RTI Simulations",
            "data_type": "REAL SIMULATION DATA"
        },
        "results": convert_numpy(results)
    }
    
    # Save results
    output_file = base_path / "scripts" / "REAL_RTI_VALIDATION_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("="*60)
    
    return report

if __name__ == "__main__":
    main()
