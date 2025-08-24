#!/usr/bin/env python3
"""
Growth Rate Extraction Pipeline
Extracts and analyzes RTI growth rates from simulation and experimental data
"""

import numpy as np
from scipy import stats, signal
from scipy.optimize import curve_fit
import pandas as pd
import json
import os
import glob
import h5py
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GrowthRateExtractor')

# Try to import PyDMD, use fallback if not available
try:
    from pydmd import DMD
    HAS_DMD = True
except ImportError:
    logger.warning("PyDMD not available, using fallback methods")
    HAS_DMD = False

class GrowthRateExtractor:
    def __init__(self, data_directory='../simulations', output_dir='../analysis'):
        self.data_dir = data_directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        self.extraction_methods = ['linear_fit', 'exponential_fit', 'dmd', 'fourier']
        
    def parse_filename(self, filename):
        """Extract parameters from filename"""
        
        # Expected format: input_A0.5_nu1.00e-04_res512_output.npz
        parts = filename.replace('.npz', '').replace('.json', '').split('_')
        
        params = {
            'id': filename,
            'atwood': 0.5,
            'viscosity': 0,
            'resolution': 512
        }
        
        for part in parts:
            if part.startswith('A'):
                params['atwood'] = float(part[1:])
            elif part.startswith('nu'):
                params['viscosity'] = float(part[2:])
            elif part.startswith('res'):
                params['resolution'] = int(part[3:])
        
        return params
    
    def load_simulation_data(self, filepath):
        """Load simulation data from various formats"""
        
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            return {
                'times': data['times'],
                'amplitudes': data.get('mixing_widths', data.get('amplitudes', [])),
                'format': 'npz'
            }
        elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
            with h5py.File(filepath, 'r') as f:
                return {
                    'times': f['time'][:],
                    'amplitudes': f.get('amplitude', f.get('mixing_width', []))[:]
    ,
                    'format': 'hdf5'
                }
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return {
                    'times': np.array(data.get('times', [])),
                    'amplitudes': np.array(data.get('amplitudes', [])),
                    'format': 'json'
                }
        else:
            logger.warning(f"Unknown file format: {filepath}")
            return None
    
    def extract_linear_growth_rate(self, times, amplitudes):
        """Extract growth rate from linear regime using log-linear fit"""
        
        # Remove zero or negative values
        valid_mask = amplitudes > 0
        if np.sum(valid_mask) < 5:
            return None
        
        t_valid = times[valid_mask]
        a_valid = amplitudes[valid_mask]
        
        # Find linear regime (typically first 10-30% of growth)
        # Look for constant slope in log space
        log_a = np.log(a_valid)
        
        # Use sliding window to find most linear region
        window_size = min(10, len(t_valid) // 3)
        if window_size < 5:
            window_size = min(5, len(t_valid))
        
        best_r2 = 0
        best_params = None
        
        for i in range(len(t_valid) - window_size):
            t_window = t_valid[i:i+window_size]
            log_a_window = log_a[i:i+window_size]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(t_window, log_a_window)
            
            r2 = r_value**2
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {
                    'growth_rate': slope,
                    'amplitude_0': np.exp(intercept),
                    'r_squared': r2,
                    'std_error': std_err,
                    'time_range': [t_window[0], t_window[-1]],
                    'n_points': window_size
                }
        
        return best_params
    
    def extract_exponential_fit(self, times, amplitudes):
        """Fit exponential growth model a(t) = a0 * exp(γt)"""
        
        # Remove initial transients
        start_idx = min(5, len(times) // 10)
        t_fit = times[start_idx:]
        a_fit = amplitudes[start_idx:]
        
        # Exponential model
        def exp_model(t, a0, gamma):
            return a0 * np.exp(gamma * t)
        
        try:
            # Initial guess
            gamma_guess = (np.log(a_fit[-1]) - np.log(a_fit[0])) / (t_fit[-1] - t_fit[0])
            a0_guess = a_fit[0] * np.exp(-gamma_guess * t_fit[0])
            
            popt, pcov = curve_fit(exp_model, t_fit, a_fit,
                                  p0=[a0_guess, gamma_guess],
                                  maxfev=5000)
            
            # Calculate R-squared
            residuals = a_fit - exp_model(t_fit, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((a_fit - np.mean(a_fit))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'growth_rate': popt[1],
                'amplitude_0': popt[0],
                'r_squared': r_squared,
                'std_error': np.sqrt(np.diag(pcov))[1] if len(pcov) > 1 else 0,
                'time_range': [t_fit[0], t_fit[-1]]
            }
            
        except Exception as e:
            logger.warning(f"Exponential fit failed: {e}")
            return None
    
    def extract_dmd_growth_rate(self, times, amplitudes):
        """Extract growth rate using Dynamic Mode Decomposition"""
        
        if not HAS_DMD or len(amplitudes) < 10:
            return None
        
        try:
            # Prepare data matrix (snapshots)
            # For 1D data, create Hankel matrix
            n_delays = min(10, len(amplitudes) // 2)
            data_matrix = np.zeros((n_delays, len(amplitudes) - n_delays))
            
            for i in range(n_delays):
                data_matrix[i, :] = amplitudes[i:i+len(amplitudes)-n_delays]
            
            # Perform DMD
            dmd = DMD(svd_rank=min(5, n_delays-1))
            dmd.fit(data_matrix)
            
            # Extract growth rates from eigenvalues
            dt = np.mean(np.diff(times))
            eigenvalues = dmd.eigs
            
            # Growth rates are real parts of log(eigenvalues)/dt
            growth_rates = np.real(np.log(eigenvalues + 1e-10) / dt)
            
            # Amplitudes of modes
            mode_amplitudes = np.abs(dmd.amplitudes)
            
            # Select dominant mode (largest amplitude)
            dominant_idx = np.argmax(mode_amplitudes)
            dominant_growth = growth_rates[dominant_idx]
            
            # Reconstruction quality
            reconstructed = dmd.reconstructed_data.real
            if reconstructed.shape[1] > 0:
                reconstruction_error = np.mean((data_matrix - reconstructed)**2)
                r_squared = 1 - reconstruction_error / np.var(data_matrix)
            else:
                r_squared = 0
            
            return {
                'growth_rate': dominant_growth,
                'all_growth_rates': growth_rates.tolist(),
                'mode_amplitudes': mode_amplitudes.tolist(),
                'r_squared': r_squared,
                'n_modes': len(eigenvalues)
            }
            
        except Exception as e:
            logger.warning(f"DMD analysis failed: {e}")
            return None
    
    def extract_fourier_growth_rate(self, times, amplitudes):
        """Extract growth rate from Fourier analysis of oscillations"""
        
        if len(amplitudes) < 10:
            return None
        
        try:
            # Detrend data
            detrended = signal.detrend(amplitudes)
            
            # Compute FFT
            dt = np.mean(np.diff(times))
            frequencies = np.fft.fftfreq(len(detrended), dt)
            fft_values = np.fft.fft(detrended)
            
            # Find dominant frequency (exclude DC component)
            positive_freq_idx = frequencies > 0
            dominant_idx = np.argmax(np.abs(fft_values[positive_freq_idx]))
            dominant_freq = frequencies[positive_freq_idx][dominant_idx]
            
            # Growth rate from envelope
            envelope = np.abs(signal.hilbert(amplitudes))
            
            # Fit exponential to envelope
            if np.all(envelope > 0):
                log_env = np.log(envelope)
                slope, intercept, r_value, p_value, std_err = \
                    stats.linregress(times, log_env)
                
                return {
                    'growth_rate': slope,
                    'dominant_frequency': dominant_freq,
                    'r_squared': r_value**2,
                    'std_error': std_err
                }
            
        except Exception as e:
            logger.warning(f"Fourier analysis failed: {e}")
            
        return None
    
    def compare_extraction_methods(self, times, amplitudes):
        """Compare different growth rate extraction methods"""
        
        methods_results = {}
        
        # Linear fit
        linear_result = self.extract_linear_growth_rate(times, amplitudes)
        if linear_result:
            methods_results['linear'] = linear_result
        
        # Exponential fit
        exp_result = self.extract_exponential_fit(times, amplitudes)
        if exp_result:
            methods_results['exponential'] = exp_result
        
        # DMD
        dmd_result = self.extract_dmd_growth_rate(times, amplitudes)
        if dmd_result:
            methods_results['dmd'] = dmd_result
        
        # Fourier
        fourier_result = self.extract_fourier_growth_rate(times, amplitudes)
        if fourier_result:
            methods_results['fourier'] = fourier_result
        
        # Select best method based on R-squared
        if methods_results:
            best_method = max(methods_results.keys(),
                            key=lambda k: methods_results[k].get('r_squared', 0))
            best_result = methods_results[best_method]
            best_result['method'] = best_method
            best_result['all_methods'] = methods_results
            
            return best_result
        
        return None
    
    def process_all_simulations(self):
        """Process all simulation files in data directory"""
        
        logger.info("Processing all simulation files...")
        
        # Find all data files
        patterns = ['*.npz', '*.h5', '*.hdf5', '*.json']
        all_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(self.data_dir, pattern))
            all_files.extend(files)
        
        print(f"Found {len(all_files)} data files to process")
        
        results_list = []
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            # Skip input files
            if 'input' in filename and 'output' not in filename:
                continue
            
            logger.info(f"Processing: {filename}")
            
            # Parse parameters
            params = self.parse_filename(filename)
            
            # Load data
            data = self.load_simulation_data(filepath)
            
            if data and len(data['times']) > 0 and len(data['amplitudes']) > 0:
                # Extract growth rate
                growth_result = self.compare_extraction_methods(
                    data['times'], data['amplitudes']
                )
                
                if growth_result:
                    # Combine parameters and results
                    result_entry = {**params, **growth_result}
                    results_list.append(result_entry)
                    
                    print(f"  ✓ A={params['atwood']:.1f}, γ={growth_result['growth_rate']:.3f} /s")
        
        # Create DataFrame
        if results_list:
            df = pd.DataFrame(results_list)
            
            # Add theoretical predictions
            df['growth_rate_theory'] = np.sqrt(
                df['atwood'] * 9.81 * 2 * np.pi / 0.01
            )
            
            # Calculate relative error
            df['relative_error'] = abs(
                df['growth_rate'] - df['growth_rate_theory']
            ) / df['growth_rate_theory']
            
            # Save to CSV
            output_file = os.path.join(self.output_dir, 'extracted_growth_rates.csv')
            df.to_csv(output_file, index=False)
            
            logger.info(f"Saved results to: {output_file}")
            
            return df
        else:
            logger.warning("No valid results extracted")
            return pd.DataFrame()
    
    def plot_growth_rate_comparison(self, df):
        """Create comparison plots of extracted vs theoretical growth rates"""
        
        if df.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Extracted vs Theoretical
        ax1 = axes[0, 0]
        
        ax1.scatter(df['growth_rate_theory'], df['growth_rate'],
                   alpha=0.6, s=50)
        
        # Perfect agreement line
        min_val = min(df['growth_rate_theory'].min(), df['growth_rate'].min())
        max_val = max(df['growth_rate_theory'].max(), df['growth_rate'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect agreement')
        
        # 10% error bounds
        ax1.fill_between([min_val, max_val],
                        [0.9*min_val, 0.9*max_val],
                        [1.1*min_val, 1.1*max_val],
                        alpha=0.2, color='gray', label='±10% bounds')
        
        ax1.set_xlabel('Theoretical Growth Rate (1/s)')
        ax1.set_ylabel('Extracted Growth Rate (1/s)')
        ax1.set_title('Growth Rate Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Growth rate vs Atwood number
        ax2 = axes[0, 1]
        
        # Group by Atwood number
        atwood_groups = df.groupby('atwood').agg({
            'growth_rate': ['mean', 'std'],
            'growth_rate_theory': 'mean'
        })
        
        atwood_values = atwood_groups.index
        measured_means = atwood_groups['growth_rate']['mean']
        measured_stds = atwood_groups['growth_rate']['std']
        theory_values = atwood_groups['growth_rate_theory']['mean']
        
        ax2.errorbar(atwood_values, measured_means, yerr=measured_stds,
                    fmt='o-', label='Measured', capsize=5)
        ax2.plot(atwood_values, theory_values, 's--', label='Theory')
        
        ax2.set_xlabel('Atwood Number')
        ax2.set_ylabel('Growth Rate (1/s)')
        ax2.set_title('Growth Rate Scaling with Atwood Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative error distribution
        ax3 = axes[1, 0]
        
        ax3.hist(df['relative_error'] * 100, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=5, color='g', linestyle='--', label='5% threshold')
        ax3.axvline(x=10, color='r', linestyle='--', label='10% threshold')
        
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Count')
        ax3.set_title('Error Distribution')
        ax3.legend()
        
        # Plot 4: Method comparison
        ax4 = axes[1, 1]
        
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            ax4.bar(method_counts.index, method_counts.values, alpha=0.7)
            ax4.set_xlabel('Extraction Method')
            ax4.set_ylabel('Times Selected as Best')
            ax4.set_title('Best Method Selection Frequency')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'growth_rate_comparison.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {output_file}")
        
        return output_file
    
    def generate_summary_report(self, df):
        """Generate summary report of growth rate extraction"""
        
        if df.empty:
            return {}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_simulations': len(df),
            'atwood_range': [df['atwood'].min(), df['atwood'].max()],
            'mean_relative_error': df['relative_error'].mean(),
            'std_relative_error': df['relative_error'].std(),
            'within_5_percent': (df['relative_error'] < 0.05).sum() / len(df) * 100,
            'within_10_percent': (df['relative_error'] < 0.10).sum() / len(df) * 100,
            'best_method_frequency': df['method'].value_counts().to_dict() if 'method' in df.columns else {},
            'validation_passed': df['relative_error'].mean() < 0.1
        }
        
        # Save summary
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        summary = convert_numpy_types(summary)
        
        output_file = os.path.join(self.output_dir, 'growth_rate_extraction_summary.json')
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to: {output_file}")
        
        return summary

def main():
    """Main execution"""
    print("=== Growth Rate Extraction Pipeline ===")
    
    extractor = GrowthRateExtractor()
    
    # Process all simulations
    df = extractor.process_all_simulations()
    
    if not df.empty:
        # Create comparison plots
        plot_file = extractor.plot_growth_rate_comparison(df)
        
        # Generate summary report
        summary = extractor.generate_summary_report(df)
        
        print(f"\n✓ Processed {len(df)} simulations")
        print(f"✓ Mean relative error: {summary['mean_relative_error']*100:.2f}%")
        print(f"✓ Within 10% accuracy: {summary['within_10_percent']:.1f}%")
        
        if summary['validation_passed']:
            print("✓ Growth rate extraction validated!")
        else:
            print("✗ Growth rates show significant deviation - review results")
    else:
        print("✗ No simulation data found to process")

if __name__ == "__main__":
    main()