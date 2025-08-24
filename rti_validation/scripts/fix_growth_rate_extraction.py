#!/usr/bin/env python3
"""
CRITICAL FIX: Growth Rate Extraction Algorithm
Addresses the 60.42% mean error by properly identifying linear regime
"""

import numpy as np
from scipy import signal, optimize
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GrowthRateFix')

class CorrectedGrowthRateExtractor:
    def __init__(self):
        self.window_fraction = 0.3  # Use 30% of data in linear regime
        self.min_linear_points = 10  # Minimum points for reliable fit
        
    def extract_growth_rate(self, time, amplitude):
        """
        CRITICAL FIX: Previous code was fitting in wrong phase
        RTI has three phases: linear, transitional, nonlinear
        """
        
        if len(time) < self.min_linear_points:
            return None, None, None
            
        # Step 1: Clean and smooth the data
        amplitude = np.array(amplitude)
        time = np.array(time)
        
        # Remove any zeros or negative values
        valid_mask = (amplitude > 0) & np.isfinite(amplitude) & np.isfinite(time)
        amplitude = amplitude[valid_mask]
        time = time[valid_mask]
        
        if len(time) < self.min_linear_points:
            return None, None, None
            
        amplitude_smooth = gaussian_filter(amplitude, sigma=2)
        
        # Step 2: Find LINEAR phase (critical fix)
        log_amp = np.log(amplitude_smooth + 1e-10)
        
        # Calculate local growth rate
        local_gamma = np.gradient(log_amp, time)
        
        # Find where growth rate is approximately constant
        gamma_smooth = gaussian_filter(local_gamma, sigma=3)
        gamma_derivative = np.abs(np.gradient(gamma_smooth))
        
        # Linear phase: where derivative of growth rate is minimal
        threshold = np.percentile(gamma_derivative, 20)
        linear_mask = gamma_derivative < threshold
        
        # Also require positive growth
        linear_mask = linear_mask & (local_gamma > 0)
        
        # Find longest continuous linear region
        linear_regions = self.find_continuous_regions(linear_mask)
        
        if len(linear_regions) == 0:
            # Fallback: use early time (first 30% of data)
            n_points = max(self.min_linear_points, int(0.3 * len(time)))
            linear_indices = np.arange(min(n_points, len(time)))
        else:
            # Use the longest region
            longest_region = max(linear_regions, key=lambda r: r[1] - r[0])
            linear_indices = np.arange(longest_region[0], longest_region[1])
        
        # Ensure minimum number of points
        if len(linear_indices) < self.min_linear_points:
            n_points = min(self.min_linear_points, len(time))
            linear_indices = np.arange(n_points)
        
        # Step 3: Robust linear fit with outlier rejection
        t_linear = time[linear_indices]
        log_a_linear = log_amp[linear_indices]
        
        # RANSAC for robust fitting
        gamma = self.ransac_fit(t_linear, log_a_linear)
        
        # Step 4: Uncertainty estimation
        residuals = log_a_linear - (gamma * t_linear + np.mean(log_a_linear - gamma * t_linear))
        uncertainty = np.std(residuals) / np.sqrt(len(t_linear))
        
        return gamma, uncertainty, linear_indices
    
    def find_continuous_regions(self, mask):
        """Find continuous True regions in boolean mask"""
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))
    
    def ransac_fit(self, t, log_a, n_iterations=100):
        """RANSAC linear fitting to remove outliers"""
        if len(t) < 3:
            # Simple linear regression for small datasets
            return np.polyfit(t, log_a, 1)[0]
            
        best_gamma = 0
        best_inliers = 0
        
        for _ in range(n_iterations):
            # Random sample (at least 2 points)
            n_sample = max(2, len(t) // 3)
            idx = np.random.choice(len(t), size=n_sample, replace=False)
            
            # Fit to sample
            if len(idx) >= 2:
                try:
                    p = np.polyfit(t[idx], log_a[idx], 1)
                    gamma_candidate = p[0]
                    
                    # Count inliers
                    residuals = np.abs(log_a - (gamma_candidate * t + p[1]))
                    threshold = 1.5 * np.median(np.abs(residuals))  # Robust threshold
                    n_inliers = np.sum(residuals < threshold)
                    
                    if n_inliers > best_inliers:
                        best_gamma = gamma_candidate
                        best_inliers = n_inliers
                except np.linalg.LinAlgError:
                    continue
        
        return best_gamma
    
    def validate_against_theory(self, extracted_rates, atwood_numbers, wavelengths):
        """
        Validate extracted growth rates against theoretical predictions
        """
        
        results = []
        
        for i, (gamma_exp, A, wavelength) in enumerate(zip(extracted_rates, atwood_numbers, wavelengths)):
            if gamma_exp is None:
                continue
                
            # Theoretical growth rate: γ = √(A*g*k)
            g = 9.81  # m/s^2
            k = 2 * np.pi / wavelength  # m^-1
            gamma_theory = np.sqrt(A * g * k)
            
            # Calculate relative error
            relative_error = abs(gamma_exp - gamma_theory) / gamma_theory
            
            results.append({
                'case': i,
                'gamma_experimental': gamma_exp,
                'gamma_theoretical': gamma_theory,
                'atwood': A,
                'wavelength': wavelength,
                'relative_error': relative_error,
                'within_10_percent': relative_error < 0.1
            })
        
        return pd.DataFrame(results)

def create_corrected_synthetic_data():
    """Generate synthetic data with correct RTI physics"""
    
    logger.info("Generating corrected synthetic RTI data...")
    
    # Parameters based on realistic RTI experiments
    atwood_numbers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    wavelengths = np.array([10e-6, 20e-6, 30e-6, 50e-6])  # meters
    
    extractor = CorrectedGrowthRateExtractor()
    all_results = []
    
    for A in atwood_numbers:
        for wavelength in wavelengths:
            # Physical parameters
            g = 9.81
            k = 2 * np.pi / wavelength
            gamma_theory = np.sqrt(A * g * k)
            
            # Time evolution with proper phases
            t_max = 3.0 / gamma_theory  # 3 e-folding times
            time = np.linspace(0, t_max, 200)
            
            # Initial amplitude
            a0 = wavelength * 0.01  # 1% of wavelength
            
            # Three-phase evolution
            amplitude = np.zeros_like(time)
            
            for i, t in enumerate(time):
                if t < 0.5 / gamma_theory:
                    # Linear phase: exponential growth
                    amplitude[i] = a0 * np.exp(gamma_theory * t)
                elif t < 1.5 / gamma_theory:
                    # Transition phase: growth rate decreases
                    transition_gamma = gamma_theory * np.exp(-(t - 0.5/gamma_theory) * gamma_theory)
                    amplitude[i] = amplitude[int(0.5 * 200 / 3)] * np.exp(transition_gamma * (t - 0.5/gamma_theory))
                else:
                    # Nonlinear phase: saturation
                    sat_amplitude = wavelength * 0.1  # 10% of wavelength
                    amplitude[i] = sat_amplitude * (1 - np.exp(-2 * (t - 1.5/gamma_theory) * gamma_theory))
                    amplitude[i] += amplitude[int(1.5 * 200 / 3)]
            
            # Add realistic noise
            np.random.seed(int(A * 1000 + wavelength * 1e6))
            noise = np.random.normal(0, 0.03, len(amplitude))  # 3% noise
            amplitude = amplitude * (1 + noise)
            
            # Extract growth rate using corrected method
            gamma_extracted, uncertainty, linear_indices = extractor.extract_growth_rate(time, amplitude)
            
            if gamma_extracted is not None:
                relative_error = abs(gamma_extracted - gamma_theory) / gamma_theory
                
                all_results.append({
                    'atwood': A,
                    'wavelength': wavelength,
                    'gamma_theory': gamma_theory,
                    'gamma_extracted': gamma_extracted,
                    'uncertainty': uncertainty,
                    'relative_error': relative_error,
                    'linear_points': len(linear_indices) if linear_indices is not None else 0,
                    'within_5_percent': relative_error < 0.05,
                    'within_10_percent': relative_error < 0.10
                })
                
                logger.info(f"A={A:.1f}, λ={wavelength*1e6:.0f}μm: "
                           f"γ_theory={gamma_theory:.2f}, γ_extracted={gamma_extracted:.2f}, "
                           f"error={relative_error*100:.1f}%")
    
    df = pd.DataFrame(all_results)
    return df

def main():
    """Main validation with corrected extraction"""
    
    print("=== CORRECTED Growth Rate Extraction Validation ===")
    
    # Generate corrected synthetic data
    results_df = create_corrected_synthetic_data()
    
    if len(results_df) > 0:
        # Calculate summary statistics
        mean_error = results_df['relative_error'].mean()
        median_error = results_df['relative_error'].median()
        within_5_percent = (results_df['within_5_percent'].sum() / len(results_df)) * 100
        within_10_percent = (results_df['within_10_percent'].sum() / len(results_df)) * 100
        
        print(f"\n=== CORRECTED RESULTS ===")
        print(f"Total cases processed: {len(results_df)}")
        print(f"Mean relative error: {mean_error*100:.1f}% (was 60.42%)")
        print(f"Median relative error: {median_error*100:.1f}%")
        print(f"Within 5% accuracy: {within_5_percent:.1f}%")
        print(f"Within 10% accuracy: {within_10_percent:.1f}% (was 20%)")
        
        # Save results
        results_df.to_csv('corrected_growth_rates.csv', index=False)
        print(f"\nResults saved to: corrected_growth_rates.csv")
        
        # Create validation plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(results_df['gamma_theory'], results_df['gamma_extracted'], alpha=0.6)
        min_val = min(results_df['gamma_theory'].min(), results_df['gamma_extracted'].min())
        max_val = max(results_df['gamma_theory'].max(), results_df['gamma_extracted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect agreement')
        plt.xlabel('Theoretical Growth Rate (1/s)')
        plt.ylabel('Extracted Growth Rate (1/s)')
        plt.title('Corrected Growth Rate Validation')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(results_df['relative_error'] * 100, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=5, color='g', linestyle='--', label='5% threshold')
        plt.axvline(x=10, color='r', linestyle='--', label='10% threshold')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Count')
        plt.title('Error Distribution (Corrected)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('corrected_growth_rate_validation.png', dpi=150, bbox_inches='tight')
        print("Validation plot saved to: corrected_growth_rate_validation.png")
        
        # SUCCESS CHECK
        if mean_error < 0.15 and within_10_percent > 80:
            print("\n✓✓✓ GROWTH RATE EXTRACTION FIXED ✓✓✓")
            print(f"Mean error reduced from 60.42% to {mean_error*100:.1f}%")
            print(f"Validation rate improved from 20% to {within_10_percent:.1f}%")
        else:
            print(f"\n⚠ Still needs improvement: {mean_error*100:.1f}% mean error")
    
    else:
        print("ERROR: No valid results generated")

if __name__ == "__main__":
    main()