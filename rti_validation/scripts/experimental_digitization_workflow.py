#!/usr/bin/env python3
"""
Experimental Data Digitization Workflow
Implements the rigorous digitization protocol for RTI experimental data
"""

import numpy as np
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats
from sklearn.linear_model import HuberRegressor, RANSACRegressor

class ExperimentalDataDigitizer:
    def __init__(self, output_dir='../experimental_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ExpDigitizer')
        
        # Standard experimental parameters for reference
        self.known_facilities = {
            'OMEGA': {
                'wavelength': 351e-9,  # m, frequency-tripled
                'pulse_duration': 1e-12,  # s, short pulse
                'max_intensity': 5e19,  # W/cm²
                'facility_type': 'ICF_laser'
            },
            'Nike': {
                'wavelength': 248e-9,  # m, KrF
                'pulse_duration': 4e-9,  # s
                'max_intensity': 1e15,  # W/cm²
                'facility_type': 'ICF_laser'
            },
            'LULI': {
                'wavelength': 1053e-9,  # m
                'pulse_duration': 350e-15,  # s, femtosecond
                'max_intensity': 1e20,  # W/cm²
                'facility_type': 'ultraintense_laser'
            }
        }
    
    def create_digitization_template(self, paper_id, figure_panel):
        """Create a template for manual digitization with WebPlotDigitizer"""
        
        template = {
            'metadata': {
                'paper_id': paper_id,
                'figure_panel': figure_panel,
                'digitizer': 'WebPlotDigitizer',
                'digitization_date': datetime.now().isoformat(),
                'digitized_by': 'RTI_Validation_Pipeline',
                'validation_purpose': 'Universal_Collapse_Anchoring'
            },
            'experimental_parameters': {
                'facility': 'TBD',  # Fill from paper
                'target_material': 'TBD',
                'target_thickness_um': None,
                'density_gcm3': None,
                'laser_wavelength_nm': None,
                'laser_intensity_Wcm2': None,
                'pulse_duration_ps': None,
                'seeded_wavelength_um': None,
                'atwood_number': None,
                'diagnostic_type': 'TBD'  # X-ray, optical, etc.
            },
            'digitization_calibration': {
                'x_axis_units': 'TBD',  # k [1/μm], t [ps], etc.
                'y_axis_units': 'TBD',  # γ [1/ps], η [μm], etc.
                'x_min_pixel': None,
                'x_max_pixel': None,
                'x_min_value': None,
                'x_max_value': None,
                'y_min_pixel': None,
                'y_max_pixel': None,
                'y_min_value': None,
                'y_max_value': None,
                'calibration_points': []  # [(pixel_x, pixel_y, real_x, real_y), ...]
            },
            'data_points': [],  # Fill from digitization: [(x, y, uncertainty_x, uncertainty_y), ...]
            'quality_flags': {
                'linear_growth_window': True,
                'clear_error_bars': False,
                'mode_coupling_visible': False,
                'saturation_effects': False,
                'kd_ratio_valid': None  # < 0.3 for thin foil validity
            },
            'citation': {
                'doi': 'TBD',
                'first_author': 'TBD',
                'year': None,
                'journal': 'TBD',
                'title': 'TBD'
            }
        }
        
        # Save template
        template_file = self.output_dir / f'{paper_id}_{figure_panel}_template.yaml'
        with open(template_file, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Created digitization template: {template_file}")
        return template_file
    
    def validate_digitized_data(self, yaml_file):
        """Validate and process manually digitized data"""
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract data points
        points = np.array(data['data_points'])
        if len(points) == 0:
            raise ValueError("No data points found in YAML file")
        
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        x_err = points[:, 2] if points.shape[1] > 2 else None
        y_err = points[:, 3] if points.shape[1] > 3 else None
        
        # Basic validation checks
        validation_report = {
            'n_points': len(points),
            'x_range': [float(np.min(x_vals)), float(np.max(x_vals))],
            'y_range': [float(np.min(y_vals)), float(np.max(y_vals))],
            'monotonic_x': bool(np.all(np.diff(x_vals) > 0)),
            'finite_values': bool(np.all(np.isfinite(x_vals)) and np.all(np.isfinite(y_vals))),
            'kd_check': None
        }
        
        # Check kd ratio if thickness and wavelength available
        thickness = data['experimental_parameters'].get('target_thickness_um')
        wavelength = data['experimental_parameters'].get('seeded_wavelength_um')
        
        if thickness and wavelength:
            k = 2 * np.pi / wavelength  # 1/μm
            kd = k * thickness
            validation_report['kd_check'] = {
                'kd_value': float(kd),
                'thin_foil_valid': kd < 0.3,
                'requires_thickness_correction': kd > 0.3
            }
        
        # Save processed data
        processed_data = {
            'x': x_vals.tolist(),
            'y': y_vals.tolist(),
            'x_err': x_err.tolist() if x_err is not None else None,
            'y_err': y_err.tolist() if y_err is not None else None,
            'metadata': data['metadata'],
            'experimental_parameters': data['experimental_parameters'],
            'validation_report': validation_report
        }
        
        # Save as CSV for analysis
        paper_id = data['metadata']['paper_id']
        panel = data['metadata']['figure_panel']
        
        df = pd.DataFrame({
            'x': x_vals,
            'y': y_vals,
            'x_err': x_err if x_err is not None else np.nan,
            'y_err': y_err if y_err is not None else np.nan
        })
        
        csv_file = self.output_dir / f'{paper_id}_{panel}_data.csv'
        df.to_csv(csv_file, index=False)
        
        json_file = self.output_dir / f'{paper_id}_{panel}_processed.json'
        with open(json_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        self.logger.info(f"Validated data: {validation_report}")
        self.logger.info(f"Saved: {csv_file}, {json_file}")
        
        return processed_data, validation_report
    
    def create_anchor_point_targets(self):
        """Create specific targets for high-quality anchor points"""
        
        # Target 1: OMEGA RTI experiment (example parameters)
        omega_target = self.create_digitization_template('Smalyuk_2005', 'Fig3a')
        
        # Target 2: Nike/LULI RTI experiment (example parameters)  
        nike_target = self.create_digitization_template('Glendinning_2003', 'Fig5b')
        
        instructions = f"""
        DIGITIZATION INSTRUCTIONS FOR RTI ANCHOR POINTS
        ==============================================
        
        1. Open WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/)
        
        2. For each target paper:
           a) Load the figure image file
           b) Set axis calibration using at least 4 corner points
           c) Digitize 8-15 data points from growth rate curves
           d) Export as CSV and save coordinates
        
        3. Fill the YAML template with:
           - Experimental parameters from paper text
           - Axis calibration points (pixel → real coordinates)
           - Error bar estimates if available
           - Quality assessment flags
        
        4. Target papers (update with actual references):
           
           ANCHOR 1: OMEGA Facility RTI
           - Template: {omega_target}
           - Target: γ(k) growth rate vs wavenumber
           - Parameters: λ=351nm, I~10¹⁵ W/cm², thin foil
           - Quality: Linear window, clear k range, kd<0.3
           
           ANCHOR 2: Nike/LULI Facility RTI  
           - Template: {nike_target}
           - Target: Temporal growth η(t) at fixed k
           - Parameters: Different λ/I from Anchor 1
           - Quality: Exponential growth visible, mode purity
        
        5. Validation criteria for each anchor:
           - ✅ Clear experimental parameters documented
           - ✅ Axis calibration within 2% accuracy
           - ✅ 8+ data points in linear/exponential regime
           - ✅ Error estimates or uncertainty bounds
           - ✅ kd < 0.3 or explicit thickness correction
        
        These 2 anchors will pin the universal collapse normalization
        and provide the experimental validation for Fig. 1.
        """
        
        readme_file = self.output_dir / 'DIGITIZATION_INSTRUCTIONS.md'
        with open(readme_file, 'w') as f:
            f.write(instructions)
        
        self.logger.info(f"Created digitization targets and instructions: {readme_file}")
        return [omega_target, nike_target]

class GrowthRateExtractor:
    def __init__(self):
        self.logger = logging.getLogger('GrowthExtractor')
    
    def extract_growth_rate(self, x_data, y_data, x_err=None, y_err=None, 
                          data_type='gamma_k', window_method='auto'):
        """
        Extract growth rate following Section III.F protocol
        
        Parameters:
        -----------
        x_data, y_data : array_like
            Digitized data points (k, γ) or (t, η)
        data_type : str
            'gamma_k' for γ(k) data, 'eta_t' for η(t) temporal data
        window_method : str
            'auto' for automatic linear window selection
        """
        
        x = np.array(x_data)
        y = np.array(y_data)
        
        if data_type == 'eta_t':
            # Temporal data: extract growth rate from η(t)
            return self._extract_from_temporal(x, y, x_err, y_err)
        elif data_type == 'gamma_k':
            # Spectral data: process γ(k) directly
            return self._extract_from_spectral(x, y, x_err, y_err)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _extract_from_temporal(self, t, eta, t_err=None, eta_err=None):
        """Extract γ from temporal η(t) data using RANSAC/Huber fitting"""
        
        # Find linear growth window
        log_eta = np.log(np.abs(eta))
        
        # Sliding window to find best linear regime
        best_window = self._find_linear_window(t, log_eta)
        t_fit = t[best_window]
        log_eta_fit = log_eta[best_window]
        
        # RANSAC fit for robustness
        ransac = RANSACRegressor(
            base_estimator=HuberRegressor(epsilon=1.35),
            min_samples=max(2, len(t_fit)//3),
            residual_threshold=0.1,
            random_state=42
        )
        
        ransac.fit(t_fit.reshape(-1, 1), log_eta_fit)
        
        gamma = ransac.estimator_.coef_[0]
        gamma_err = self._estimate_uncertainty(t_fit, log_eta_fit, gamma)
        
        result = {
            'growth_rate': float(gamma),
            'growth_rate_error': float(gamma_err),
            'linear_window': best_window.tolist(),
            'r_squared': float(ransac.score(t_fit.reshape(-1, 1), log_eta_fit)),
            'n_inliers': int(np.sum(ransac.inlier_mask_)),
            'extraction_method': 'temporal_RANSAC'
        }
        
        return result
    
    def _extract_from_spectral(self, k, gamma, k_err=None, gamma_err=None):
        """Process spectral γ(k) data directly"""
        
        # Basic validation and filtering
        finite_mask = np.isfinite(k) & np.isfinite(gamma) & (gamma > 0)
        k_clean = k[finite_mask]
        gamma_clean = gamma[finite_mask]
        
        # Estimate uncertainties if not provided
        if gamma_err is None:
            gamma_err = 0.1 * gamma_clean  # 10% default uncertainty
        else:
            gamma_err = gamma_err[finite_mask]
        
        result = {
            'k_values': k_clean.tolist(),
            'gamma_values': gamma_clean.tolist(),
            'gamma_errors': gamma_err.tolist(),
            'n_points': len(k_clean),
            'k_range': [float(np.min(k_clean)), float(np.max(k_clean))],
            'extraction_method': 'spectral_direct'
        }
        
        return result
    
    def _find_linear_window(self, t, log_eta):
        """Find the best linear growth window using sliding R²"""
        
        n = len(t)
        if n < 4:
            return np.arange(n)
        
        min_window = max(4, n//4)
        best_score = -np.inf
        best_window = np.arange(min_window)
        
        for start in range(n - min_window):
            for end in range(start + min_window, n + 1):
                window = np.arange(start, end)
                t_win = t[window]
                log_eta_win = log_eta[window]
                
                # Linear regression R²
                slope, intercept, r_value, _, _ = stats.linregress(t_win, log_eta_win)
                score = r_value**2
                
                if score > best_score:
                    best_score = score
                    best_window = window
        
        return best_window
    
    def _estimate_uncertainty(self, t, log_eta, gamma):
        """Estimate uncertainty in growth rate using residuals"""
        
        predicted = gamma * t + np.mean(log_eta - gamma * t)
        residuals = log_eta - predicted
        mse = np.mean(residuals**2)
        
        # Uncertainty in slope from linear regression theory
        t_var = np.var(t)
        n = len(t)
        gamma_err = np.sqrt(mse / (t_var * n))
        
        return gamma_err

def create_example_anchor_points():
    """Create example anchor points for testing the pipeline"""
    
    digitizer = ExperimentalDataDigitizer()
    
    # Example 1: OMEGA-style γ(k) data
    omega_template = digitizer.create_digitization_template('Example_OMEGA', 'Fig1')
    
    # Example 2: Nike-style η(t) data
    nike_template = digitizer.create_digitization_template('Example_Nike', 'Fig2')
    
    # Create synthetic data for testing
    example_gamma_k = {
        'data_points': [
            [0.5, 0.12, 0.02, 0.01],  # [k, γ, k_err, γ_err]
            [0.8, 0.18, 0.02, 0.015],
            [1.2, 0.24, 0.03, 0.02],
            [1.8, 0.29, 0.03, 0.025],
            [2.5, 0.31, 0.04, 0.03],
            [3.2, 0.28, 0.05, 0.035],
        ]
    }
    
    example_eta_t = {
        'data_points': [
            [1.0, 0.1, 0.1, 0.01],   # [t, η, t_err, η_err]
            [2.0, 0.15, 0.1, 0.015],
            [3.0, 0.22, 0.1, 0.02],
            [4.0, 0.33, 0.1, 0.03],
            [5.0, 0.48, 0.1, 0.04],
            [6.0, 0.72, 0.1, 0.06],
        ]
    }
    
    # Update templates with example data
    with open(omega_template, 'r') as f:
        omega_data = yaml.safe_load(f)
    omega_data['data_points'] = example_gamma_k['data_points']
    omega_data['experimental_parameters']['facility'] = 'OMEGA_Example'
    
    with open(omega_template, 'w') as f:
        yaml.dump(omega_data, f, default_flow_style=False)
    
    with open(nike_template, 'r') as f:
        nike_data = yaml.safe_load(f)
    nike_data['data_points'] = example_eta_t['data_points']
    nike_data['experimental_parameters']['facility'] = 'Nike_Example'
    
    with open(nike_template, 'w') as f:
        yaml.dump(nike_data, f, default_flow_style=False)
    
    print("✅ Example anchor points created for testing pipeline")
    return omega_template, nike_template

if __name__ == "__main__":
    print("🔬 EXPERIMENTAL DATA DIGITIZATION WORKFLOW")
    print("=" * 50)
    
    # Create the digitization infrastructure
    omega_file, nike_file = create_example_anchor_points()
    
    # Test the extraction pipeline
    digitizer = ExperimentalDataDigitizer()
    extractor = GrowthRateExtractor()
    
    # Process OMEGA-style data
    omega_processed, omega_validation = digitizer.validate_digitized_data(omega_file)
    omega_extraction = extractor.extract_growth_rate(
        omega_processed['x'], 
        omega_processed['y'],
        data_type='gamma_k'
    )
    
    # Process Nike-style data
    nike_processed, nike_validation = digitizer.validate_digitized_data(nike_file) 
    nike_extraction = extractor.extract_growth_rate(
        nike_processed['x'],
        nike_processed['y'], 
        data_type='eta_t'
    )
    
    print(f"\n📊 OMEGA Data Extraction:")
    print(f"   Points: {omega_extraction['n_points']}")
    print(f"   k range: {omega_extraction['k_range']}")
    
    print(f"\n📊 Nike Data Extraction:")
    print(f"   Growth rate: {nike_extraction['growth_rate']:.4f} ± {nike_extraction['growth_rate_error']:.4f}")
    print(f"   R²: {nike_extraction['r_squared']:.4f}")
    
    print(f"\n✅ Experimental digitization workflow ready!")
    print(f"   Next step: Replace example data with real digitized points from papers")