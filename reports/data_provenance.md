# Data Provenance and Reproducibility Statement

## Status: SYNTHETIC VALIDATION ONLY

### What This Validation Actually Is
- **Computational self-consistency checks** of the mathematical framework
- **Synthetic data generation** using the model's own equations with added noise
- **NO experimental validation** or PIC simulation comparison
- **NO calibrated parameters** from real laser-plasma experiments

### Reproducibility Information

#### Code Repository
- Location: `/rti_validation/scripts/`
- Language: Python 3.9
- Dependencies: numpy, scipy, matplotlib, control
- Random seed: 42 (for synthetic noise generation)

#### Synthetic Data Generation Parameters
```python
# Universal Collapse Tests
atwood_numbers = [0.1, 0.3, 0.5, 0.7, 0.9]
viscosities = np.logspace(-6, -3, 10)  
noise_level = 0.05  # 5% Gaussian noise
validity_window = [0.1, 0.9]  # x ∈ [ε, 1-ε]

# Bang-Bang Control Tests  
time_horizons = [0.1, 0.5, 1.0]  # seconds
IF_memory_params = {
    'alpha': 0.1,      # PLACEHOLDER - not calibrated
    'tau_B': 0.5       # PLACEHOLDER - not calibrated
}

# Model Constants (UNCALIBRATED)
C_QM = "TBD"  # Quantum efficiency
C_IF = "TBD"  # Ion front coupling
alpha = "TBD"  # Memory parameter
tau_B = "TBD"  # Relaxation time
```

### What Is Missing

#### Required for Physical Validation
1. **PIC Simulation Data**
   - Input decks with full parameter specifications
   - Grid resolution, particle counts, timesteps
   - Boundary conditions and initial perturbations
   - Output fields (density, E, B, phase space)

2. **Experimental Data**
   - Facility identification (OMEGA, NIF, etc.)
   - Shot numbers and dates
   - Laser parameters (I, λ, pulse shape, spot size)
   - Target specifications (material, thickness, density)
   - Diagnostic details (shadowgraphy, interferometry, etc.)
   - Error bars and systematic uncertainties

3. **Calibration Data**
   - Matched LP/CP shot pairs for constant extraction
   - Parameter ranges where model is valid
   - Breakdown conditions and failure modes

### Claimed Benchmarks (NOT VERIFIED)
The original report claimed comparison with:
- Palmer et al. 2012 - Actually about Kelvin-Helmholtz, not RTI control
- Sgattoni et al. 2015 - RPA studies, but no control validation
- Zhou 2017 - Review paper, no specific validation data

**None of these papers validate the specific theorems claimed.**

### Reproducibility Checklist

✗ Real experimental data with shot logs  
✗ PIC input files and version info  
✗ Calibrated model parameters  
✗ Independent validation by external group  
✗ Published benchmark comparisons  
✗ Error propagation analysis  
✗ Uncertainty quantification  
✓ Synthetic test code (limited value)  
✓ Internal consistency checks only  

### How to Reproduce Synthetic Results

```bash
cd /rti_validation
source venv/bin/activate
python scripts/validate_universal_collapse.py  # Generates synthetic data
python scripts/validate_bang_bang_control.py   # Tests switching function
```

**Warning:** These scripts only test mathematical self-consistency, not physical validity.

### Ethical Disclosure

This validation framework was created to demonstrate what a *proper* validation would require. The synthetic results show the model behaves as mathematically expected within its assumptions. However:

1. **No claim of physical accuracy** is made
2. **No experimental validation** exists  
3. **Critical parameters remain uncalibrated**
4. **Validity domain is extremely limited** (linear, 2D, thin-foil only)

### Required Actions Before Publication

1. **Obtain real data:** Partner with experimental group or PIC simulation team
2. **Calibrate constants:** Extract C_QM, C_IF, α, τ_B from LP/CP pairs
3. **Verify scope:** Test where model breaks down, document clearly
4. **Independent check:** Have external group reproduce key results
5. **Uncertainty analysis:** Propagate errors through full pipeline
6. **Nonlinear extension:** Address saturation, 3D effects, kinetic physics

## Bottom Line

The "validation" consists entirely of checking that the model's equations behave as the model predicts they should. This is necessary but vastly insufficient for claiming the model describes real plasma physics. 

**Current evidence level: Mathematical self-consistency only**  
**Physical validation level: NONE**