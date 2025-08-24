PARAMETER SWEEP VALIDATION DATA
===============================

This directory contains the comprehensive parameter sweep data supporting the manuscript:
"Optimal Control and Leading-Order Thin-Foil Universality for RTI-Limited Radiation-Pressure Acceleration"

Author: Sunil Rao
ORCID: 0009-0009-3337-5726

DATA FILE DESCRIPTION
--------------------

enhanced_physical_sweep_results.csv (68 MB, 567,000 entries)
- Physically-constrained parameter sweep focusing on RTI-relevant regimes
- Columns: a0, ne_nc, thickness_nm, k_over_klaser, incidence_angle, polarization, growth_rate, k_cutoff, similarity_number
- Parameter ranges:
  * a0: [8, 60] (35 values, log-spaced)
  * k/k_laser: [0.3, 3.0] (30 values)
  * Density: 30-250 nc (9 values near relativistic critical)
  * Thickness: 5, 8, 10, 15, 20, 30, 40, 50, 70, 100 nm
  * Incidence angle: 0°, 20°, 40°
  * Polarization: CP, LP
- Results: 419,400 unstable configurations (74%), max growth rate γ/ωp = 0.221

DATA FORMAT
-----------
CSV with headers, columns:
1. a0 - Normalized laser amplitude
2. ne_nc - Electron density in units of critical density
3. thickness_nm - Foil thickness in nanometers
4. k_over_klaser - Wavenumber normalized to laser wavenumber
5. incidence_angle - Laser incidence angle in degrees
6. polarization - 'CP' or 'LP'
7. growth_rate - RTI growth rate γ/ωp
8. k_cutoff - Cutoff wavenumber
9. similarity_number - Φ₃ parameter

USAGE
-----
Python example:
```python
import pandas as pd
data = pd.read_csv('enhanced_physical_sweep_results.csv')
unstable = data[data['growth_rate'] > 0]
print(f"Unstable configurations: {len(unstable)} ({100*len(unstable)/len(data):.1f}%)")
```

VALIDATION
----------
This dataset was used to validate the theoretical model against:
- 44 experimental data points from OMEGA, Nike, LULI2000, and NIF
- Achieved R² = 0.89 ± 0.05 in cross-validation
- Confirms leading-order universality across entire parameter space

CITATION
--------
If you use this data, please cite the main manuscript and include the data DOI when available.

Contact: sunilkgrao@gmail.com
