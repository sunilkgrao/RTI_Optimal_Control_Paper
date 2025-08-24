# RTI Optimal Control Paper Validation System

## Overview

This automated validation system verifies the theoretical predictions in the Rayleigh-Taylor Instability (RTI) optimal control paper. The system performs comprehensive numerical simulations and statistical analysis to validate key theorems including:

1. **Universal Collapse Theorem** - Cubic viscous scaling law validation
2. **Bang-Bang Control** - Optimal CP/LP switching strategy verification  
3. **Edge-of-Transparency** - Critical surface tracking for RTI suppression
4. **Growth Rate Predictions** - Quantitative validation against theory

## Quick Start

### One-Command Execution

```bash
cd rti_validation
python master_validation_executor.py
```

### Monitor Progress

```bash
tail -f rti_validation.log
```

### Check Results

```bash
cat validation_results.json | python -m json.tool
```

## System Requirements

- **Memory**: 64-128 GB RAM recommended
- **Storage**: ~50 GB free space
- **Python**: 3.8 or higher
- **OS**: Linux/macOS (Windows via WSL)
- **CPU**: Multi-core processor (8+ cores recommended)

## Installation

### 1. Basic Setup

```bash
# Clone or download the validation system
cd rti_validation

# Make scripts executable
chmod +x scripts/*.sh

# Run setup (installs dependencies)
bash scripts/setup_repositories.sh
```

### 2. Python Dependencies

The setup script automatically creates a virtual environment and installs:
- numpy, scipy, matplotlib
- h5py, netcdf4, pandas
- control, qutip
- pydmd (Dynamic Mode Decomposition)

## Usage

### Full Validation Pipeline

```bash
# Run complete validation (skips completed steps)
python master_validation_executor.py

# Force re-run all steps
python master_validation_executor.py --no-skip

# Run with parallel execution
python master_validation_executor.py --parallel
```

### Individual Validation Steps

```bash
# Run specific validation
python master_validation_executor.py --step universal_collapse
python master_validation_executor.py --step bang_bang_control
python master_validation_executor.py --step edge_transparency
```

### Available Steps

1. `setup_repositories` - Set up dependencies and repositories
2. `extract_data` - Extract and prepare validation data
3. `universal_collapse` - Validate universal collapse theorem
4. `bang_bang_control` - Validate bang-bang optimal control
5. `edge_transparency` - Validate edge-of-transparency tracking
6. `extract_growth_rates` - Extract growth rates from simulations
7. `statistical_validation` - Statistical analysis and report generation

## Output Files

### Main Results

- `validation_results.json` - Master validation results
- `rti_validation.log` - Detailed execution log

### Analysis Results

- `analysis/universal_collapse_validation.json` - Universal collapse results
- `analysis/bang_bang_validation.json` - Bang-bang control results
- `analysis/edge_transparency_validation.json` - Edge transparency results
- `analysis/extracted_growth_rates.csv` - Growth rate data

### Reports

- `reports/validation_report.tex` - LaTeX validation report
- `reports/validation_summary.json` - Machine-readable summary

### Visualizations

- `analysis/*.png` - Validation plots and figures
- `simulations/*.png` - Simulation visualizations

## Validation Criteria

### Success Metrics

- **Universal Collapse**: Variance ratio < 10% in collapsed coordinates
- **Bang-Bang Control**: Single switch optimal in >80% of cases
- **Edge Transparency**: Stable tracking in >70% of simulations
- **Growth Rates**: Mean relative error < 10%

### Statistical Confidence

- 95% confidence intervals via bootstrap (10,000 samples)
- Kolmogorov-Smirnov tests for distribution comparison
- R² > 0.9 for theoretical predictions

## Runtime Estimates

| Task | Estimated Time | Memory Usage |
|------|---------------|--------------|
| Setup | 5 minutes | 2 GB |
| Data Extraction | 1 minute | 1 GB |
| Universal Collapse | 10 minutes | 16-24 GB |
| Bang-Bang Control | 15 minutes | 8-16 GB |
| Edge Transparency | 10 minutes | 8-16 GB |
| Growth Rate Extraction | 5 minutes | 4-8 GB |
| Statistical Analysis | 2 minutes | 2-4 GB |
| **Total** | **~48 minutes** | **Peak: 24 GB** |

## Troubleshooting

### Memory Issues

If you encounter memory errors:

```bash
# Reduce resolution in validators
# Edit scripts/validate_universal_collapse.py
self.resolutions = [128, 256]  # Instead of [256, 512]
```

### Missing Dependencies

```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Specific Step Failures

```bash
# Check log for details
grep ERROR rti_validation.log

# Re-run failed step
python master_validation_executor.py --step <step_name> --no-skip
```

## Advanced Usage

### Custom Parameters

Edit validation parameters in individual scripts:

```python
# scripts/validate_universal_collapse.py
self.atwood_numbers = [0.1, 0.5, 0.9]  # Modify Atwood numbers
self.viscosities = np.logspace(-5, -3, 5)  # Modify viscosity range
```

### Parallel Execution

For systems with many cores:

```bash
# Use parallel validation (3 concurrent processes)
python master_validation_executor.py --parallel
```

### Generate LaTeX Report Only

```bash
# If validation is complete, regenerate report
python scripts/statistical_validation.py
```

## Expected Output

Upon successful completion:

```
VALIDATION SUMMARY
==================
Status: SUCCESS
Total time: 0:48:23
Steps completed: 7/7

✓ Completed steps:
  - setup_repositories
  - extract_data
  - universal_collapse
  - bang_bang_control
  - edge_transparency
  - extract_growth_rates
  - statistical_validation

THEORETICAL VALIDATION RESULTS
===============================
✓✓✓ ALL THEOREMS VALIDATED SUCCESSFULLY ✓✓✓
Theorems validated: Universal Collapse, Bang-Bang Control, Edge-of-Transparency
```

## Paper Integration

To insert validation results into your LaTeX paper:

1. Copy the generated table from `reports/validation_report.tex`
2. Insert into your paper after the Results section:

```latex
\input{validation_report}  % Or copy specific tables
```

## Citation

If you use this validation system, please cite:

```bibtex
@software{rti_validation_2024,
  title = {RTI Optimal Control Validation System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rti-validation}
}
```

## Support

For issues or questions:
- Check `rti_validation.log` for detailed error messages
- Review individual script outputs in their respective directories
- Ensure all dependencies are correctly installed

## License

This validation system is provided as-is for academic validation purposes.