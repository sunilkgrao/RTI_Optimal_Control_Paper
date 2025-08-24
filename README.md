# Optimal Control and Leading-Order Thin-Foil Universality for RTI-Limited Radiation-Pressure Acceleration

**Author:** Sunil Rao  
**Email:** sunilkgrao@gmail.com  
**ORCID:** 0009-0009-3337-5726

## Directory Structure

### `/manuscript/`
Main paper files:
- `PhysicsPropulsionMod.tex` - Main LaTeX source
- `PhysicsPropulsionMod.pdf` - Compiled PDF
- `added_refs.bib` - Bibliography
- `table_calibration_template.tex` - Calibration table
- `validation_table_final.tex` - Validation results table

### `/figures/`
All figures used in the manuscript:
- `fig_universality_overlay.pdf` - Figure 1: Universal collapse with experimental data
- `fig_spectrum_validation.pdf` - Full spectrum validation
- `rti_validation_comprehensive_final.pdf` - Comprehensive validation figure
- `omega_parameter_coverage.pdf` - 567k parameter space coverage
- Additional supporting figures

### `/data/`
Parameter sweep data and analysis scripts:
- `enhanced_physical_sweep_results.csv` - Complete 567,000 parameter dataset
- `enhanced_physical_sweep_results.csv.gz` - Compressed version
- `run_enhanced_physical_sweep.py` - Parameter sweep generation code
- `create_enhanced_validation_summary.py` - Visualization code
- `sample_data_analysis.py` - Example analysis showing how to verify claims

### `/supplementary/`
Interactive tools and advanced analysis:
- `RTI_Interactive_Validation.ipynb` - Jupyter notebook for interactive exploration
- `advanced_ml_rti.py` - Machine learning models for validation
- `DATA_README.txt` - Detailed data documentation

### `/submission/`
Ready-to-submit packages:
- `PhysRevE_Submission_Rao.zip` - Main submission package
- `PhysRevE_SupplementaryMaterials_Rao.zip` - Supplementary data package
- `cover_letter_PR.txt` - Cover letter to editor
- `README_submission.txt` - Submission instructions
- `PhysRevE_Submission_Checklist.md` - Pre-submission checklist

## Key Results

- **567,000** physically-constrained parameter combinations tested
- **419,400** unstable configurations identified (74%)
- **R² = 0.89 ± 0.05** validation against 44 experimental points
- Maximum growth rate: γ/ωₚ = 0.221
- Single-switch CP→LP optimal control proven

## Compilation Instructions

From the `/manuscript/` directory:
```bash
pdflatex PhysicsPropulsionMod.tex
bibtex PhysicsPropulsionMod
pdflatex PhysicsPropulsionMod.tex
pdflatex PhysicsPropulsionMod.tex
```

## Citation

If you use this work, please cite:
```
Rao, S. "Optimal Control and Leading-Order Thin-Foil Universality for 
RTI-Limited Radiation-Pressure Acceleration." Physical Review E (submitted).
```

## Data Availability

All data supporting the validation claims are included in `/data/`. 
The interactive notebook allows real-time exploration of the parameter space.
