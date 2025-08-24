# RTI Validation Framework

This directory contains the validation framework for the RTI optimal control theory.

## Important Note

Due to repository size constraints, large datasets and installations have been removed:
- Athena++ data (2.9GB) - Available at: https://github.com/connor-mcclellan/rayleigh-taylor
- Virtual environments - Can be recreated with `pip install -r requirements.txt`
- Smilei PIC code - See installation instructions at: https://smileipic.github.io/Smilei/

## Key Components

### Scripts
- `analyze_real_rti_data.py` - Main analysis script for Athena++ data
- `comprehensive_real_analysis.py` - Comprehensive validation framework
- Various validation and analysis utilities

### Analysis Results
- `FINAL_REALISTIC_VALIDATION_REPORT.md` - Final validation summary
- `validation_results.json` - Validation metrics

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download Athena++ data (optional):
   ```bash
   git clone https://github.com/connor-mcclellan/rayleigh-taylor
   ```

## Usage

See individual script documentation for usage details.
