#!/bin/bash

# RTI Validation Setup Script
# Sets up all necessary repositories and dependencies

echo "=== RTI Validation Repository Setup ==="
echo "Starting at: $(date)"

# Set base directory
BASE_DIR="$(pwd)/rti_validation"
cd $BASE_DIR

# Create subdirectories
mkdir -p {repositories,warpx_setup,analysis_tools,simulation_outputs,zhou2017_data}

echo ">>> Step 1: Cloning RTI simulation repositories..."

# Clone RT1D for 1D simulations
if [ ! -d "repositories/RT1D" ]; then
    echo "Cloning RT1D repository..."
    git clone https://github.com/duffell/RT1D.git repositories/RT1D 2>/dev/null || echo "RT1D repo not accessible - will use alternative"
fi

# Clone compressible RTI CUDA solver
if [ ! -d "repositories/compressible-rayleigh-taylor-instability" ]; then
    echo "Cloning compressible RTI repository..."
    git clone https://github.com/ctjacobs/compressible-rayleigh-taylor-instability.git repositories/compressible-rayleigh-taylor-instability 2>/dev/null || echo "Compressible RTI repo not accessible - will use alternative"
fi

echo ">>> Step 2: Setting up analysis tools..."

# Clone PyDMD for Dynamic Mode Decomposition
if [ ! -d "analysis_tools/PyDMD" ]; then
    echo "Cloning PyDMD..."
    git clone https://github.com/PyDMD/PyDMD.git analysis_tools/PyDMD 2>/dev/null || echo "PyDMD will be installed via pip"
fi

# Clone bootstrap statistics tool
if [ ! -d "analysis_tools/bstrap" ]; then
    echo "Cloning bootstrap statistics..."
    git clone https://github.com/fpgdubost/bstrap.git analysis_tools/bstrap 2>/dev/null || echo "Bootstrap tools will be installed via pip"
fi

echo ">>> Step 3: Creating Python virtual environment..."

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

echo ">>> Step 4: Installing Python dependencies..."

# Install core dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib h5py netcdf4 pandas jupyter plotly
pip install pydmd tikzplotlib control qutip tabulate
pip install scikit-learn seaborn tqdm

echo ">>> Step 5: Checking system resources..."

# Check available memory
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
    echo "Total system memory: ${TOTAL_MEM} GB"
else
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "Total system memory: ${TOTAL_MEM} GB"
fi

echo ">>> Setup completed at: $(date)"
echo "==================================="
echo "Next steps:"
echo "1. Run data extraction scripts"
echo "2. Execute simulation validation"
echo "3. Generate analysis reports"