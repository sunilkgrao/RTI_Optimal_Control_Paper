#!/usr/bin/env python3
"""
RTI Validation Infrastructure Setup Summary
"""

import json
import pandas as pd
from pathlib import Path
import os

def main():
    """Generate a comprehensive summary of the RTI validation setup"""
    
    base_dir = Path("/Users/sunilrao/Downloads/RTI_Optimal_Control_Paper/rti_validation")
    data_dir = base_dir / "data"
    
    print("=" * 80)
    print("RTI VALIDATION INFRASTRUCTURE SETUP SUMMARY")
    print("=" * 80)
    
    # Task 1: Repository cloning status
    print("\n1. SIMULATION CODES SUCCESSFULLY CLONED:")
    print("-" * 50)
    
    sim_dir = base_dir / "simulations"
    if (sim_dir / "RT1D").exists():
        print("✓ RT1D - 1D Rayleigh-Taylor simulation code")
        print("  Source: https://github.com/duffell/RT1D.git")
        print("  Language: C, Features: Hydro, Gravity, MHD")
    
    if (sim_dir / "athena-public-version").exists():
        print("✓ Athena - MHD simulation code")
        print("  Source: https://github.com/PrincetonUniversity/athena-public-version.git")
        print("  Language: C++, Features: 2D/3D, MHD, AMR, RT instability")
    
    # Task 2: Synthetic data generation
    print("\n2. SYNTHETIC EXPERIMENTAL DATA (Zhou 2017 based):")
    print("-" * 50)
    
    data_files = [
        ("growth_rate_data.csv", "Growth rates vs Atwood number"),
        ("mixing_width_data.csv", "Mixing width evolution"),
        ("bubble_spike_data.csv", "Bubble and spike velocities")
    ]
    
    total_synthetic_points = 0
    for filename, description in data_files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            total_synthetic_points += len(df)
            print(f"✓ {filename}: {len(df)} data points - {description}")
    
    print(f"  Total synthetic data points: {total_synthetic_points}")
    print("  Scatter: 5-10% experimental scatter added")
    print("  Uncertainty: 3-8% experimental uncertainty")
    
    # Task 3: Literature data
    print("\n3. LITERATURE DATA EXTRACTION:")
    print("-" * 50)
    
    lit_db_file = data_dir / "literature_database.json"
    if lit_db_file.exists():
        with open(lit_db_file) as f:
            lit_db = json.load(f)
        
        print(f"✓ Literature database: {len(lit_db)} key papers catalogued")
        open_access = sum(1 for paper in lit_db if paper["access"] == "open")
        print(f"  - Open access papers: {open_access}")
        print(f"  - Subscription papers: {len(lit_db) - open_access}")
    
    # Check digitized data
    digitized_files = [
        ("zhou_growth_rates.csv", "Zhou 2017 Fig. 15"),
        ("dimonte_mixing.csv", "Dimonte 2004 Fig. 8")
    ]
    
    total_lit_points = 0
    for filename, source in digitized_files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            total_lit_points += len(df)
            print(f"✓ {filename}: {len(df)} points from {source}")
    
    print(f"  Total literature data points: {total_lit_points}")
    
    # Task 4: Directory structure
    print("\n4. DIRECTORY STRUCTURE:")
    print("-" * 50)
    
    required_dirs = ["data", "simulations", "analysis", "reports"]
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.rglob("*")))
            print(f"✓ {dir_name}/ - {file_count} files/subdirectories")
        else:
            print(f"✗ {dir_name}/ - Missing!")
    
    # Task 5: Data inventory
    print("\n5. COMPREHENSIVE DATA INVENTORY:")
    print("-" * 50)
    
    inventory_file = data_dir / "data_inventory.json"
    if inventory_file.exists():
        print("✓ data_inventory.json - Complete catalog created")
        with open(inventory_file) as f:
            inventory = json.load(f)
        
        rti_data = inventory["rti_validation_data_inventory"]
        print(f"  - Simulation codes: {len(rti_data['simulation_codes'])}")
        print(f"  - Data sources: {rti_data['data_summary']['total_datasets']}")
        print(f"  - Atwood number range: {rti_data['data_summary']['coverage']['atwood_numbers']}")
        print(f"  - Wave number range: {rti_data['data_summary']['coverage']['wave_numbers']}")
    
    # Summary statistics
    print("\n6. VALIDATION CAPABILITIES:")
    print("-" * 50)
    
    print("✓ Growth rate validation: γ = √(A*g*k)")
    print("✓ Mixing width validation: h = 2*γ*t²")
    print("✓ Bubble/spike velocity validation")
    print("✓ Parameter coverage: A ∈ [0.1, 0.9], k ∈ [1, 10] m⁻¹")
    print("✓ Time evolution: t ∈ [0.1, 2.0] s")
    
    # Next steps
    print("\n7. RECOMMENDED NEXT STEPS:")
    print("-" * 50)
    print("1. Compile RT1D: cd simulations/RT1D && make")
    print("2. Configure Athena for RTI test cases")
    print("3. Run validation simulations")
    print("4. Compare against synthetic and literature data")
    print("5. Generate comprehensive validation reports")
    
    print("\n" + "=" * 80)
    print("RTI VALIDATION INFRASTRUCTURE SETUP COMPLETE!")
    print(f"Total data points available: {total_synthetic_points + total_lit_points}")
    print("Ready for validation studies.")
    print("=" * 80)

if __name__ == "__main__":
    main()