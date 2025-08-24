#!/usr/bin/env python3
"""
SMILEI Input Deck: Thin-Foil RTI with Single-Mode Seed
Implements the exact PIC setup described in the validation plan
"""

from math import pi, sin, cos, exp, sqrt
import numpy as np

# =================== USER PARAMETERS ===================
# Laser parameters
lambda0_um = 0.8      # Laser wavelength [μm]
a0 = 15.0             # Normalized vector potential (scan parameter)
pol = "CP"            # Polarization: "CP" or "LP"
pulse_duration_fs = 40.0  # FWHM pulse duration [fs]

# Target parameters  
ne_over_nc = 100.0    # Electron density [units of nc]
thickness_nm = 50.0   # Foil thickness [nm]
target_material = "DLC"  # Diamond-like carbon surrogate

# Simulation domain
Lx_um = 20.0         # Domain length [μm]
Ly_um = 20.0         # Domain width [μm]
cells_per_lambda = 40 # Spatial resolution
ppc = 100            # Particles per cell

# RTI seeding
k_seed_fraction = 0.6  # k_seed / k_thermal (ensure kd < 0.3)
ripple_amplitude_fraction = 0.01  # η0 / thickness

# Physical parameters
Te_eV = 10.0         # Electron temperature [eV]
Ti_eV = 1.0          # Ion temperature [eV]

# ================= DERIVED PARAMETERS ==================
# Unit conversions and constants
c = 299792458.0       # Speed of light [m/s]
lambda0 = lambda0_um * 1e-6  # [m]
w0 = 2.0 * pi * c / lambda0  # Laser frequency [rad/s]
k0 = 2.0 * pi / lambda0      # Laser wavenumber [1/m]
nc = 1.1e21 / (lambda0_um**2)  # Critical density [cm⁻³]

# Spatial discretization
dx = lambda0 / cells_per_lambda  # Grid spacing [m]
dy = dx
nx = int(Lx_um * 1e-6 / dx)
ny = int(Ly_um * 1e-6 / dy)

# Temporal parameters
dt = 0.95 * dx / c   # Timestep [s]
sim_time = 100.0 * (2*pi/w0)  # Total simulation time [s]

# Target parameters
thickness = thickness_nm * 1e-9  # [m]
ne = ne_over_nc * nc * 1e6  # [m⁻³] (convert cm⁻³ to m⁻³)
x_target = 10.0 * lambda0   # Target position [m]

# RTI seeding parameters
k_thermal = sqrt(ne * 1.602e-19**2 / (8.854e-12 * 1.381e-23 * (Te_eV * 11604.5)))
k_seed = k_seed_fraction * k_thermal
lambda_seed = 2*pi / k_seed
ripple_amplitude = ripple_amplitude_fraction * thickness

# Validation check
kd_ratio = k_seed * thickness
if kd_ratio > 0.3:
    print(f"WARNING: kd = {kd_ratio:.3f} > 0.3, may need thickness correction")

print(f"""
SMILEI PIC SIMULATION SETUP
===========================
Laser: λ₀ = {lambda0_um} μm, a₀ = {a0}, {pol} polarization
Target: {thickness_nm} nm {target_material}, nₑ = {ne_over_nc:.0f} nc
Domain: {Lx_um} × {Ly_um} μm², {nx} × {ny} cells
RTI seed: λ = {lambda_seed*1e6:.2f} μm, kd = {kd_ratio:.3f}
""")

# ================= SMILEI CONFIGURATION =================

Main(
    geometry = "2Dcartesian",
    interpolation_order = 2,
    timestep = dt,
    simulation_time = sim_time,
    cell_length = [dx, dy],
    grid_length = [nx*dx, ny*dy],
    number_of_patches = [8, 8],
    EM_boundary_conditions = [
        ["silver-muller", "silver-muller"],  # x boundaries
        ["periodic", "periodic"]             # y boundaries (for single mode)
    ],
    random_seed = 42,
    print_every = 100
)

# Laser pulse definition
LaserGaussian2D(
    box_side = "xmin",
    a0 = a0,
    omega = w0,
    waist = 5.0 * lambda0,  # Focused waist
    focus = [x_target - 2*lambda0, ny*dy/2],  # Focus slightly before target
    incidence_angle = 0.0,
    time_envelope = tgaussian(
        center = 50.0 * (2*pi/w0),  # Pulse center
        fwhm = pulse_duration_fs * 1e-15 / (2*pi/w0) * w0  # Convert fs to laser periods
    ),
    space_time_profile = []
)

# Polarization control
if pol == "CP":
    # Circular polarization
    LaserGaussian2D(
        box_side = "xmin",
        a0 = a0,
        omega = w0,
        waist = 5.0 * lambda0,
        focus = [x_target - 2*lambda0, ny*dy/2],
        incidence_angle = 0.0,
        polarization_phi = pi/2,  # 90° phase shift for circular
        time_envelope = tgaussian(
            center = 50.0 * (2*pi/w0),
            fwhm = pulse_duration_fs * 1e-15 / (2*pi/w0) * w0
        )
    )

# Define surface modulation function for RTI seeding
def surface_modulation(y):
    """Sinusoidal surface ripple for single-mode RTI seeding"""
    return ripple_amplitude * sin(k_seed * y)

# Target: Ultrathin foil with sinusoidal surface modulation
Species(
    name = "electrons",
    position_initialization = "regular",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = ppc,
    mass = 1.0,  # Normalized units
    charge = -1.0,
    number_density = trapezoidal(
        xvacuum = x_target - thickness/2,
        xplateau = thickness,
        xslope1 = 0.0,
        xslope2 = 0.0,
        density = ne
    ),
    temperature = [Te_eV],
    boundary_conditions = [
        ["remove", "remove"],
        ["periodic", "periodic"]
    ],
    # Apply surface modulation
    position_initialization_along_x = lambda x, y, z: x + surface_modulation(y)
)

# Ion species (Carbon-like)
Species(
    name = "ions",
    position_initialization = "regular", 
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = ppc,
    mass = 12.0 * 1836.15,  # C12 mass in electron masses
    charge = +6.0,  # Fully ionized carbon
    number_density = trapezoidal(
        xvacuum = x_target - thickness/2,
        xplateau = thickness,
        xslope1 = 0.0,
        xslope2 = 0.0,
        density = ne / 6.0  # Charge neutrality
    ),
    temperature = [Ti_eV],
    boundary_conditions = [
        ["remove", "remove"], 
        ["periodic", "periodic"]
    ],
    # Apply same surface modulation
    position_initialization_along_x = lambda x, y, z: x + surface_modulation(y)
)

# ================= DIAGNOSTICS ===================

# Scalar diagnostics (global quantities)
DiagScalar(
    every = 50,
    vars = ["Uelm", "Ukin", "Utot", "Uexp", "Ubal",
           "Rho_electrons", "Rho_ions"]
)

# Field diagnostics (electromagnetic fields)
DiagFields(
    every = 100,
    fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz",
             "Jx", "Jy", "Jz", "Rho_electrons", "Rho_ions"],
    subgrid = [2, 2]  # Reduce output size
)

# Probe diagnostics for interface tracking
# This samples the electron density along the target surface
DiagProbe(
    every = 10,
    origin = [x_target, 0.0],
    corners = [
        [x_target, ny*dy]
    ],
    number = [ny//4],  # Sample points along y
    fields = ["Rho_electrons", "Rho_ions", "Ex", "Ey"]
)

# Particle diagnostics (sample particles for analysis)
DiagParticleBinning(
    deposited_quantity = "weight",
    every = 200,
    species = ["electrons"],
    axes = [
        ["x", x_target - 2*thickness, x_target + 2*thickness, 100],
        ["y", 0.0, ny*dy, 50]
    ]
)

DiagParticleBinning(
    deposited_quantity = "weight", 
    every = 200,
    species = ["ions"],
    axes = [
        ["x", x_target - 2*thickness, x_target + 2*thickness, 100],
        ["y", 0.0, ny*dy, 50]
    ]
)

# Track particles for detailed analysis
DiagTrackParticles(
    species = "electrons",
    every = 50,
    flush_every = 500,
    filter = lambda particles: (
        (particles.x > x_target - thickness) & 
        (particles.x < x_target + thickness)
    ),
    attributes = ["x", "y", "px", "py", "pz", "w"]
)

print(f"""
SIMULATION READY TO RUN
======================
Command: ./smilei thinfoil_rti_smilei.py
Expected runtime: ~2-4 hours on M3 MacBook (8 cores)
Output: ./thinfoil_rti_smilei/
Key files:
  - Fields*.h5: EM fields and densities
  - Probes*.h5: Interface tracking data
  - ParticleBinning*.h5: Particle distributions
""")

# ================= ANALYSIS NOTES =================
"""
POST-PROCESSING ANALYSIS:
========================

1. Interface Tracking:
   - Use Probes*.h5 to extract ρ_e(x,y,t) 
   - Find interface position: x_interface(y,t) = contour(ρ_e = 0.5*ne)
   - Extract perturbation: η(y,t) = x_interface(y,t) - <x_interface>

2. Growth Rate Extraction:
   - FFT: η_k(t) = FFT_y[η(y,t)]
   - Focus on k = k_seed mode
   - Fit: ln|η_k(t)| = γt + const in linear window
   - Use RANSAC for robustness against noise

3. LP/CP Comparison:
   - Run identical setup with pol="LP" 
   - Extract γ_LP and γ_CP at same k_seed
   - Compute r_γ = γ_CP / γ_LP
   - Measure a_max from Fields*.h5: a_max = max(sqrt(Ey² + Ez²))
   - Compute r_a = a_max,CP / a_max,LP

4. Universal Collapse Validation:
   - Extract effective viscosity: ν_eff from γ(k) fitting
   - Test cubic scaling: h(t) ∝ (ν_eff * t³)^(1/3)
   - Compare with theoretical prediction

5. Table II Calibration:
   - Use closure relations (eqs. 20-22) with extracted ν_eff
   - Fit C_QM, C_IF, α, τ_B parameters
   - Propagate uncertainties via Jacobian

This single PIC run provides all data needed for:
- Fig. 1 overlay point (one anchor)
- LP/CP efficacy measurement (κ parameter)  
- Table II parameter calibration
- Universal collapse validation
"""