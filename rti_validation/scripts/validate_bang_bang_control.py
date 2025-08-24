#!/usr/bin/env python3
"""
Bang-Bang Control Validator
Validates optimal control strategy for RTI suppression using CP/LP switching
"""

import numpy as np
import control as ct
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BangBangControl')

class BangBangControlValidator:
    def __init__(self, output_dir='../analysis'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Control bounds for bang-bang
        self.control_bounds = [-1.0, 1.0]  # Normalized: -1 = LP, +1 = CP
        
        # Physical parameters
        self.gravity = 9.81  # m/s^2
        self.wavelength = 0.01  # 1 cm perturbation
        self.wavenumber = 2 * np.pi / self.wavelength
        
        # Test parameters
        self.atwood_numbers = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.time_horizons = [0.1, 0.5, 1.0]  # seconds
        
        self.results = {}
        
    def create_rti_control_system(self, atwood):
        """Create linearized RTI control system"""
        
        # State vector: [amplitude, velocity]
        # Control input: acceleration (switching between CP and LP)
        
        # System matrices for dx/dt = Ax + Bu
        omega_squared = atwood * self.gravity * self.wavenumber
        
        A_matrix = np.array([
            [0, 1],
            [omega_squared, 0]
        ])
        
        B_matrix = np.array([[0], [1]])
        C_matrix = np.eye(2)
        D_matrix = np.zeros((2, 1))
        
        sys = ct.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)
        
        return sys, omega_squared
    
    def hamiltonian(self, state, costate, control, A, B):
        """Compute Hamiltonian for optimal control"""
        
        # H = λ^T * (Ax + Bu)
        dx = A @ state + B.flatten() * control
        H = np.dot(costate, dx)
        
        return H
    
    def switching_function(self, costate, B):
        """Compute switching function σ(t) = λ^T * B"""
        
        return np.dot(costate, B.flatten())
    
    def optimal_control_law(self, sigma):
        """Bang-bang control law based on switching function"""
        
        if sigma > 0:
            return self.control_bounds[1]  # CP (positive acceleration)
        elif sigma < 0:
            return self.control_bounds[0]  # LP (negative acceleration)
        else:
            return 0  # Singular arc (rare in practice)
    
    def solve_two_point_bvp(self, sys, T_final, x0):
        """Solve two-point boundary value problem for optimal control"""
        
        A = sys.A
        B = sys.B
        
        def dynamics_with_costate(t, y):
            """Combined state and costate dynamics"""
            n = len(x0)
            state = y[:n]
            costate = y[n:]
            
            # Switching function
            sigma = self.switching_function(costate, B)
            
            # Optimal control
            u = self.optimal_control_law(sigma)
            
            # State dynamics
            dx = A @ state + B.flatten() * u
            
            # Costate dynamics (adjoint equation)
            # dλ/dt = -A^T * λ
            dlambda = -A.T @ costate
            
            return np.concatenate([dx, dlambda])
        
        # Initial guess for final costate (shooting method)
        def shooting_objective(lambda0):
            """Minimize final amplitude"""
            
            # Initial conditions: state + costate
            y0 = np.concatenate([x0, lambda0])
            
            # Solve ODE
            sol = solve_ivp(dynamics_with_costate, [0, T_final], y0,
                           method='RK45', dense_output=True, max_step=0.001)
            
            # Final state
            final_state = sol.y[:len(x0), -1]
            
            # Objective: minimize final amplitude squared
            return final_state[0]**2
        
        # Find optimal initial costate
        result = differential_evolution(shooting_objective, 
                                       bounds=[(-10, 10), (-10, 10)],
                                       seed=42, maxiter=100)
        
        optimal_lambda0 = result.x
        
        # Solve with optimal costate
        y0 = np.concatenate([x0, optimal_lambda0])
        sol = solve_ivp(dynamics_with_costate, [0, T_final], y0,
                       method='RK45', t_eval=np.linspace(0, T_final, 1000))
        
        # Extract control history
        controls = []
        switching_times = []
        last_control = None
        
        for i, t in enumerate(sol.t):
            costate = sol.y[len(x0):, i]
            sigma = self.switching_function(costate, B)
            u = self.optimal_control_law(sigma)
            controls.append(u)
            
            # Detect switches
            if last_control is not None and u != last_control:
                switching_times.append(t)
            last_control = u
        
        return {
            'times': sol.t,
            'states': sol.y[:len(x0), :],
            'costates': sol.y[len(x0):, :],
            'controls': np.array(controls),
            'switching_times': switching_times,
            'final_amplitude': sol.y[0, -1]
        }
    
    def verify_single_switch_optimality(self, sys, T_final, x0):
        """Verify that single CP->LP switch is optimal"""
        
        # Test single switch strategy
        def single_switch_cost(t_switch):
            """Cost function for single switch at time t_switch"""
            
            if t_switch <= 0 or t_switch >= T_final:
                return np.inf
            
            # Phase 1: CP control (u = +1)
            t1 = np.linspace(0, t_switch, 500).flatten()
            response1 = ct.forced_response(
                sys, t1, U=self.control_bounds[1]*np.ones(len(t1)), X0=x0
            )
            
            # Phase 2: LP control (u = -1)
            t2 = np.linspace(t_switch, T_final, 500).flatten()
            x_switch = response1.states[:, -1]
            response2 = ct.forced_response(
                sys, t2, U=self.control_bounds[0]*np.ones(len(t2)), X0=x_switch
            )
            
            # Final amplitude
            return abs(response2.states[0, -1])
        
        # Find optimal single switch time
        result = minimize(single_switch_cost, T_final/2,
                         bounds=[(0.01, T_final-0.01)],
                         method='L-BFGS-B')
        
        optimal_t_switch = result.x[0]
        single_switch_amplitude = result.fun
        
        # Test multiple switch strategies
        multi_switch_amplitudes = []
        
        for n_switches in [2, 3, 5]:
            amplitude = self.test_n_switches(sys, T_final, x0, n_switches)
            multi_switch_amplitudes.append(amplitude)
        
        # Compare strategies
        return {
            'single_switch_time': optimal_t_switch,
            'single_switch_amplitude': single_switch_amplitude,
            'multi_switch_amplitudes': multi_switch_amplitudes,
            'single_optimal': single_switch_amplitude <= min(multi_switch_amplitudes),
            'improvement_ratio': min(multi_switch_amplitudes) / single_switch_amplitude if single_switch_amplitude > 0 else np.inf
        }
    
    def test_n_switches(self, sys, T_final, x0, n_switches):
        """Test control strategy with n switches"""
        
        if n_switches == 0:
            # No control
            t = np.linspace(0, T_final, 1000).flatten()
            response = ct.forced_response(sys, t, U=np.zeros(len(t)), X0=x0)
            return abs(response.states[0, -1])
        
        # Equal time intervals
        switch_times = np.linspace(0, T_final, n_switches + 1)
        
        # Alternate between CP and LP
        x_current = x0
        
        for i in range(n_switches):
            t_segment = np.linspace(switch_times[i], switch_times[i+1], 100).flatten()
            
            # Alternate control
            if i % 2 == 0:
                u = self.control_bounds[1]  # CP
            else:
                u = self.control_bounds[0]  # LP
            
            response = ct.forced_response(
                sys, t_segment, U=u*np.ones(len(t_segment)), X0=x_current
            )
            x_current = response.states[:, -1]
        
        return abs(x_current[0])
    
    def plot_control_results(self, results_dict, atwood):
        """Create visualization of bang-bang control results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: State trajectory with control
        ax1 = axes[0, 0]
        
        for T_final, results in results_dict.items():
            bvp_sol = results['bvp_solution']
            t = bvp_sol['times']
            amplitude = bvp_sol['states'][0, :]
            
            ax1.plot(t, amplitude, label=f'T={T_final}s', linewidth=2)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (m)')
        ax1.set_title(f'RTI Amplitude Evolution (A={atwood})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Control signal
        ax2 = axes[0, 1]
        
        for T_final, results in results_dict.items():
            bvp_sol = results['bvp_solution']
            t = bvp_sol['times']
            u = bvp_sol['controls']
            
            ax2.step(t, u, where='post', label=f'T={T_final}s', linewidth=2)
            
            # Mark switching times
            for t_switch in bvp_sol['switching_times']:
                ax2.axvline(x=t_switch, color='red', linestyle='--', alpha=0.3)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control (CP=+1, LP=-1)')
        ax2.set_title('Bang-Bang Control Signal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Switching function
        ax3 = axes[1, 0]
        
        for T_final, results in results_dict.items():
            bvp_sol = results['bvp_solution']
            t = bvp_sol['times']
            costates = bvp_sol['costates']
            B = np.array([[0], [1]])
            
            sigma = [self.switching_function(costates[:, i], B) for i in range(len(t))]
            
            ax3.plot(t, sigma, label=f'T={T_final}s', linewidth=2)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Switching Function σ(t)')
        ax3.set_title('Switching Function Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Comparison of strategies
        ax4 = axes[1, 1]
        
        strategies = ['No Control', 'Single Switch', '2 Switches', '3 Switches', '5 Switches']
        amplitudes = []
        
        # Get representative results (T=1.0s)
        if 1.0 in results_dict:
            rep_results = results_dict[1.0]
            amplitudes = [
                self.test_n_switches(rep_results['system'], 1.0, [0.001, 0], 0),
                rep_results['single_switch']['single_switch_amplitude']
            ] + rep_results['single_switch']['multi_switch_amplitudes']
        
        if amplitudes:
            bars = ax4.bar(strategies[:len(amplitudes)], amplitudes, alpha=0.7)
            bars[1].set_color('green')  # Highlight single switch
            
            ax4.set_ylabel('Final Amplitude (m)')
            ax4.set_title('Control Strategy Comparison')
            ax4.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f'bang_bang_control_A{atwood:.1f}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {output_file}")
        
        return output_file
    
    def run_validation(self):
        """Execute complete bang-bang control validation"""
        
        logger.info("Starting Bang-Bang Control validation...")
        print("\n=== Bang-Bang Control Validation ===")
        print(f"Testing {len(self.atwood_numbers)} Atwood numbers")
        print(f"Testing {len(self.time_horizons)} time horizons")
        
        validation_results = {}
        
        for atwood in self.atwood_numbers:
            print(f"\nProcessing Atwood number: {atwood}")
            
            # Create control system
            sys, omega_squared = self.create_rti_control_system(atwood)
            
            # Initial condition: small perturbation
            x0 = np.array([0.001, 0])  # 1mm initial amplitude, zero velocity
            
            atwood_results = {}
            
            for T_final in self.time_horizons:
                print(f"  Time horizon: {T_final}s")
                
                # Solve optimal control problem
                bvp_solution = self.solve_two_point_bvp(sys, T_final, x0)
                
                # Verify single switch optimality
                single_switch_test = self.verify_single_switch_optimality(sys, T_final, x0)
                
                atwood_results[T_final] = {
                    'system': sys,
                    'bvp_solution': bvp_solution,
                    'single_switch': single_switch_test,
                    'n_switches': len(bvp_solution['switching_times'])
                }
                
                print(f"    ✓ Switches: {len(bvp_solution['switching_times'])}")
                print(f"    ✓ Single optimal: {single_switch_test['single_optimal']}")
            
            # Create visualization
            plot_file = self.plot_control_results(atwood_results, atwood)
            
            validation_results[f'A_{atwood}'] = {
                'atwood': atwood,
                'results': {
                    T: {
                        'n_switches': res['n_switches'],
                        'final_amplitude': res['bvp_solution']['final_amplitude'],
                        'single_optimal': res['single_switch']['single_optimal'],
                        'optimal_switch_time': res['single_switch']['single_switch_time']
                    }
                    for T, res in atwood_results.items()
                },
                'plot_file': plot_file
            }
        
        # Analyze results
        single_switch_count = 0
        total_tests = 0
        
        for atwood_key, atwood_data in validation_results.items():
            for T, res in atwood_data['results'].items():
                total_tests += 1
                if res['n_switches'] == 1 and res['single_optimal']:
                    single_switch_count += 1
        
        # Save validation summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'theorem': 'Bang-Bang Optimal Control',
            'atwood_numbers': self.atwood_numbers,
            'time_horizons': self.time_horizons,
            'total_tests': total_tests,
            'single_switch_optimal_count': single_switch_count,
            'validation_passed': single_switch_count / total_tests > 0.8,
            'detailed_results': validation_results
        }
        
        output_file = os.path.join(self.output_dir, 'bang_bang_validation.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_summary = convert_to_serializable(summary)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\n✓ Validation complete!")
        print(f"Results saved to: {output_file}")
        
        return summary

def main():
    """Main execution"""
    validator = BangBangControlValidator()
    results = validator.run_validation()
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Theorem validated: {results['validation_passed']}")
    print(f"Single switch optimal: {results['single_switch_optimal_count']}/{results['total_tests']} cases")
    
    if results['validation_passed']:
        print("✓ Bang-bang control with single CP->LP switch confirmed optimal!")
    else:
        print("✗ Bang-bang optimality not fully confirmed - review results")

if __name__ == "__main__":
    main()