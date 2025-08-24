#!/usr/bin/env python3
"""
Master RTI Validation Executor
Orchestrates the complete validation pipeline for RTI control theorems
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime, timedelta
import traceback
import argparse
import multiprocessing

# Add scripts directory to path
sys.path.append('scripts')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rti_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MasterValidator')

class MasterRTIValidator:
    def __init__(self, skip_completed=True, parallel=False):
        self.skip_completed = skip_completed
        self.parallel = parallel
        self.start_time = datetime.now()
        
        # Validation pipeline steps
        self.validation_steps = [
            {
                'name': 'setup_repositories',
                'script': 'scripts/setup_repositories.sh',
                'type': 'bash',
                'description': 'Setting up repositories and dependencies',
                'estimated_time': 300  # seconds
            },
            {
                'name': 'extract_data',
                'script': 'scripts/create_data_extraction_script.py',
                'type': 'python',
                'description': 'Extracting and preparing validation data',
                'estimated_time': 60
            },
            {
                'name': 'universal_collapse',
                'script': 'scripts/validate_universal_collapse.py',
                'type': 'python',
                'description': 'Validating Universal Collapse theorem',
                'estimated_time': 600
            },
            {
                'name': 'bang_bang_control',
                'script': 'scripts/validate_bang_bang_control.py',
                'type': 'python',
                'description': 'Validating Bang-Bang optimal control',
                'estimated_time': 900
            },
            {
                'name': 'edge_transparency',
                'script': 'scripts/validate_edge_transparency.py',
                'type': 'python',
                'description': 'Validating Edge-of-Transparency tracking',
                'estimated_time': 600
            },
            {
                'name': 'extract_growth_rates',
                'script': 'scripts/extract_growth_rates.py',
                'type': 'python',
                'description': 'Extracting growth rates from simulations',
                'estimated_time': 300
            },
            {
                'name': 'statistical_validation',
                'script': 'scripts/statistical_validation.py',
                'type': 'python',
                'description': 'Performing statistical validation and generating reports',
                'estimated_time': 120
            }
        ]
        
        self.results = {
            'start_time': self.start_time.isoformat(),
            'system_info': self.get_system_info(),
            'steps_completed': [],
            'steps_failed': [],
            'validation_results': {}
        }
        
    def get_system_info(self):
        """Get system information"""
        
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': multiprocessing.cpu_count()
        }
        
        # Get memory info
        try:
            if sys.platform == 'darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True)
                memory_gb = int(result.stdout.strip()) / (1024**3)
                info['memory_gb'] = memory_gb
            else:  # Linux
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal'):
                            memory_kb = int(line.split()[1])
                            info['memory_gb'] = memory_kb / (1024**2)
                            break
        except:
            info['memory_gb'] = 'Unknown'
        
        return info
    
    def check_step_completed(self, step_name):
        """Check if a step has already been completed"""
        
        completion_markers = {
            'setup_repositories': 'venv',
            'extract_data': 'data/zhou2017_growth_rates.csv',
            'universal_collapse': 'simulations/universal_collapse_validation.json',
            'bang_bang_control': 'analysis/bang_bang_validation.json',
            'edge_transparency': 'analysis/edge_transparency_validation.json',
            'extract_growth_rates': 'analysis/extracted_growth_rates.csv',
            'statistical_validation': 'reports/validation_report.tex'
        }
        
        marker = completion_markers.get(step_name)
        if marker and os.path.exists(marker):
            return True
        
        return False
    
    def execute_step(self, step):
        """Execute a single validation step"""
        
        step_name = step['name']
        script_path = step['script']
        step_type = step['type']
        
        logger.info(f"{'='*60}")
        logger.info(f"Step: {step['description']}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Estimated time: {step['estimated_time']}s")
        
        # Check if already completed
        if self.skip_completed and self.check_step_completed(step_name):
            logger.info(f"✓ Step already completed, skipping...")
            self.results['steps_completed'].append(step_name)
            return True
        
        # Execute the step
        start_time = time.time()
        
        try:
            if step_type == 'bash':
                # Execute bash script
                result = subprocess.run(['bash', script_path],
                                      capture_output=True, text=True,
                                      timeout=step['estimated_time'] * 3)
                
                if result.returncode != 0:
                    logger.error(f"Bash script failed: {result.stderr}")
                    return False
                    
            elif step_type == 'python':
                # Execute Python script
                result = subprocess.run([sys.executable, script_path],
                                      capture_output=True, text=True,
                                      timeout=step['estimated_time'] * 3)
                
                if result.returncode != 0:
                    logger.error(f"Python script failed: {result.stderr}")
                    return False
            
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Step completed in {elapsed_time:.1f}s")
            
            self.results['steps_completed'].append(step_name)
            self.results['validation_results'][step_name] = {
                'status': 'completed',
                'elapsed_time': elapsed_time
            }
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Step timed out after {step['estimated_time']*3}s")
            self.results['steps_failed'].append(step_name)
            return False
            
        except Exception as e:
            logger.error(f"Step failed with error: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['steps_failed'].append(step_name)
            return False
    
    def execute_parallel_validation(self):
        """Execute validation steps that can run in parallel"""
        
        # Steps that can run in parallel (after data extraction)
        parallel_steps = [
            'universal_collapse',
            'bang_bang_control',
            'edge_transparency'
        ]
        
        logger.info("Executing parallel validation steps...")
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            for step in self.validation_steps:
                if step['name'] in parallel_steps:
                    future = executor.submit(self.execute_step, step)
                    futures[future] = step['name']
            
            for future in as_completed(futures):
                step_name = futures[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(f"✓ Parallel step {step_name} completed")
                    else:
                        logger.error(f"✗ Parallel step {step_name} failed")
                except Exception as e:
                    logger.error(f"Parallel step {step_name} crashed: {str(e)}")
    
    def execute_full_validation(self):
        """Execute the complete validation pipeline"""
        
        logger.info("="*70)
        logger.info("RTI VALIDATION PIPELINE STARTING")
        logger.info("="*70)
        logger.info(f"System: {self.results['system_info']}")
        logger.info(f"Skip completed: {self.skip_completed}")
        logger.info(f"Parallel execution: {self.parallel}")
        
        # Sequential steps that must run first
        sequential_first = ['setup_repositories', 'extract_data']
        
        # Execute initial sequential steps
        for step in self.validation_steps:
            if step['name'] in sequential_first:
                success = self.execute_step(step)
                if not success and step['name'] == 'setup_repositories':
                    logger.error("Repository setup failed - cannot continue")
                    self.results['status'] = 'FAILED'
                    return self.results
        
        # Execute validation steps (parallel or sequential)
        if self.parallel:
            self.execute_parallel_validation()
        else:
            for step in self.validation_steps:
                if step['name'] not in sequential_first and \
                   step['name'] not in ['extract_growth_rates', 'statistical_validation']:
                    success = self.execute_step(step)
                    if not success:
                        logger.warning(f"Step {step['name']} failed, continuing...")
        
        # Final sequential steps
        sequential_last = ['extract_growth_rates', 'statistical_validation']
        
        for step in self.validation_steps:
            if step['name'] in sequential_last:
                success = self.execute_step(step)
                if not success:
                    logger.warning(f"Step {step['name']} failed")
        
        # Finalize results
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_time'] = str(datetime.now() - self.start_time)
        
        # Determine overall status
        if len(self.results['steps_failed']) == 0:
            self.results['status'] = 'SUCCESS'
        elif len(self.results['steps_completed']) > len(self.validation_steps) / 2:
            self.results['status'] = 'PARTIAL_SUCCESS'
        else:
            self.results['status'] = 'FAILED'
        
        # Save results
        output_file = 'validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        return self.results
    
    def print_summary(self):
        """Print validation summary"""
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        print(f"Status: {self.results['status']}")
        print(f"Total time: {self.results.get('total_time', 'N/A')}")
        print(f"Steps completed: {len(self.results['steps_completed'])}/{len(self.validation_steps)}")
        
        if self.results['steps_completed']:
            print("\n✓ Completed steps:")
            for step in self.results['steps_completed']:
                print(f"  - {step}")
        
        if self.results['steps_failed']:
            print("\n✗ Failed steps:")
            for step in self.results['steps_failed']:
                print(f"  - {step}")
        
        # Check for final validation results
        if os.path.exists('reports/validation_summary.json'):
            with open('reports/validation_summary.json', 'r') as f:
                final_results = json.load(f)
                
            print("\n" + "="*70)
            print("THEORETICAL VALIDATION RESULTS")
            print("="*70)
            
            if final_results.get('overall_validation'):
                print("✓✓✓ ALL THEOREMS VALIDATED SUCCESSFULLY ✓✓✓")
            else:
                print("⚠ Partial validation achieved")
            
            print(f"Theorems validated: {', '.join(final_results.get('theorems', []))}")
        
        print("\n" + "="*70)
        print("Output files generated:")
        print("  - validation_results.json (master results)")
        print("  - rti_validation.log (detailed log)")
        
        if os.path.exists('reports/validation_report.tex'):
            print("  - reports/validation_report.tex (LaTeX report)")
        
        if os.path.exists('analysis/'):
            analysis_files = os.listdir('analysis/')
            print(f"  - {len(analysis_files)} analysis files in analysis/")
        
        print("="*70)

def main():
    """Main execution with command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Master RTI Validation Executor'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-run all steps even if completed'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run validation steps in parallel where possible'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        help='Run only a specific step'
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           RTI OPTIMAL CONTROL PAPER VALIDATION              ║
║                  Automated Validation System                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create validator
    validator = MasterRTIValidator(
        skip_completed=not args.no_skip,
        parallel=args.parallel
    )
    
    try:
        if args.step:
            # Run specific step
            step_dict = next((s for s in validator.validation_steps 
                            if s['name'] == args.step), None)
            if step_dict:
                success = validator.execute_step(step_dict)
                print(f"\nStep '{args.step}' {'completed' if success else 'failed'}")
            else:
                print(f"Unknown step: {args.step}")
                print(f"Available steps: {[s['name'] for s in validator.validation_steps]}")
        else:
            # Run full validation
            results = validator.execute_full_validation()
            validator.print_summary()
            
            # Return appropriate exit code
            if results['status'] == 'SUCCESS':
                sys.exit(0)
            elif results['status'] == 'PARTIAL_SUCCESS':
                sys.exit(1)
            else:
                sys.exit(2)
                
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        print(traceback.format_exc())
        sys.exit(255)

if __name__ == "__main__":
    main()