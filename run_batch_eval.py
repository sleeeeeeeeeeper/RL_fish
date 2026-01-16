#!/usr/bin/env python3
"""
Batch Evaluation Runner

此脚本用于自动化评估训练好的强化学习模型。
它会扫描 RESULTS_DIR 目录下的实验文件夹，并使用 `task_path/eval.py` 对训练好的模型进行评估。
"""

import os
import sys
import time
import glob
import subprocess
import argparse
import re
from datetime import datetime
from pathlib import Path

# Configuration
DEFAULT_CONCURRENCY = 2
EVAL_SCRIPT = "task_path/eval.py"
RESULTS_DIR = "task_path/results"
CONFIG_BASE_DIR = "task_path/configs/experiments"
LOG_DIR_ROOT = "eval_batch_logs"

# Mapping from results subdirectory to config subdirectory
RESULT_TO_CONFIG_MAP = {
    # 'a1': 'a1_algorithm',
    'a2': 'a2_environment',
    'a3': 'a3_hyperparameter',
    # 'a4': 'a4_reward',
}

def parse_experiment_dir(exp_dir_name):
    """
    Parse experiment directory name to extract config name.
    
    Format: {algo}_{config_name}_{timestamp}
    Example: ppo_exp_a1_l1_ppo_20260108_001306
    Returns: (algo, config_name) or (None, None) if parsing fails
    """
    # Pattern: algorithm name (ppo/sac/td3) + _ + config_name + _ + timestamp (YYYYMMDD_HHMMSS)
    pattern = r'^(ppo|sac|td3)_(.+)_(\d{8}_\d{6})$'
    match = re.match(pattern, exp_dir_name)
    
    if match:
        algo = match.group(1)
        config_name = match.group(2)
        timestamp = match.group(3)
        return algo, config_name
    
    return None, None

def find_config_file(config_name, config_subdir, config_base_dir):
    """
    Find the configuration file for given config name.
    
    Args:
        config_name: Name extracted from experiment directory
        config_subdir: Subdirectory in configs (e.g., 'a1_algorithm')
        config_base_dir: Base config directory path
    
    Returns:
        Full path to config file or None if not found
    """
    config_dir = os.path.join(config_base_dir, config_subdir)
    
    # Try exact match first
    config_path = os.path.join(config_dir, f"{config_name}.json")
    if os.path.exists(config_path):
        return config_path
    
    # Try searching recursively in subdirectories
    search_pattern = os.path.join(config_dir, "**", f"{config_name}.json")
    matches = glob.glob(search_pattern, recursive=True)
    
    if matches:
        return matches[0]
    
    return None

def get_experiments(results_dir):
    """
    Scan results directory and collect all experiments with their metadata.
    Supports both flat structure (a1, a4) and nested structure (a2, a3).
    
    Returns:
        List of dicts with keys: exp_path, model_path, config_path, exp_name
    """
    experiments = []
    
    # Scan each subdirectory in results (a1, a2, a3, a4, etc.)
    for result_subdir in os.listdir(results_dir):
        result_subdir_path = os.path.join(results_dir, result_subdir)
        
        if not os.path.isdir(result_subdir_path):
            continue
        
        # Get corresponding config subdirectory
        config_subdir = RESULT_TO_CONFIG_MAP.get(result_subdir)
        if config_subdir is None:
            # Skip subdirectories not in mapping (like 'analysis')
            continue
        
        # Recursively find all experiment directories with models/best_model.zip
        # This handles both flat (a1/exp_dir) and nested (a2/current/exp_dir) structures
        for root, dirs, files in os.walk(result_subdir_path):
            # Skip analysis directories
            if 'analysis' in root:
                continue
            
            # Check if this directory contains models/best_model.zip
            model_path = os.path.join(root, "models", "best_model.zip")
            if not os.path.exists(model_path):
                continue
            
            # This is an experiment directory
            exp_path = root
            exp_dir = os.path.basename(exp_path)
            
            # Parse experiment directory name
            algo, config_name = parse_experiment_dir(exp_dir)
            if algo is None or config_name is None:
                print(f"Warning: Cannot parse experiment directory name: {exp_dir}")
                continue
            
            # Find config file
            config_path = find_config_file(config_name, config_subdir, CONFIG_BASE_DIR)
            if config_path is None:
                print(f"Warning: Config file not found for experiment: {exp_dir}")
                print(f"         Expected config: {config_name}.json in {config_subdir}")
                continue
            
            experiments.append({
                'exp_path': exp_path,
                'model_path': model_path,
                'config_path': config_path,
                'exp_name': exp_dir,
                'algo': algo,
            })
    
    return experiments

def run_batch_eval(concurrency, dry_run=False, results_filter=None, n_episodes=100):
    """Run evaluations with specified concurrency."""
    
    # 1. Setup paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script_path = os.path.join(root_dir, EVAL_SCRIPT)
    results_dir_path = os.path.join(root_dir, RESULTS_DIR)
    
    if not os.path.exists(eval_script_path):
        print(f"Error: Evaluation script not found at {eval_script_path}")
        return
    
    if not os.path.exists(results_dir_path):
        print(f"Error: Results directory not found at {results_dir_path}")
        return
    
    # 2. Find experiments
    print(f"Scanning for experiments in {results_dir_path}...")
    experiments = get_experiments(results_dir_path)
    
    # Apply filter if specified
    if results_filter:
        experiments = [e for e in experiments if results_filter in e['exp_path']]
        print(f"Applied filter: {results_filter}")
    
    if not experiments:
        print("No experiments found to evaluate!")
        return
    
    print(f"Found {len(experiments)} experiments to evaluate.")
    
    # 3. Create log directory for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_dir = os.path.join(root_dir, LOG_DIR_ROOT, f"batch_{timestamp}")
    
    if not dry_run:
        os.makedirs(batch_log_dir, exist_ok=True)
        print(f"Logging output to: {batch_log_dir}")
    
    # 4. Execution Loop
    queue = experiments.copy()
    active_processes = []  # List of (process_handle, exp_name, log_file_handle)
    completed_count = 0
    total_count = len(experiments)
    
    start_time = time.time()
    
    print("\nStarting evaluation...")
    print(f"Concurrency level: {concurrency}")
    print(f"Episodes per evaluation: {n_episodes}")
    print("-" * 60)
    
    try:
        while queue or active_processes:
            # Check for completed processes
            still_active = []
            for p, name, f_log in active_processes:
                if p.poll() is not None:
                    # Process finished
                    completed_count += 1
                    duration = time.time() - p.start_time
                    status = "Success" if p.returncode == 0 else f"Failed (code {p.returncode})"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished: {name} ({status}) - {duration:.1f}s")
                    
                    if f_log:
                        f_log.close()
                else:
                    still_active.append((p, name, f_log))
            
            active_processes = still_active
            
            # Start new processes if slots are available
            while len(active_processes) < concurrency and queue:
                exp_info = queue.pop(0)
                exp_name = exp_info['exp_name']
                model_path = exp_info['model_path']
                config_path = exp_info['config_path']
                algo = exp_info['algo']
                
                # Determine log file path
                log_file_path = os.path.join(batch_log_dir, f"{exp_name}.log")
                
                cmd = [
                    sys.executable, eval_script_path,
                    "--model-path", model_path,
                    "--config", config_path,
                    "--algo", algo,
                    "--render",
                    "--n-episodes", str(n_episodes),
                ]
                
                if dry_run:
                    print(f"[DRY RUN] Would evaluate: {exp_name}")
                    print(f"           Model: {model_path}")
                    print(f"           Config: {config_path}")
                    print(f"           Command: {' '.join(cmd)}")
                    completed_count += 1  # Simulate completion
                    continue
                
                # Launch process
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {exp_name}")
                try:
                    f_log = open(log_file_path, "w")
                    p = subprocess.Popen(
                        cmd,
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        cwd=root_dir,
                        env=os.environ.copy()
                    )
                    p.start_time = time.time()
                    active_processes.append((p, exp_name, f_log))
                except Exception as e:
                    print(f"Failed to launch {exp_name}: {e}")
                    completed_count += 1
            
            if dry_run and not queue:
                break
            
            # Wait a bit before polling again
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n\nUser interrupted! Terminating active processes...")
        for p, name, f_log in active_processes:
            p.terminate()
            if f_log:
                f_log.close()
        print("Done.")
        sys.exit(1)
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Batch evaluation completed in {total_time:.1f}s.")
    print(f"Total: {total_count}, Completed: {completed_count}")
    if not dry_run:
        print(f"Logs available in: {batch_log_dir}")
        print("Check individual log files for evaluation details and potential errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch model evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
        help="Number of parallel evaluations"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter experiments by path substring (e.g., 'a1' or 'ppo')"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of episodes per evaluation"
    )
    
    args = parser.parse_args()
    
    run_batch_eval(
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        results_filter=args.filter,
        n_episodes=args.n_episodes
    )
