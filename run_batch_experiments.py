#!/usr/bin/env python3
"""
Batch Experiment Runner

此脚本用于自动执行多个强化学习训练实验。
它会扫描 CONFIG_DIR 目录中的 JSON 配置文件，并使用 `task_path/train.py` 运行它们。
"""

import os
import sys
import time
import glob
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Configuration
DEFAULT_CONCURRENCY = 3
TRAIN_SCRIPT = "task_path/train.py"
# CONFIG_DIR = "task_path/configs"
CONFIG_DIR = "task_path/configs/experiments/a4_reward"
LOG_DIR_ROOT = "batch_logs"

def get_config_files(config_dir):
    """Recursively find all .json files in the config directory."""
    files = []
    # Use glob to find all json files recursively
    search_pattern = os.path.join(config_dir, "**", "*.json")
    for filepath in glob.glob(search_pattern, recursive=True):
        files.append(filepath)
    return sorted(files)

def run_batch_experiments(concurrency, dry_run=False):
    """Run experiments with specified concurrency."""
    
    # 1. Setup paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(root_dir, TRAIN_SCRIPT)
    config_dir_path = os.path.join(root_dir, CONFIG_DIR)
    
    if not os.path.exists(train_script_path):
        print(f"Error: Training script not found at {train_script_path}")
        return

    # 2. Find configs
    print(f"Scanning for configurations in {config_dir_path}...")
    configs = get_config_files(config_dir_path)
    
    if not configs:
        print("No configuration files found!")
        return
    
    print(f"Found {len(configs)} experiments to run.")
    
    # 3. Create log directory for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_dir = os.path.join(root_dir, LOG_DIR_ROOT, f"batch_{timestamp}")
    
    if not dry_run:
        os.makedirs(batch_log_dir, exist_ok=True)
        print(f"Logging output to: {batch_log_dir}")
    
    # 4. Execution Loop
    queue = configs.copy()
    active_processes = [] # List of (process_handle, config_name, log_file_handle)
    completed_count = 0
    total_count = len(configs)
    
    start_time = time.time()
    
    print("\nStarting execution...")
    print(f"Concurrency level: {concurrency}")
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
                config_path = queue.pop(0)
                config_name = os.path.basename(config_path)
                
                # Determine log file path
                log_file_path = os.path.join(batch_log_dir, f"{config_name}.log")
                
                cmd = [sys.executable, train_script_path, "--config", config_path]
                
                if dry_run:
                    print(f"[DRY RUN] Would start: {config_name}")
                    print(f"           Command: {' '.join(cmd)}")
                    completed_count += 1 # Simulate completion
                    continue
                
                # Launch process
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {config_name}")
                try:
                    f_log = open(log_file_path, "w")
                    p = subprocess.Popen(
                        cmd,
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        cwd=root_dir,
                        env=os.environ.copy() # Pass current env
                    )
                    p.start_time = time.time()
                    active_processes.append((p, config_name, f_log))
                except Exception as e:
                    print(f"Failed to launch {config_name}: {e}")
                    completed_count += 1 
            
            if dry_run and not queue:
                break
                
            # Wait a bit before polling again to save CPU
            time.sleep(1.0)
            
            # Optional: Print status update periodically (e.g. every 30s) if wanted, 
            # but the start/finish logs are usually enough.

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
    print(f"Batch execution completed in {total_time:.1f}s.")
    print(f"Total: {total_count}, Completed: {completed_count}")
    if not dry_run:
        print(f"Logs available in: {batch_log_dir}")
        print("Check individual log files for training details and potential errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch RL experiments.")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Path to configuration directory (default: task_path/configs)")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, 
                        help="Number of parallel experiments")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without executing")
    
    args = parser.parse_args()
    
    # Override CONFIG_DIR if provided
    if args.config_dir:
        CONFIG_DIR = args.config_dir
    
    run_batch_experiments(args.concurrency, args.dry_run)
