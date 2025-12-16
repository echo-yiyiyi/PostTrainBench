#!/usr/bin/env python3
"""
Copy solve_parsed.txt (or solve_out.txt fallback) from result directories
to a new organized structure.
"""
import os
import shutil
from pathlib import Path
# Constants - modify these as needed
INPUT_DIRS = [
    "claude_claude-opus-4-5_final_v3",
    "codex_gpt-5.1-codex-max_final_v3",
    "codex_gpt-5.1-codex_final_v3",
    "codex_gpt-5.2_final_v3",
    "gemini_models_gemini-3-pro-preview_final_v3",
]
RESULTS_BASE = Path(os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
OUTPUT_DIR = os.path.join(RESULTS_BASE, "collected_results")
def extract_model_name(dir_name: str) -> str:
    """
    Extract model name from directory name.
    e.g., 'claude_claude-opus-4-5_final_v3' -> 'claude-opus-4-5'
         'codex_gpt-5.1-codex-max_final_v3' -> 'gpt-5.1-codex-max'
    """
    parts = dir_name.split("_")
    # Join the middle parts (skip first and last two)
    model_name = "_".join(parts[1:-2])
    return model_name
def main():
    output_base = Path(OUTPUT_DIR)
    
    for input_dir_name in INPUT_DIRS:
        input_dir = RESULTS_BASE / input_dir_name
        
        if not input_dir.is_dir():
            print(f"Warning: Directory does not exist: {input_dir}")
            continue
        
        model_name = extract_model_name(input_dir_name)
        model_dir = output_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate over all subdirectories
        for subdir in input_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Determine source file (prefer solve_parsed.txt)
            src_file = subdir / "solve_parsed.txt"
            solve_filename = "solve_parsed.txt"
            if not src_file.exists():
                src_file = subdir / "solve_out.txt"
                solve_filename = "solve_out.txt"
                if not src_file.exists():
                    print(f"Warning: No solve_parsed.txt or solve_out.txt in {subdir}")
                    continue
            
            # Create output directory with same name as original subdirectory
            task_name = subdir.name
            dest_dir = model_dir / task_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy solve file with original filename
            dest_file = dest_dir / solve_filename
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {src_file} -> {dest_file}")
            
            # Copy metrics file
            src_metrics = subdir / "metrics.json"
            dest_metrics = dest_dir / "metrics.json"
            if src_metrics.exists():
                shutil.copy2(src_metrics, dest_metrics)
            else:
                with open(dest_metrics, 'w') as f:
                    f.write("No metrics.json produced.")

if __name__ == "__main__":
    main()