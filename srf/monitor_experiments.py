#!/usr/bin/env python3
"""
Monitor and track all running SRF experiments in real-time.
"""
import json
import os
from pathlib import Path
from datetime import datetime
import time

# Experiment directories to monitor
EXPERIMENT_DIRS = {
    "safe_advanced": "results/autoresearch_advanced/",
    "comprehensive": "results/autoresearch_comprehensive/",
    "advanced_cl": "results/autoresearch_advanced_cl/",
}


def check_experiment_status(output_dir: str) -> dict:
    """Check the status of experiments in a directory."""
    results = {}

    if not os.path.exists(output_dir):
        return {"status": "not_started", "completed": 0, "total": 0, "results": []}

    # Check for completed experiments
    exp_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name != ".git"]

    for exp_dir in exp_dirs:
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)

                # Extract accuracy from POPE results
                if "pope" in data:
                    pope_data = data["pope"]
                    if "method" in pope_data:
                        # Get the SRF accuracy (not baseline)
                        method_data = pope_data["method"]
                        if method_data:
                            # Use the first beta value (usually "0.0" for base SRF)
                            first_beta = list(method_data.values())[0]
                            accuracy = first_beta.get("accuracy", 0.0) * 100
                            results[exp_dir.name] = {
                                "accuracy": accuracy,
                                "status": "completed",
                                "timestamp": os.path.getmtime(summary_file),
                            }
            except Exception as e:
                results[exp_dir.name] = {"status": "error", "error": str(e)}
        else:
            # Check if experiment is running
            output_file = exp_dir / "output.txt"
            if output_file.exists():
                file_age = time.time() - os.path.getmtime(output_file)
                if file_age < 300:  # Modified in last 5 minutes
                    results[exp_dir.name] = {"status": "running", "age": file_age}
                else:
                    results[exp_dir.name] = {"status": "unknown", "age": file_age}
            else:
                results[exp_dir.name] = {"status": "pending"}

    return results


def print_status():
    """Print current status of all experiments."""
    print("\n" + "="*70)
    print("🔍 SRF AUTOSEARCH EXPERIMENT MONITOR")
    print("="*70)
    print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_completed = 0
    total_running = 0
    total_pending = 0

    for exp_type, exp_dir in EXPERIMENT_DIRS.items():
        print(f"\n📁 {exp_type.upper()}:")
        print(f"   Directory: {exp_dir}")

        results = check_experiment_status(exp_dir)

        completed = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "completed")
        running = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "running")
        pending = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") in ["pending", "unknown"])

        total_completed += completed
        total_running += running
        total_pending += pending

        print(f"   Status: {completed} completed, {running} running, {pending} pending")

        if completed > 0:
            best_exp = None
            best_acc = 0.0
            for name, data in results.items():
                if data.get("status") == "completed":
                    acc = data.get("accuracy", 0.0)
                    if acc > best_acc:
                        best_acc = acc
                        best_exp = name

            print(f"   Best so far: {best_exp} ({best_acc:.1f}%)")

            # Show all completed results
            print(f"   Results:")
            for name, data in sorted(results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
                if data.get("status") == "completed":
                    acc = data.get("accuracy", 0.0)
                    print(f"     {name}: {acc:.1f}%")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_completed} completed, {total_running} running, {total_pending} pending")
    print(f"{'='*70}")

    # Recommendations
    if total_completed > 0:
        print(f"\n💡 NEXT STEPS:")
        print(f"  • Check detailed results in experiment directories")
        print(f"  • If best config found, test with full dataset (n=1000)")
        print(f"  • If no improvement, try advanced CLIP models (ViT-L/14)")
    else:
        print(f"\n⏳ WAITING for experiments to complete...")
        print(f"  • Check back in 10-15 minutes")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor SRF experiments")
    parser.add_argument("--watch", action="store_true", help="Watch mode - update every 30s")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    args = parser.parse_args()

    if args.once or not args.watch:
        print_status()
    else:
        try:
            while True:
                print_status()
                print(f"\n⏰ Next update in 30 seconds... (Ctrl+C to exit)")
                time.sleep(30)
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped.")


if __name__ == "__main__":
    main()
