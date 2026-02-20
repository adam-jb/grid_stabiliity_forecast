"""
Run all data fetchers in sequence.

Usage:
  python fetch/fetch_all.py

This will fetch/refresh all datasets. Each fetcher saves incrementally,
so it's safe to interrupt and resume.
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("Octopus Agile tariff", "fetch/fetch_agile.py"),
    ("Elexon generation + wholesale", "fetch/fetch_elexon.py"),
    ("BESS capacity", "fetch/fetch_bess_capacity.py"),
    ("Ancillary markets", "fetch/fetch_ancillary.py"),
    ("Renewable capacity", "fetch/fetch_renewables.py"),
]


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("FETCHING ALL DATA")
    print("=" * 60)

    for name, script in SCRIPTS:
        print(f"\n{'─' * 60}")
        print(f"▶ {name} ({script})")
        print(f"{'─' * 60}\n")

        script_path = os.path.join(base_dir, script)
        start = time.time()

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=base_dir,
        )

        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"\n  [{status}] {name} ({elapsed:.0f}s)")

    print(f"\n{'=' * 60}")
    print("ALL FETCHERS COMPLETE")
    print(f"{'=' * 60}")

    # Show what we've got
    raw_dir = os.path.join(base_dir, "data", "raw")
    proc_dir = os.path.join(base_dir, "data", "processed")

    print("\nFiles in data/raw/:")
    if os.path.exists(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            size = os.path.getsize(os.path.join(raw_dir, f))
            print(f"  {f} ({size / 1024:.0f} KB)")

    print("\nFiles in data/processed/:")
    if os.path.exists(proc_dir):
        for f in sorted(os.listdir(proc_dir)):
            path = os.path.join(proc_dir, f)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"  {f} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
