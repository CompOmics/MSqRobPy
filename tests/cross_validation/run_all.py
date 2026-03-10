"""All-in-one runner: generate data, call R, compare with Python."""

import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print(f"\n{'─'*60}")
    print(f"  Step: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'─'*60}\n")
    result = subprocess.run(cmd, cwd=THIS_DIR)
    if result.returncode != 0:
        print(f"\n!! {label} failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    python = sys.executable

    # 1. Generate test data
    run([python, "generate_test_data.py"], "Generate shared test data")

    # 2. Run R script
    # Try common Rscript locations
    rscript = "Rscript"
    try:
        subprocess.run([rscript, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        # Try common Windows paths
        for candidate in [
            r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
            r"C:\Program Files\R\R-4.3.2\bin\Rscript.exe",
            r"C:\Program Files\R\R-4.3.1\bin\Rscript.exe",
        ]:
            if os.path.exists(candidate):
                rscript = candidate
                break
        else:
            print("\n!! Rscript not found. Please run Step 2 manually:")
            print(f"   Rscript run_msqrob2_r.R {THIS_DIR}")
            print("   Then re-run: python compare_r_python.py")
            sys.exit(1)

    run([rscript, "run_msqrob2_r.R", THIS_DIR], "Run msqrob2 in R")

    # 3. Compare
    run([python, "compare_r_python.py"], "Compare R vs Python results")


if __name__ == "__main__":
    main()
