"""DAMPED kernel guard: the decayed features and their boxes pass the brute-force tests before any pair is built."""
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         str(R / "scripts" / "t_damped.py"), str(R / "scripts" / "t_damped_features.py")]).returncode)
