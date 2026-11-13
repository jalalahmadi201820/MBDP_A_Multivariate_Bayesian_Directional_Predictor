"""Run only posterior_threshold_sensitivity. Outputs: outputs/posterior_threshold_sensitivity/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'posterior_threshold_sensitivity'], check=True)
