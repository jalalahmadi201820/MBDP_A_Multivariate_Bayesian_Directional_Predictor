"""Run only ridge_lambda_sensitivity. Outputs: outputs/ridge_lambda_sensitivity/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'ridge_lambda_sensitivity'], check=True)
