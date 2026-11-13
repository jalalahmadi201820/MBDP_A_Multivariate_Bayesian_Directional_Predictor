"""Run only transaction_cost_sensitivity. Outputs: outputs/transaction_cost_sensitivity/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'transaction_cost_sensitivity'], check=True)
