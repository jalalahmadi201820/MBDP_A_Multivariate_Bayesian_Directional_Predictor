"""Run only btc_lag_selector_comparison. Outputs: outputs/btc_lag_selector_comparison/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'btc_lag_selector_comparison'], check=True)
