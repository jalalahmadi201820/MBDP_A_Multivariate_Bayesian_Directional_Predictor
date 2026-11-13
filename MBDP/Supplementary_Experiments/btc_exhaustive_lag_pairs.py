"""Run only btc_exhaustive_lag_pairs. Outputs: outputs/btc_exhaustive_lag_pairs/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'btc_exhaustive_lag_pairs'], check=True)
