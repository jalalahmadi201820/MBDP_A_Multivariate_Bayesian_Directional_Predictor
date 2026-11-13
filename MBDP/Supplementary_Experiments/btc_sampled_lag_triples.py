"""Run only btc_sampled_lag_triples. Outputs: outputs/btc_sampled_lag_triples/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'btc_sampled_lag_triples'], check=True)
