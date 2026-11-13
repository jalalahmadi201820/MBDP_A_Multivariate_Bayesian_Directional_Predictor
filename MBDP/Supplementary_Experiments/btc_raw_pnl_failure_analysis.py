"""Run only btc_raw_pnl_failure_analysis. Outputs: outputs/btc_raw_pnl_failure_analysis/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'btc_raw_pnl_failure_analysis'], check=True)
