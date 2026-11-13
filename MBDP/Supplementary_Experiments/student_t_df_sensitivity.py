"""Run only student_t_df_sensitivity. Outputs: outputs/student_t_df_sensitivity/"""
from pathlib import Path
import subprocess,sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('shared_engine.py')), 'student_t_df_sensitivity'], check=True)
