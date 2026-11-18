from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(p.parents[1]/'workflow'/'new_build.py'),'--characters',p.name,'--render'],check=True)
