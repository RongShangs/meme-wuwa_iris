from pathlib import Path
import subprocess, sys
here = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(here.parents[1] / 'workflow' / 'render.py'), str(here / '项目.json')], check=True)
