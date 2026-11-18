from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
if '--validate' not in sys.argv[1:]:
    subprocess.run([sys.executable, str(here / '准备.py')], check=True)
subprocess.run([sys.executable, str(here.parents[1] / 'workflow' / 'render.py'),
                str(here / '项目.json'), *sys.argv[1:]], check=True)
