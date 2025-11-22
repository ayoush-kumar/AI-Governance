# main.py
from pathlib import Path
import subprocess
import sys
import os

# 1) move into project root so all paths are relative
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # everything below uses relative paths

# 2) relative script locations
MODULE_FILE = Path("module/hackathonpipeline.py")
DASHBOARD_FILE = Path("Dashboard/dashboard_streamlit.py")

def run_py(path: Path, *args: str) -> None:
    if not path.exists():
        print(f"missing: {path}")
        sys.exit(1)
    cmd = [sys.executable, str(path), *args]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(rc)

def run_streamlit(path: Path, port: int = 8501, *args: str) -> None:
    if not path.exists():
        print(f"missing: {path}")
        sys.exit(1)
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(path),
        "--server.headless=true",
        "--server.port", str(port),
        "--browser.gatherUsageStats=false",
    ]
    if args:
        cmd.extend(map(str, args))
    print(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        sys.exit(rc)

if __name__ == "__main__":
    # ensures outputs like governance_prototype/predictions.csv resolve relative to repo root
    run_py(MODULE_FILE)
    run_streamlit(DASHBOARD_FILE, port=8080)
