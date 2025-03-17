import sys
import os
import subprocess
import shutil
from pathlib import Path

# Dynamically set the project root based on the script's location
PROJECT_ROOT = Path(__file__).resolve().parent
HALL_OPT_DIR = PROJECT_ROOT / "hall_opt"

# Ensure the `hall_opt` directory is in `sys.path`
if str(HALL_OPT_DIR) not in sys.path:
    sys.path.append(str(HALL_OPT_DIR))

# Import `main.py` dynamically
from hall_opt.main import main as main_script

def debug(msg):
    """Helper function to print debug messages."""
    print(f"[DEBUG] {msg}")

def find_julia():
    """Find the Julia executable dynamically."""
    debug("Searching for Julia executable...")
    julia_executable = shutil.which("julia")

    if not julia_executable:
        possible_paths = [
            Path.home() / ".julia/juliaup/bin/julia.exe",
            Path("C:/Users") / os.getlogin() / ".julia/juliaup/bin/julia.exe",
            Path("C:/Users") / os.getlogin() / ".julia/juliaup/julia-1.11.3+0.x64.w64.mingw32/bin/julia.exe",
            Path.home() / "AppData/Local/Programs/Julia-1.11.3/bin/julia.exe",
            Path("/usr/local/bin/julia"),
            Path("/opt/julia/bin/julia"),
        ]

        for path in possible_paths:
            if path.exists():
                julia_executable = str(path.resolve())
                debug(f"Found Julia at: {julia_executable}")
                break

    if not julia_executable:
        print("[ERROR] Julia not found. Install Julia or add it to PATH.", file=sys.stderr)
        sys.exit(1)

    debug(f"Using Julia: {julia_executable}")
    return julia_executable

def find_hallthruster_path(julia_executable):
    """Find the HallThruster Python package using Julia."""
    try:
        debug(f"Using Julia executable: {julia_executable}")
        hallthruster_path = subprocess.check_output(
            [julia_executable, "-e", 'using HallThruster; print(HallThruster.PYTHON_PATH)'],
            text=True
        ).strip()

        hallthruster_path = Path(hallthruster_path).resolve()

        if hallthruster_path.exists():
            debug(f"HallThruster path found: {hallthruster_path}")
            return str(hallthruster_path)
        else:
            print(f"[ERROR] HallThruster path '{hallthruster_path}' does not exist!")
            return None
    except subprocess.CalledProcessError as e:
        print("[ERROR] Julia command failed! Could not determine HallThruster path.")
        debug(f"Error message: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error locating HallThruster path: {e}")
        return None

def setup_paths():
    """Ensure the correct environment paths are set."""
    julia_executable = find_julia()
    hallthruster_path = find_hallthruster_path(julia_executable)

    if hallthruster_path and hallthruster_path not in sys.path:
        sys.path.append(hallthruster_path)
        debug(f"HallThruster path added to sys.path: {hallthruster_path}")

    # Fix PYTHONPATH for proper imports
    os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{HALL_OPT_DIR}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    debug(f"Set PYTHONPATH={os.environ['PYTHONPATH']}")

    # Fix PATH to include Julia bin
    os.environ["PATH"] = f"{Path(julia_executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    debug(f"Set PATH={os.environ['PATH']}")

def main():
    try:
        setup_paths()
        debug(f"Arguments received: {sys.argv[1:]}")

        # Call main.py's main() function directly
        main_script()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Exiting cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    main()
