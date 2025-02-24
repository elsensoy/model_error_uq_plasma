#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
from pathlib import Path

def debug(msg):
    #Force debug messages for immediate print
    print(f"[DEBUG] {msg}", file=sys.stderr)
    sys.stderr.flush()

def find_project_root():
    #Find the project root directory ( hall_opt/main.py).
    script_dir = os.path.abspath(os.getcwd())
    debug(f"Starting search for project root from: {script_dir}")

    while not os.path.exists(os.path.join(script_dir, "hall_opt", "main.py")) and script_dir != os.path.dirname(script_dir):
        debug(f"Checking: {script_dir}")
        script_dir = os.path.dirname(script_dir)

    if not os.path.exists(os.path.join(script_dir, "hall_opt", "main.py")):
        print("Error: Could not find hall_opt/main.py", file=sys.stderr)
        sys.exit(1)

    debug(f"Project root found: {script_dir}")
    return script_dir

def find_python():
    debug("Searching for Python executable...")
    python_executable = shutil.which("python") or shutil.which("python3")
    if not python_executable:
        print("Error: Python not found. Please install Python and add it to PATH.", file=sys.stderr)
        sys.exit(1)
    
    debug(f"Using Python: {python_executable}")
    return python_executable

# def find_julia():
#     """Find the Julia executable dynamically."""
#     debug("Searching for Julia executable...")
#     julia_executable = shutil.which("julia")
    
#     if not julia_executable:
#         print("Error: Julia not found. Please install Julia and add it to PATH.", file=sys.stderr)
#         sys.exit(1)
    
#     debug(f"Using Julia: {julia_executable}")
#     return julia_executable

def find_julia():
    """Find the Julia executable dynamically and return its absolute path."""
    debug("Searching for Julia executable...")

    # Step 1: First, Julia PATH check using shutil.which()
    julia_executable = shutil.which("julia")

    # Step 2: If not found, search manually in "known" Julia installation directories
    if not julia_executable:
        debug("Julia not found in system PATH. Searching common installation locations...")

        possible_paths = [
            Path.home() / ".julia/juliaup/bin/julia.exe",  # JuliaUp Windows generic path
            Path("C:/Users") / os.getlogin() / ".julia/juliaup/bin/julia.exe",  # Specific Windows user path
            Path("C:/Users") / os.getlogin() / ".julia/juliaup" / "julia-1.11.3+0.x64.w64.mingw32/bin/julia.exe",
            Path("C:/Users") / os.getlogin() / ".julia/juliaup" / "julia-*/bin/julia.exe",  # Handle different Julia versions
            Path.home() / "AppData/Local/Programs/Julia-1.11.3/bin/julia.exe",  # Default for Windows install
            Path("/usr/local/bin/julia"),  # macOS/Linux common location
            Path("/opt/julia/bin/julia"),  # Alternative Linux install
        ]

        for path in possible_paths:
            # If wildcard path (like julia-*), find the newest Julia version
            if "*" in str(path):
                found_versions = sorted(Path(path.parent).glob("*/bin/julia.exe"))
                if found_versions:
                    path = found_versions[-1]  # Select the latest version
            
            if path.exists():
                julia_executable = str(path)
                debug(f"Found Julia at: {julia_executable}")
                break

    if not julia_executable:
        print("Error: Julia not found. Please install Julia or add it to PATH.", file=sys.stderr)
        return None  # Return None instead of exiting

    debug(f"Using Julia: {julia_executable}")
    return julia_executable  # Return Julia's absolute path

#TODO: PROJECT PATH CAN'T ACCESS TO HALLTHRUSTER_PROJECT FROM THE ROOT
def find_hallthruster_path():
    """Find the HallThruster Python package using Julia's HallThruster.PYTHON_PATH."""
    try:
        #  Run Julia command to get HallThruster's Python path
        hallthruster_path = subprocess.check_output(["julia", "-e", 'using HallThruster; print(HallThruster.PYTHON_PATH)'], text=True).strip()
        hallthruster_path = Path(hallthruster_path).resolve()

        if hallthruster_path.exists():
            print(f"[DEBUG] Found HallThruster path: {hallthruster_path}")
            return str(hallthruster_path)
        else:
            print("[ERROR] HallThruster path not found in expected location!")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to locate HallThruster path via Julia: {e}")
        return None

#  Find HallThruster path
hallthruster_path = find_hallthruster_path()

if hallthruster_path and hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)
    print(f"[DEBUG] HallThruster path set to: {hallthruster_path}")

#  import HallThruster
try:
    import hallthruster as het
except ModuleNotFoundError:
    print(" [ERROR] Could not import HallThruster! Check if the package is installed correctly.")

def main():
    """Execute main.py inside hall_opt/ with arguments."""
    debug(f"Arguments received: {sys.argv[1:]}")
    
    project_root = find_project_root()
    python_executable = find_python()
    julia_executable = find_julia()
    hall_opt_dir = os.path.join(project_root, "hall_opt")  # hall_opt as absolute
    main_py = os.path.join(hall_opt_dir, "main.py")  #  main .pypath

    # TODO: Set the working directory to the project root ( -c )
    os.chdir(hall_opt_dir)
    debug(f"Changed working directory to {hall_opt_dir}")

    # Fix: Add project root to PYTHONPATH ensure imports work
    os.environ["PYTHONPATH"] = project_root + os.pathsep + hall_opt_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    debug(f"Set PYTHONPATH={os.environ['PYTHONPATH']}")

    # Fix: Add Julia's bin directory to PATH so it's available
    julia_bin_dir = os.path.dirname(julia_executable)
    os.environ["PATH"] = julia_bin_dir + os.pathsep + os.environ.get("PATH", "")
    debug(f"Set PATH={os.environ['PATH']}")

    debug(f"Running: {python_executable} {main_py} {' '.join(sys.argv[1:])}")

    # Run main.py with all user-provided arguments
    os.execv(python_executable, [python_executable, main_py] + sys.argv[1:])

if __name__ == "__main__":
    main()
