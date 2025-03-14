import sys
import os
import subprocess
import shutil
from pathlib import Path

def debug(msg):
    """Helper function to print debug messages."""
    print(f"[DEBUG] {msg}")

def find_python():
    debug("Searching for Python executable...")
    python_executable = shutil.which("python") or shutil.which("python3")
    
    if not python_executable:
        print("[ERROR] Python not found. Please install Python and add it to PATH.", file=sys.stderr)
        sys.exit(1)
    
    debug(f"Using Python: {python_executable}")
    return python_executable

def find_julia():
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
        print("[ERROR] Julia not found. Please install Julia or add it to PATH.", file=sys.stderr)
        sys.exit(1)  # Exit if Julia is not found

    debug(f"Using Julia: {julia_executable}")
    return julia_executable  # Return Julia's absolute path

def find_hallthruster_path(julia_executable):
    """Find the HallThruster Python package using Julia's HallThruster.PYTHON_PATH."""
    if not julia_executable:
        print("[ERROR] Cannot determine HallThruster path because Julia was not found.")
        return None

    try:
        debug(f"Using Julia executable: {julia_executable}")
        debug("Attempting to find HallThruster path using Julia...")

        # Run Julia command to get HallThruster's Python path
        hallthruster_path = subprocess.check_output(
            [julia_executable, "-e", 'using HallThruster; print(HallThruster.PYTHON_PATH)'],
            text=True
        ).strip()

        # Resolve to absolute path
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
    except FileNotFoundError:
        print("[ERROR] Julia is not installed or not found in system PATH.")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error locating HallThruster path: {e}")
        return None

def main():
    try:
        """Execute main.py inside hall_opt/ with arguments."""
        debug(f"Arguments received: {sys.argv[1:]}")

        # Step 1: Find Julia
        julia_executable = find_julia()

        # Step 2: Find HallThruster
        hallthruster_path = find_hallthruster_path(julia_executable)

        # Step 3: Check HallThruster path and import
        if hallthruster_path:
            debug(f"Checking if HallThruster path '{hallthruster_path}' is already in sys.path...")
            if hallthruster_path not in sys.path:
                sys.path.append(hallthruster_path)
                debug(f"HallThruster path added to sys.path: {hallthruster_path}")
            else:
                debug(f"HallThruster path already in sys.path: {hallthruster_path}")

            # Try importing HallThruster
            debug("Attempting to import HallThruster module...")
            try:
                import hallthruster as het
                debug("[SUCCESS] HallThruster imported successfully!")
            except ModuleNotFoundError:
                print("[ERROR] Could not import HallThruster! The package may not be installed correctly.")
                debug(f"Checked path: {hallthruster_path}")
        else:
            print("[ERROR] HallThruster path could not be determined!")
            debug("Ensure Julia is installed and HallThruster is properly configured.")
            sys.exit(1)  # Exit if HallThruster path is not found

        # Step 4: Prepare execution environment
        project_root = Path(__file__).resolve().parent
        python_executable = find_python()
        hall_opt_dir = project_root / "hall_opt"
        main_py = hall_opt_dir / "main.py"

        # Set working directory
        os.chdir(hall_opt_dir)
        debug(f"Changed working directory to {hall_opt_dir}")

        # Fix PYTHONPATH for proper imports
        os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{hall_opt_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        debug(f"Set PYTHONPATH={os.environ['PYTHONPATH']}")

        # Fix PATH to include Julia bin
        os.environ["PATH"] = f"{Path(julia_executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
        debug(f"Set PATH={os.environ['PATH']}")

        debug(f"Running: {python_executable} {main_py} {' '.join(sys.argv[1:])}")

        os.execv(python_executable, [python_executable, str(main_py)] + sys.argv[1:])
        process = subprocess.Popen([python_executable, str(main_py)] + sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Exiting.")
        sys.exit(0)  

if __name__ == "__main__":
    main()
