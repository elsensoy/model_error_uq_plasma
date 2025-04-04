### File Searching Utility: `find_file_anywhere`

This utility function helps locate a specific file within a project structure, even if the exact path isn't known beforehand. It searches the current directory and optionally parent directories, while allowing certain common directories (like virtual environments) to be skipped.

**Purpose:**

To find the most recently modified instance of a file with a specific `filename`, searching from a `start_dir` and up a specified number of parent directory levels (`max_depth_up`). It intelligently skips searching within specified `exclude_dirs`. This is useful for locating configuration files, data files, or outputs when their exact location might vary slightly or when we want to avoid searching within dependency/build folders.

**Parameters:**

* `filename` (str): **Required.** The exact name of the file we want to find (e.g., `"output_multilogbohm.json"`, `"config.yaml"`).
* `start_dir` (str): *Optional.* The directory path where the search should begin. Defaults to `"."` (the current working directory).
* `max_depth_up` (int): *Optional.* How many parent directory levels *above* the `start_dir` should also be searched.
    * Defaults to `1`.
    * `0` means only search within the `start_dir` tree.
    * `1` means search `start_dir` tree AND the parent directory's tree.
    * `2` means search `start_dir`, parent, and grandparent trees, and so on.
* `exclude_dirs` (Optional[List[str]]): *Optional.* A list of directory *names* to completely ignore during the search. The function will not look inside any directory whose name is in this list. Defaults to `None` (no exclusions). Common examples: `['venv', '.venv', '__pycache__', '.git', 'node_modules', 'build', 'dist']`.

**Behavior:**

1.  **Starting Point:** Begins searching from the resolved absolute path of `start_dir`.
2.  **Upward Traversal:** It then searches the parent directory tree, the grandparent tree, and so on, up to the level specified by `max_depth_up`.
3.  **Search Mechanism:** Uses Python's standard `os.walk` to traverse directory trees efficiently.
4.  **Exclusion:** Crucially, if `exclude_dirs` is provided, `os.walk` is prevented from descending into any directory whose *name* matches an entry in the list. This avoids unnecessary searching in virtual environments, Git folders, etc.
5.  **Matching:** Looks for files exactly matching the provided `filename`.
6.  **Selection:** If multiple matching files are found (outside the excluded directories), it identifies the one with the most recent modification time (`st_mtime`).
7.  **Error Handling:** Includes basic handling for permission errors during directory traversal or file access issues.

**Return Value:**

* **`pathlib.Path`:** An absolute `pathlib.Path` object pointing to the most recently modified matching file found.
* **`None`:** If the file is not found within the searched directories (respecting exclusions and depth limits), or if an error prevented determining the latest file.

**Usage Example:**

```python
from pathlib import Path
from typing import Optional, List, Set
import os
#  the find_file_anywhere function definition should be available here

# --- Example Scenario ---
# Imagine running a script in /path/to/project/scripts/
# and need to find 'results.json' which might be in ./outputs/
# or ../outputs/ relative to the script.

# Define directories we definitely don't want to search inside
common_excludes = ['venv', '.venv', '__pycache__', '.git', 'node_modules', 'build']

# Search starting from the current directory, going up 1 parent level
found_path = find_file_anywhere(
    filename="results.json",
    start_dir=".",      # Start search where the script is running
    max_depth_up=1,     # Search '.' and '..' trees
    exclude_dirs=common_excludes
)

if found_path:
    print(f"Found results file at: {found_path}")
    # We can now use the found_path object, e.g., read the file
    # with open(found_path, 'r') as f:
    #     content = f.read()
else:
    print("Could not locate 'results.json'.")

# Example searching only the current directory tree:
config_path = find_file_anywhere(
    filename="config.yaml",
    start_dir=".",
    max_depth_up=0, # Only search current dir tree
    exclude_dirs=common_excludes
)

if config_path:
    print(f"Config file found at: {config_path}")


---
---
---



## iter_methods.py 

### Utility Functions: Directory and Filename Generation

This script provides two key utility functions for safely and dynamically creating result directories and filenames. These are especially useful in scenarios where you want to avoid overwriting previous results, such as during iterative model training, logging, or experimentation.

---

The following Python libraries are used in this script:

- [os](https://docs.python.org/3/library/os.html) — For interacting with the operating system, e.g., paths and directories.
- [json](https://docs.python.org/3/library/json.html) — Standard library for encoding and decoding JSON (not directly used in the provided functions, but likely used elsewhere).
- [numpy](https://numpy.org/) — Powerful library for numerical computing (not used in the provided functions, but likely required in the broader module).
- [pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) — Object-oriented file system paths.
- `from ..config.dict import Settings` — Assumes a custom module within the project for configuration handling. This import is not used in the given functions.

---

## Function: `get_next_results_dir`

```python
def get_next_results_dir(base_dir: str, base_name: str) -> str:
```

### Description:
Creates a uniquely named directory under `base_dir` by appending a numeric suffix (e.g., `base_name-1`, `base_name-2`, etc.), ensuring that existing folders are not overwritten.

### Parameters:
- `base_dir` *(str)*: Root directory where results directories are to be created.
- `base_name` *(str)*: Base name for the subdirectory.

### Returns:
- *(str)*: The path to the newly created, unique results directory.

### Example:
```python
get_next_results_dir("hall_opt/results/mcmc", "mcmc-results")
# Might return: hall_opt/results/mcmc/mcmc-results-1
```

---

## Function: `get_next_filename`

```python
def get_next_filename(base_filename: str, directory: str, extension=".json") -> str:
```

### Description:
Generates a unique filename by appending a numeric suffix, ensuring no existing file in the directory is overwritten. Useful for logging iteration outputs, metrics, checkpoints, etc.

### Parameters:
- `base_filename` *(str)*: The base name of the file (e.g., "metrics").
- `directory` *(str)*: The folder where the file will be saved.
- `extension` *(str, optional)*: File extension, defaults to ".json".

### Returns:
- *(str)*: A full path to the uniquely generated file.

### Example:
```python
get_next_filename("metrics", "hall_opt/results/mcmc/mcmc-results-1/iter_metrics/")
# Might return: hall_opt/results/mcmc/mcmc-results-1/iter_metrics/metrics_1.json
```

## Notes

- Both functions provide **directory creation** using `Path(...).mkdir(parents=True, exist_ok=True)`.
- They are **safe for concurrent or repeated runs** since they will not overwrite existing files or folders.


##  Function: `find_latest_results_dir`

```python
def find_latest_results_dir(base_dir: str, base_name: str) -> Optional[Path]:
```

### Description:
Searches for directories matching the naming pattern `{base_name}-N` inside `base_dir` and returns the one with the highest number `N`. Useful for retrieving the most recent set of results or experiment output.

### Parameters:
- `base_dir` *(str)*: Parent folder where result directories are stored.
- `base_name` *(str)*: Prefix used in directory names (e.g., "map-results").

### Returns:
- *(Path or None)*: Path to the latest directory (highest `N`) or `None` if none found.

### Example:
```python
find_latest_results_dir("hall_opt/results/mcmc", "mcmc-results")
# Might return: Path('hall_opt/results/mcmc/mcmc-results-3')
```

### Notes:
- Uses a regex pattern to match folder names like `base_name-N`.
- Logs warnings or info if the directory doesn't exist or no matches are found.
- Returns `None` on error or if no matching directories exist.
