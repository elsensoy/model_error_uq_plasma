### ERROR CAUSE AND SOLUTION


- Running MAP optimization failed with an error similar to:

```
	ERROR: SystemError: opening file ".../postprocess/output_twozonebohm.json": No such file or directory
``` 
This was caused because the simulation attempted to write the output file to a directory that didn’t exist yet. While the file path was configured, Python (and Julia underneath) cannot automatically create intermediate folders when writing to a file. This didn't happen with gen-data because it was already handled in path resolution, but for `twozonebohm` runs,  postprocess file creating step during yaml validation was missing.

### Solution: Auto-Creating the Output Directory
To prevent this, the following added in the `run_model()` function before simulation begins:

### Create Output Directory before W Stage
```
output_path = Path(output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Ensured directory exists for output file: {output_path.parent}")
```
Output_file now exist before attempting to save simulation results.




## Resolution for Data Generation or Upload


### Update 2: YAML Placeholder Resolution (Preferred Path)
- first update: If reference_data is set in the YAML like so:
```
reference_data: "${settings.output_dir}/postprocess/output_multilogbohm.json"
```

This string is automatically resolved during Pydantic config verification using the `resolve_all_paths()` method. This guarantees that dynamic paths like `${settings.output_dir}` point to the correct directory at runtime.

- If the resolved file exists, it's loaded immediately.

#### Fallback: Project-Wide Search
If the resolved path doesn't point to a valid file, the system invokes:

```
	find_file_anywhere("output_multilogbohm.json")
```
This searches the current directory and parent directories (excluding .venv and inaccessible locations) for the most recently modified match.

If found, it proceeds with that file.

### Last Resort: Fallback to postprocess.output_file

If no file is found using the reference path or search, it falls back to the default:
```
	settings.postprocess.output_file["MultiLogBohm"]
```
### Final Catch: Failure
If no usable file is found and gen_data: false, the system logs:

```
	[FATAL] No ground truth data could be found.
	[SUGGESTION] Try setting `gen_data: true` in your input YAML.
```