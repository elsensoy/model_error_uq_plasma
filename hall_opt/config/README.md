
## Config/ METHODS


### verifier.py
This module provides utility functions for loading, validating, and extracting configurations from YAML files used in the optimization pipeline. It also supports model-specific configuration extraction and method validation.

---

### Function: `load_yaml`

```python
def load_yaml(file_path: str) -> Optional[dict]
```

**Purpose:**  
Safely loads a YAML configuration file.

**Parameters:**
- `file_path` *(str)*: Path to the YAML file.

**Returns:**
- *(dict or None)*: Parsed YAML content or `None` on failure.

**Notes:**
- Handles file-not-found and YAML parsing errors with clear error messages.

---

### Function: `verify_all_yaml`

```python
def verify_all_yaml(yaml_data: dict, source_path: Optional[str] = None) -> Optional[Settings]
```

**Purpose:**  
Validates the loaded YAML content against the `Settings` Pydantic model and resolves placeholders.

**Parameters:**
- `yaml_data` *(dict)*: Parsed YAML data.
- `source_path` *(str, optional)*: Original YAML file path for tracking.

**Returns:**
- *(Settings or None)*: Validated `Settings` object or `None` if validation fails.

**Notes:**
- Embeds the config file path into the `Settings.general.config_file`.
- Performs validation and path resolution using `Settings.resolve_all_paths()`.

---

### Function: `get_valid_optimization_method`

```python
def get_valid_optimization_method(method: Optional[str], source_yaml: Optional[str] = None) -> str
```

**Purpose:**  
Validates the specified optimization method, falling back to a default if needed.

**Parameters:**
- `method` *(str, optional)*: Optimization method from the config.
- `source_yaml` *(str, optional)*: Name of the source YAML file (for logging).

**Returns:**
- *(str)*: Valid method name. Defaults to `"Nelder-Mead"`.

**Notes:**
- Only `"Nelder-Mead"` is currently supported.
- Logs informative messages based on input.

---

### Function: `extract_anom_model`

```python
def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]
```

**Purpose:**  
Extracts and returns the anomalous model configuration for a given `model_type`.

**Parameters:**
- `settings` *(Settings)*: Validated configuration object.
- `model_type` *(str)*: The anomalous model type to extract.

**Returns:**
- *(dict)*: Model-specific config dictionary with type included.

**Notes:**
- Raises an error if the requested `model_type` is not found under `settings.config_settings.anom_model`.

---

## Dependencies

- [yaml (PyYAML)](https://pyyaml.org/)
- [re (Regular Expressions)](https://docs.python.org/3/library/re.html)
- [pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
- [pydantic.BaseModel](https://docs.pydantic.dev/)
- [typing](https://docs.python.org/3/library/typing.html) — for type annotations.
- `Settings` — Custom configuration schema from `config.dict`.

---
