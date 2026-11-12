# CalibrationSystem.py — Multimedia Geometry Changes Explanation

## What Was Changed

`CalibrationSystem.py` contains `CalibrationConfig` and `CalibrationSystem`.
Four additions expose the multimedia parameters through the existing `.cfg`
and `.json` file formats without changing any existing behaviour.

---

## Overview of Changes

```
CalibrationConfig (dataclass)
  [NEW] n1, n2, d, n3, z_glass_top  — five new fields
  [NEW] use_multimedia property       — True/False gate
  [NEW] to_multimedia_params()        — builds MultimediaParams

_load_from_cfg
  [NEW] reads optional [Multimedia] section
  [NEW] _read_multimedia_from_cfg() helper

_save_as_cfg
  [NEW] writes [Multimedia] section (only when use_multimedia is True)

_save_as_json
  [FIX] switched from __dict__ to dataclasses.fields() to avoid
        including class-level attributes and properties

print_config_summary
  [NEW] prints multimedia stack

save_results → summary.txt
  [NEW] writes multimedia status
```

---

## New Fields on CalibrationConfig

### Five scalar fields (dataclass)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n1` | float | `1.0` | Refractive index, camera-side medium (air) |
| `n2` | list[float] | `[1.0]` | Refractive indices of glass layer(s) |
| `d` | list[float] | `[0.0]` | Thickness of each glass layer in mm |
| `n3` | float | `1.0` | Refractive index, object-side medium (fluid) |
| `z_glass_top` | float | `0.0` | Z-coordinate of air/glass interface in mm |

All defaults correspond to air (n=1.0 everywhere), so `use_multimedia`
returns `False` and nothing in the pipeline changes for existing experiments.

### Convention
Matches Maas (1996) and the liboptv / OpenPTV `mm_np` struct:

```
n1 = air (camera side)  →  n2 = glass  →  n3 = fluid (object side)
```

---

## New Property — use_multimedia

```python
@property
def use_multimedia(self) -> bool:
    return (
        abs(self.n1 - 1.0) > 1e-9
        or any(abs(n - 1.0) > 1e-9 for n in self.n2)
        or abs(self.n3 - 1.0) > 1e-9
    )
```

### Purpose
Single gate used throughout the pipeline:
```python
if config.use_multimedia:
    mm = config.to_multimedia_params()
    # ... build CameraParams, LUT, etc.
```

Mirrors the condition in liboptv `ray_tracing.c`:
```c
if (n1 != 1 || n2[0] != 1 || n3 != 1) → init_mmLUT()
```

---

## New Method — to_multimedia_params

```python
def to_multimedia_params(self) -> MultimediaParams:
```

Constructs a `MultimediaParams` object from the config fields. Called once
after `load_config()` in `GoCal.py`. Returns `MultimediaParams(n1, n2, d, n3)`.

Raises `RuntimeError` if `multimedia_params.py` is not importable.

---

## .cfg File Extension — [Multimedia] Section

### Reading
`_load_from_cfg` calls the new `_read_multimedia_from_cfg(parser)` helper
after building the `CalibrationConfig` object. If the `[Multimedia]` section
is absent, the defaults (n=1 everywhere) remain unchanged.

```ini
[Multimedia]
N1: 1.0
N2: 1.5
D:  5.0
N3: 1.333
Z-Glass-Top: 0.0
```

Multi-layer glass uses space-separated values:
```ini
N2: 1.5   1.5
D:  4.0   4.0
```

### Writing
`_save_as_cfg` writes the `[Multimedia]` section **only** when
`config.use_multimedia` is `True`. Existing air-only `.cfg` files written by
the old code are not modified.

---

## JSON Fix — _save_as_json

### Before
```python
data = {k: list(v) if isinstance(v, tuple) else v
        for k, v in config.__dict__.items()}
```
This included class-level attributes (`principal_point`), properties
(`use_multimedia`), and methods (`to_multimedia_params`), causing errors.

### After
```python
import dataclasses
data = {}
for f in dataclasses.fields(config):
    val = getattr(config, f.name)
    if isinstance(val, (tuple, np.ndarray)):
        val = list(val)
    data[f.name] = val
```
Only declared dataclass instance fields are serialised.

---

## Backward Compatibility

| Scenario | Behaviour |
|---|---|
| `.cfg` file without `[Multimedia]` | Defaults (n=1) kept, `use_multimedia = False`, no pipeline change |
| `.cfg` file with `[Multimedia]` and `N2: 1.0`, `N3: 1.0` | `use_multimedia = False`, same as above |
| `multimedia_params.py` not on path | `to_multimedia_params()` raises `RuntimeError`; import guard in `GoCal.py` catches this and logs a warning |
| Existing JSON configs | Load correctly — new fields get defaults via dataclass |

---

## Module-Level Import Guard

```python
try:
    from multimedia_params import MultimediaParams
    _MM_AVAILABLE = True
except ImportError:
    _MM_AVAILABLE = False
```

`CalibrationSystem.py` still loads and runs fully when `multimedia_params.py`
is absent. Only `to_multimedia_params()` will raise if called.

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV*.
  Section 2.1: the 16-parameter camera model. Section 2.2: multimedia geometry.
- liboptv source: `ray_tracing.c`, struct `mm_np`.
- See also: `GoCal_EXPLANATION.md`, `Configuration_File_Guide.md`
