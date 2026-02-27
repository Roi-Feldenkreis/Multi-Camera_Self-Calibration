# Structure-from-Motion Python Modules

## Module Organization

### Core Modules

1. **Utils.py** - Core utility functions
   - `p2e()` - Projective to Euclidean conversion (replaces MATLAB's nhom)
   - `k2i()` - Index conversion for triplets
   - `normu()` - Hartley normalization
   - `spread_depths_col()` - Depth column spreading
   - `subseq_longest()` - Longest subsequence finder
   - `raddist_apply()` - Apply radial distortion to points
   - `raddist_deriv()` - Compute Jacobians for radial distortion
   - And many more...

### Main Functions

2. **u2FI.py** - Fundamental matrix estimation
   - Uses: `Utils.normu()`, `Utils.p2e()`

3. **M2Fe.py** - Multi-view epipolar geometry
   - Uses: `u2FI`, `Utils.p2e()`

4. **depth_estimation.py** - Depths from epipolar geometry
   - Uses: `Utils.p2e()`, `Utils.k2i()`

5. **approximate.py** - Rank approximation
   - Uses: None (standalone)

6. **create_nullspace.py** - Null-space creation
   - Uses: `Utils.k2i()`, `Utils.spread_depths_col()`

7. **L2depths.py** - Depths from algebraic basis
   - Uses: `Utils.k2i()`, `Utils.spread_depths_col()`

8. **balance_triplets.py** - Matrix balancing
   - Uses: `Utils.k2i()`

9. **eval_y_and_dy.py** - Bundle adjustment objective function
    - Uses: `Utils.p2e()`, `Utils.raddist_apply()`, `Utils.raddist_deriv()`
    - No internal helper functions

10. **qPXbundle_cmp.py** - Complete bundle adjustment solver
    - Uses: `Utils.p2e()`, `Utils.normP()`, `Utils.normx()`, `Utils.raddist_apply()`, `Utils.raddist_deriv()`
    - **Internal helper function:**
      - `F()` - Internal objective function evaluator
    - Supports radial distortion (3-parameter model)
    - These are NOT in Utils.py, they are specific to this module

11. **fill_prmm.py** - Complete PRMM filling pipeline
    - Uses: `create_nullspace`, `L2depths`, `approximate`, `Utils.comb()`, `Utils.k2i()`
    - **Internal helper functions:**
      - `nullspace2L()` - Compute basis from null-space via SVD
      - `svd_suff_data()` - Check if SVD has sufficient data
    - Orchestrates the complete reconstruction pipeline

12. **fill_mm_sub.py** - Sub-scene projective reconstruction
    - Uses: `M2Fe`, `depth_estimation`, `balance_triplets`, `fill_prmm`, `Utils.k2i()`, `Utils.eucl_dist_only()`
    - No internal helper functions
    - High-level reconstruction for single sub-scene/sequence
    - Handles both sequence and central image modes

13. **fill_mm.py** - Main projective reconstruction
    - Uses: `fill_mm_sub`, `balance_triplets`, `Utils.normu()`, `Utils.dist()`, `Utils.subseq_longest()`, `Utils.k2i()`
    - **Internal helper functions:**
      - `compute_predictions()` - Compute predictor functions for strategies
      - `strength()` - Compute central image strength
      - `set_rows_cols()` - Set rows/cols for strategy
      - `set_sequence()` - Set sequence information
      - `normM()` - Normalize measurements (Hartley)
      - `normMback()` - Undo normalization
    - **Complete reconstruction pipeline** with strategy selection
    - Iteratively fills missing data and performs final SVD factorization

14. **bundle_PX_proj.py** - Projective bundle adjustment with image conditioning
    - Uses: `Utils.k2i()`, `Utils.p2e()`, `Utils.hom()`, `Utils.normP()`, `Utils.normx()`
    - **Internal helper functions:**
      - `levmarq()` - Levenberg-Marquardt optimizer
      - `vgg_conditioner_from_image()` - Create image conditioning matrix
      - `ObjectiveWithJacobian` class - Compute residuals and Jacobian
    - Image-based preconditioning for better numerical stability
    - Alternative to qPXbundle_cmp (no radial distortion support)

15. **fill_mm_bundle.py** - ULTIMATE TOP-LEVEL WRAPPER
    - Uses: `fill_mm`, `bundle_PX_proj`, `Utils.k2i()`, `Utils.normalize_cut()`, `Utils.dist()`
    - **No internal helper functions** (pure wrapper)
    - **THE main entry point** for complete reconstruction
    - Combines: fill_mm (reconstruction) + bundle_PX_proj (refinement)
    - Stores linear result before BA for comparison
    - **ONE FUNCTION TO RULE THEM ALL** 🚀

## Import Structure

```
Utils.py ───────────────────────┐
                                ├──> u2FI.py ────────────────┐
                                ├──> M2Fe.py ────────────────┤
                                ├──> depth_estimation.py ────┤
                                ├──> approximate.py ──────┐  │
                                ├──> create_nullspace.py ─┤  │
                                ├──> L2depths.py ──────────┤  │
                                ├──> balance_triplets.py ─┼──┤
                                ├──> eval_y_and_dy.py ────┼──┤
                                ├──> qPXbundle_cmp.py ────┤  │
                                └──────────────────────┬───┴──┴──> fill_prmm.py ─┐
                                                       │                         │
                                                       └─────────────────────────┴──> fill_mm_sub.py ─┐
                                                                                                       │
                                                                                                       └──> fill_mm.py ─┐
                                                                                                                        │
Utils.py ──────> bundle_PX_proj.py ─────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                        │
                                                                                                                        └──> fill_mm_bundle.py (ULTIMATE)
```

**Hierarchy:**
- **Level 1 (Core):** Utils, u2FI
- **Level 2 (Geometry):** M2Fe, depth_estimation
- **Level 3 (Factorization):** create_nullspace, L2depths, approximate, balance_triplets
- **Level 4 (Optimization):** eval_y_and_dy, qPXbundle_cmp, bundle_PX_proj
- **Level 5 (Mid-Pipeline):** fill_prmm
- **Level 6 (Sub-Scene):** fill_mm_sub
- **Level 7 (Reconstruction):** fill_mm
- **Level 8 (ULTIMATE):** **fill_mm_bundle** ← THE main entry point 🎯

**Note:** bundle_PX_proj is also at Level 4 but used directly by fill_mm_bundle

## Key Conversions from MATLAB

| MATLAB Function | Python Equivalent | Location |
|----------------|-------------------|----------|
| `nhom(x)` | `Utils.p2e(x)` | Utils.py |
| `k2i(rows)` | `Utils.k2i(rows, step=3)` | Utils.py |
| `normu(u)` | `Utils.normu(u)` | Utils.py |
| `spread_depths_col(...)` | `Utils.spread_depths_col(...)` | Utils.py |
| `raddist(q, u0, kappa)` | `Utils.raddist_apply(q, u0, kappa)` | Utils.py |
| `[dqdu, dqdu0, dqdkappa] = raddist(...)` | `Utils.raddist_deriv(...)` | Utils.py |

## Function-Specific Helpers

Some functions have internal helper functions that are NOT in Utils.py:

### Shared Utility Functions (in Utils.py)

**Coordinate transformations:**
- `p2e(x)` - Projective to Euclidean coordinates (homogeneous → Euclidean)
- `hom(x)` - Euclidean to homogeneous coordinates (Euclidean → homogeneous)
- `k2i(k, step)` - Index conversion for matrix rows

**Normalization:**
- `normu(u)` - Hartley normalization for image points
- `normP(P)` - Normalize camera matrices by Frobenius norm
- `normx(x)` - Normalize 3D points by Euclidean norm

**Radial distortion:**
- `raddist_apply()` - Apply radial distortion
- `raddist_deriv()` - Compute radial distortion Jacobians

These functions are centralized in Utils.py and used across multiple modules.

## Usage Example

### Recommended: Use fill_mm_bundle (Complete Pipeline)

```python
from fill_mm_bundle import fill_mm_bundle
import numpy as np

# Measurement matrix
M = ...  # (3m × n) with NaNs for missing data

# Image sizes
imsize = np.array([[width, height]] * m).T  # (2 × m)

# Options
opt = {
    'strategy': -1,      # Auto-select best
    'verbose': True,
    'no_BA': False,      # Enable bundle adjustment
    'metric': 1
}

# ONE LINE → Complete reconstruction!
P, X, u1, u2, info = fill_mm_bundle(M, imsize, None, opt)

print(f"Linear error: {info['err']['fact']:.6f}")
print(f"Refined error: {info['err']['BA']:.6f}")
```

### Advanced: Manual Pipeline (for fine control)

```python
from fill_mm import fill_mm
from bundle_PX_proj import bundle_PX_proj
from Utils import Utils

# Step 1: Initial reconstruction
P, X, u1, u2, info = fill_mm(M, opt)

# Step 2: Prepare for bundle adjustment
m = M.shape[0] // 3
n = M.shape[1]
r1 = np.setdiff1d(np.arange(m), u1)
r2 = np.setdiff1d(np.arange(n), u2)

M_subset = M[Utils.k2i(r1, step=3), :][:, r2]
q = Utils.normalize_cut(M_subset)

# Step 3: Bundle adjustment
P, X = bundle_PX_proj(P, X, q, imsize, None, opt)
```

## Testing

All modules include `if __name__ == "__main__":` test blocks:

```bash
python u2FI.py
python M2Fe.py
python depth_estimation.py
python approximate.py
python create_nullspace.py
python L2depths.py
python balance_triplets.py
python eval_y_and_dy.py
python qPXbundle_cmp.py
python fill_prmm.py
python fill_mm_sub.py
python fill_mm.py
python bundle_PX_proj.py
python fill_mm_bundle.py  # THE main entry point
```

## Notes

- **Main entry point:** Use `fill_mm_bundle()` for complete reconstruction pipeline (reconstruction + bundle adjustment)
- **Common utilities consolidated in Utils.py:** `hom()`, `p2e()`, `normP()`, `normx()`, `normu()`, `k2i()`, `raddist_apply()`, `raddist_deriv()`
- **No code duplication:** All shared functions are in Utils.py and imported where needed
- **Coordinate conventions:** All functions use `Utils.p2e()` instead of MATLAB's `nhom()` for consistency
- **Normalization:** `normP()` and `normx()` are now shared utilities, not module-specific
- **14 modules total:** From low-level utilities to complete reconstruction pipeline

