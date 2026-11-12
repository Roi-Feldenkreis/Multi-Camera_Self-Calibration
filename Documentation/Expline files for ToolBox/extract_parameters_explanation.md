# `extract_parameters` — Function Explanation

## What does it do?

After the calibration optimiser has converged, all camera parameters live
packed together in a single flat array called `solution`. `extract_parameters`
**unpacks that flat vector** into the individual named parameters that every
other function in the toolbox expects: focal lengths, principal point, skew,
distortion coefficients, and for each image its rotation vector, translation
vector, rotation matrix, and homography.

---

## The `solution` vector layout

| Python index | MATLAB index | Parameter | Size |
|---|---|---|---|
| 0–1 | 1–2 | `fc` — focal lengths [fx, fy] | 2 |
| 2–3 | 3–4 | `cc` — principal point [cx, cy] | 2 |
| 4 | 5 | `alpha_c` — skew coefficient | 1 |
| 5–9 | 6–10 | `kc` — distortion coefficients | 5 |
| 10–14 | 11–15 | *(reserved / padding)* | 5 |
| 15 + 6k … +2 | 16+6k … +3 | `omc_k` — rotation vector image k | 3 |
| 15 + 6k + 3 … +5 | 16+6k+4 … +6 | `Tc_k` — translation vector image k | 3 |

Total length: **15 + 6 × n_ima**

The first 15 slots are reserved for intrinsics even though only 10 are used —
a design choice in the CalTech toolbox that leaves room for future parameters.

---

## Inputs & Outputs

| Name | Type | Description |
|------|------|-------------|
| `solution` | (15+6n,) array | Flat optimised parameter vector |
| `n_ima` | int | Total number of calibration images |
| `active_images` | list of int | 1 = active, 0 = skip (None = all active) |
| `solution_error` | (15+6n,) array | Parameter uncertainties (None = zeros) |
| `intrinsics` *(out)* | dict | fc, cc, alpha_c, kc, KK, inv_KK + errors |
| `extrinsics` *(out)* | list of n dicts | omc, Tc, Rc, H + errors per image |

---

## Step-by-step Breakdown

### Step 1 — Validate active images
Calls `check_active_images` to get `active_flags` and `ind_active` — which
images to process and which to fill with NaN.

### Step 2 — Extract intrinsic parameters
Slice the first 10 elements of `solution` directly into `fc`, `cc`, `alpha_c`,
and `kc`. Same for `solution_error`.

### Step 3 — Build the calibration matrix K
```
KK = [[fx,  fx·alpha_c,  cx],
      [ 0,  fy,          cy],
      [ 0,  0,           1 ]]
```
This is the standard camera intrinsic matrix. The skew term `alpha_c` appears
as an off-diagonal element multiplied by `fx`. `inv_KK` is pre-computed as it
is frequently needed to back-project image points to normalised coordinates.

### Step 4 — Extract extrinsic parameters per image
For each image `kk`, the 6-parameter block starts at index `15 + 6*kk`:
- First 3 values → rotation vector `omc` → converted to rotation matrix `Rc`
  by calling `rodrigues()`
- Last 3 values → translation vector `Tc`

**Homography H** is then computed as:
```
H = K · [r1 | r2 | t]    (first two columns of R, plus t)
```
normalised so that `H[2,2] = 1`. This is the planar homography relating the
calibration pattern to the image.

**Inactive images** get NaN arrays for all fields so downstream code can
detect them.

---

## `check_active_images` alignment

`extract_parameters` relies on all three values returned by
`check_active_images`:

| Return value | Used for |
|---|---|
| `n_ima` | loop bound |
| `active_flags` | decide active vs NaN per image |
| `ind_active` | available for callers that need the index list |

The Python `check_active_images` was updated alongside this conversion to
return `ind_active` — a 0-based list of active image indices — mirroring
MATLAB's `find(active_images)` idiom.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Dynamic variables | `eval(['omc_' num2str(kk) ' = omckk'])` | `extrinsics[kk]['omc'] = omckk` — list of dicts |
| 1-based indexing | `solution(16)` = first extrinsic | `solution[15]` (0-based) |
| `inv(KK)` | Built-in `inv` | `np.linalg.inv(KK)` |
| Column selection | `Rckk(:,1)`, `Rckk(:,2)` | `Rckk[:, 0]`, `Rckk[:, 1]` |

As in `comp_error_calib`, the biggest structural improvement is replacing
MATLAB's `eval`-based dynamic variable names (`omc_1`, `omc_2`, ...) with a
clean Python list of dicts — one dict per image.
