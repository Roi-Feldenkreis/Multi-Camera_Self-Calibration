# `preparedata` — Function Explanation

## What does it do?

`preparedata` reads a **BlueC `.rad` correspondence file** — a plain ASCII table where each row describes one matched point pair — and splits it into the two arrays that a camera calibration toolbox needs:

- **X** — the 3D world coordinates of each point
- **x** — the corresponding 2D pixel projections in the image

These two arrays are the fundamental input to any calibration algorithm: given enough point pairs, the toolbox can solve for the camera's intrinsic parameters (focal length, principal point, distortion) and extrinsic parameters (rotation and translation).

---

## File Format

The `.rad` file is a whitespace-delimited ASCII table with at least 6 columns per row:

| Col (MATLAB) | Col (Python) | Content |
|---|---|---|
| 1 | 0 | X world coordinate |
| 2 | 1 | Y world coordinate |
| 3 | 2 | Z world coordinate |
| 4 | 3 | *(unused — skipped)* |
| 5 | 4 | u image coordinate |
| 6 | 5 | v image coordinate |

Each row = one point correspondence (one 3D point and its 2D image projection).

---

## Inputs & Outputs

| Name | Type | Description |
|------|------|-------------|
| `filename` | str | Path to the `.rad` ASCII file |
| `X` *(output)* | (3, N) ndarray | 3D world points |
| `x` *(output)* | (2, N) ndarray | 2D image projections |

Arrays are **transposed** so that each column is one point — the convention used throughout the CalTech calibration toolbox.

---

## Step-by-step Breakdown

### Step 1 — Load the file
MATLAB uses `load(filename, '-ASCII')` which reads any whitespace-delimited numeric file into a matrix. The Python equivalent is `np.loadtxt(filename)`, which behaves identically for this file format.

### Step 2 — Extract 3D world points
```python
X = datamat[:, 0:3].T   # shape (3, N)
```
Columns 0–2 hold the XYZ world coordinates. Transposing turns the (N, 3) slice into (3, N) so each column is one point.

### Step 3 — Extract 2D image projections
```python
x = datamat[:, 4:6].T   # shape (2, N)
```
Column 3 is skipped (unused in the original). Columns 4–5 hold the (u, v) pixel coordinates. Same transposition gives shape (2, N).

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| File loading | `load(filename, '-ASCII')` | `np.loadtxt(filename)` |
| Column indexing | 1-based: columns `1:3`, `5:6` | 0-based: `[:, 0:3]`, `[:, 4:6]` |
| Octave compatibility check | `exist('OCTAVE_VERSION')` branch | Not needed — `np.loadtxt` handles both cases uniformly |

The original MATLAB code has an Octave branch that handles `.mat`-style struct loading. In Python this is unnecessary — `np.loadtxt` reads plain ASCII reliably regardless of platform.
