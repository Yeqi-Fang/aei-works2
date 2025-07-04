#!/usr/bin/env python3
"""
Linear interpolation of a huge HierarchSearch .dat file
with robust scaling to keep Qhull happy.
"""

import argparse, glob, pathlib
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import LinearNDInterpolator
from  scipy.spatial      import Delaunay


# ----------------------------------------------------------------------
def read_data_from_file(path: str) -> np.ndarray:
    """Return ndarray(N,4): freq, f1dot, f2dot, 2Fr from **one** .dat file"""
    rows = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('%') or not ln.strip():        # comments / blanks
                continue
            p = ln.split()
            if len(p) < 8:
                continue
            try:
                rows.append((float(p[0]), float(p[3]), float(p[4]), float(p[7])))
            except ValueError:
                pass
    return np.asarray(rows, dtype=np.float64)

def read_data_from_files(paths) -> np.ndarray:
    """
    Path list/iterator → one concatenated Nx4 array.
    *Keeps memory use modest* by collecting row-lists, then one final vstack.
    """
    chunks = [read_data_from_file(p) for p in paths]
    return np.vstack(chunks) if chunks else np.empty((0, 4), dtype=np.float64)


# ----------------------------------------------------------------------
def _scale(X):
    """min-max scale → [0,1]"""
    mins = X.min(0)
    span = np.ptp(X, 0)
    span[span == 0] = 1.0          # guard against zero range
    return (X - mins) / span, mins, span


def build_interpolator(data: np.ndarray,
                       n_sample: int | None = 200_000):
    """
    Returns
    -------
    interp : LinearNDInterpolator
    mins   : 3-vector of column minima   (for later queries)
    span   : 3-vector of column ranges   (  ''     ''   )
    """
    X, y = data[:, :3], data[:, 3]

    # ---------- optional random sub-sampling (huge speed-up) -------------
    if n_sample and n_sample < len(X):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(X), size=n_sample, replace=False)
        X, y = X[keep], y[keep]

    # ---------- rescale & robust Delaunay --------------------------------
    Xn, mins, span = _scale(X)
    tri = Delaunay(Xn, qhull_options='QJ')     # 'QJ' = joggle
    return LinearNDInterpolator(tri, y), mins, span


# ----------------------------------------------------------------------
def plot_slice(interp, mins, span,
               f_grid, f1dot_fix, f2dot_fix,
               ax=None):
    pts = np.column_stack([f_grid,
                           np.full_like(f_grid, f1dot_fix),
                           np.full_like(f_grid, f2dot_fix)])
    z = interp((pts - mins) / span)            # same scaling!
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f_grid, z, lw=1.2)
    ax.set_xlabel("Frequency  $f$  [Hz]")
    ax.set_ylabel(r"$2\mathcal{F}_r$")
    ax.set_title(fr"$\dot f_1={f1dot_fix:.3e}$,  $\dot f_2={f2dot_fix:.3e}$")
    plt.tight_layout()

def plot_3D(datfile):
    # ---------- load & build interpolator ------------------------------


    # ---------- Plotly setup -------------------------------------------
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"          # change to 'notebook' if you prefer

    # ---------- 1-D slice ----------------------------------------------
    f_line = np.linspace(data[:, 0].min(), data[:, 0].max(), 15_000)
    f1dot0 = -1e-10
    f2dot0 = -1e-20

    pts_1d = np.column_stack([f_line,
                              np.full_like(f_line, f1dot0),
                              np.full_like(f_line, f2dot0)])
    z_1d = interp((pts_1d - mins) / span)

    fig1 = go.Figure(go.Scatter(x=f_line, y=z_1d, mode="lines"))
    fig1.update_layout(title="2Fr vs f  ("
                              + f"f1dot = {f1dot0:.3e}, "
                              + f"f2dot = {f2dot0:.3e})",
                       xaxis_title="Frequency  f  [Hz]",
                       yaxis_title="2Fr")
    pio.write_html(fig1, "slice_1d.html", auto_open=False)

    # ---------- helper for 3-D surfaces --------------------------------
    def make_surface(xv, yv, fixed_val, fixed_idx,
                     xlabel, ylabel, title, outfile):
        """
        xv, yv : 1-D coordinate vectors that will be meshed
        fixed_idx : which coordinate (0=f, 1=f1dot, 2=f2dot) is fixed
        """
        X, Y = np.meshgrid(xv, yv, indexing="ij")

        if fixed_idx == 0:          # f fixed
            P = np.column_stack([np.full(X.size, fixed_val), X.ravel(), Y.ravel()])
        elif fixed_idx == 1:        # f1dot fixed
            P = np.column_stack([X.ravel(), np.full(X.size, fixed_val), Y.ravel()])
        else:                       # f2dot fixed
            P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, fixed_val)])

        Z = interp((P - mins) / span).reshape(X.shape)

        fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Viridis"))
        fig.update_layout(scene=dict(xaxis_title=xlabel,
                                     yaxis_title=ylabel,
                                     zaxis_title="2Fr"),
                          title=title)
        pio.write_html(fig, outfile, auto_open=False)

    NX, NY = 120, 120  # grid density for each surface

    # ---------- surface 1: f vs f1dot  (f2dot fixed) -------------------
    make_surface(
        xv=np.linspace(data[:, 0].min(), data[:, 0].max(), NX),
        yv=np.linspace(data[:, 1].min(), data[:, 1].max(), NY),
        fixed_val=f2dot0, fixed_idx=2,
        xlabel="f  [Hz]", ylabel="f1dot  [Hz/s]",
        title=f"2Fr(f, f1dot)   |   f2dot = {f2dot0:.3e}",
        outfile="surface_f_f1dot.html",
    )

    # ---------- surface 2: f vs f2dot  (f1dot fixed) -------------------
    make_surface(
        xv=np.linspace(data[:, 0].min(), data[:, 0].max(), NX),
        yv=np.linspace(data[:, 2].min(), data[:, 2].max(), NY),
        fixed_val=f1dot0, fixed_idx=1,
        xlabel="f  [Hz]", ylabel="f2dot  [Hz/s²]",
        title=f"2Fr(f, f2dot)   |   f1dot = {f1dot0:.3e}",
        outfile="surface_f_f2dot.html",
    )

    # ---------- surface 3: f1dot vs f2dot  (f fixed) -------------------
    f0 = np.median(data[:, 0])
    make_surface(
        xv=np.linspace(data[:, 1].min(), data[:, 1].max(), NX),
        yv=np.linspace(data[:, 2].min(), data[:, 2].max(), NY),
        fixed_val=f0, fixed_idx=0,
        xlabel="f1dot  [Hz/s]", ylabel="f2dot  [Hz/s²]",
        title=f"2Fr(f1dot, f2dot)   |   f = {f0:.6f} Hz",
        outfile="surface_f1dot_f2dot.html",
    )

    print("HTML files written:")
    print("  slice_1d.html")
    print("  surface_f_f1dot.html")
    print("  surface_f_f2dot.html")
    print("  surface_f1dot_f2dot.html")


# ----------------------------------------------------------------------
def predict_2Fr(freq: float, f1dot: float, f2dot: float,
                interp: LinearNDInterpolator,
                mins: np.ndarray, span: np.ndarray) -> float | np.floating:
    """
    Return the interpolated 2Fr for one parameter triple (freq, f1dot, f2dot).

    Raises
    ------
    ValueError  if the point lies outside the convex hull and
                the interpolator returns NaN.
    """
    p = np.array([freq, f1dot, f2dot], dtype=np.float64)
    p_scaled = (p - mins) / span          # same min–max scaling!
    z = interp(p_scaled)                  # returns a length-1 array
    if np.isnan(z):
        raise ValueError("query point is outside the convex hull "
                         "covered by the data – extrapolation not defined")
    return float(z)                       # plain Python float for convenience



# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 1️⃣  collect file list ------------------------------------------------
    parser = argparse.ArgumentParser(description="Interpolate HierarchSearch results")
    parser.add_argument("dat_files", nargs="+",
                        help="Path(s) or glob patterns for *.dat files")
    args = parser.parse_args()

    # Expand any globs on our own so the script works on Windows too
    paths = [str(p) for pat in args.dat_files for p in glob.glob(pat)]
    if not paths:
        raise SystemExit("No .dat files found!")

    print(f"Loading {len(paths)} file(s)…")
    data = read_data_from_files(paths)
    print(f"Total rows loaded: {len(data):,}")

    # ---- 2️⃣  build interpolator once, for *all* data --------------------------
    interp, mins, span = build_interpolator(data, n_sample=200_000)

    # ---- 3️⃣  downstream code needs **no further changes** ---------------------
    # e.g. quick sanity check:
    freq0, f1dot0, f2dot0 = 151.5, -1e-10, -1e-20
    twofr = float(interp(((freq0, f1dot0, f2dot0) - mins) / span))
    print(f"2Fr({freq0}, {f1dot0}, {f2dot0}) = {twofr:.6g}")
    
    
    duration = 120 * 86400  # 120 days
    tStack = 20 * 86400  # 15 day coherent segments
    nStacks = int(duration / tStack)  # Number of segments
    mf = 0.1
    mf1 = 0.3
    mf2 = 0.01
    gamma1 = 6
    gamma2 = 69
    dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
    dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2) / gamma1
    df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3) / gamma2
    
    N = 1000
    # Single random offset (instead of array)
    F0_random = np.random.uniform(- dF0 / 2.0, dF0 / 2.0, size=N)
    F1_random = np.random.uniform(- dF1 / 2.0, dF1 / 2.0, size=N)
    F2_random = np.random.uniform(- df2 / 2.0, df2 / 2.0, size=N)
    
    # search
    F0_inj = 151.5
    F1_inj = -1e-10
    F2_inj = -1e-20
    DeltaF0 = 2 * dF0
    DeltaF1 = 3 * dF1 
    DeltaF2 = 3 * df2
    
    F0_min = F0_inj - DeltaF0 / 2.0 + F0_random
    F1_min = F1_inj - DeltaF1 / 2.0 + F1_random
    F2_min = F2_inj - DeltaF2 / 2.0 + F2_random
    
    def one_run():
        pass
    
    # for i in range(N):
    # ---- parameters that may come from elsewhere -------------------------
    MAX_CHUNK = 250_000          # how many points to pass to interp at once

    
    max_twofs = []
    for i in range(len(F0_min)):
        f0_min, f1dot_min, f2dot_min = F0_min[i], F1_min[i], F2_min[i]

        f0_vals   = np.arange(f0_min,   f0_min   + DeltaF0, dF0)
        f1dot_vals = np.arange(f1dot_min, f1dot_min + DeltaF1, dF1)
        f2dot_vals = np.arange(f2dot_min, f2dot_min + DeltaF2, df2)

        # -------- vectorised Cartesian product ----------------------------
        F0g, F1g, F2g = np.meshgrid(f0_vals, f1dot_vals, f2dot_vals,
                                    indexing="ij", copy=False)

        pts = np.column_stack([F0g.ravel(), F1g.ravel(), F2g.ravel()])
        n_tot = len(pts)

        # -------- optionally process in manageable chunks -----------------
        twofr_vals = np.empty(n_tot, dtype=np.float64)
        for k in range(0, n_tot, MAX_CHUNK):
            sl = slice(k, k + MAX_CHUNK)
            twofr_vals[sl] = interp((pts[sl] - mins) / span)

        # reshape back to (len(f0_vals), len(f1dot_vals), len(f2dot_vals))
        twofr_grid = twofr_vals.reshape(F0g.shape)

        # now you have the full 3-D block in one go – do whatever you need,
        # e.g. locate the maximum:
        best_idx   = np.nanargmax(twofr_grid)
        best_f0    = F0g.ravel()[best_idx]
        best_f1dot = F1g.ravel()[best_idx]
        best_f2dot = F2g.ravel()[best_idx]
        best_val   = twofr_vals[best_idx]
        # print(f"[{i}]  max 2Fr = {best_val:.6g} "
        #     f"at (f0={best_f0:.6f}, f1dot={best_f1dot:.3e}, f2dot={best_f2dot:.3e})")
        
        max_twofs.append((best_val, best_f0, best_f1dot, best_f2dot))
        
    max_twofs = np.array(max_twofs)
    # predict_2Frs = 
    twofr_perf = 104726848
    mismatch = (twofr_perf - max_twofs[:, 0]) / (twofr_perf - 4)
    
    plt.hist(mismatch)
    plt.show()
    print(f"Mean mismatch: {np.mean(mismatch):.3f}")
    # print(max_twofs)