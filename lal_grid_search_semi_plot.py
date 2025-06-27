import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn
from matplotlib import gridspec


# Create output directory
N = 500
print_output = False  # Set to False to suppress output
label = "LALSemiCoherentF0F1F2_corrected_fast_plot"
outdir = os.path.join("LAL_example_data", label)
os.makedirs(outdir, exist_ok=True)

# Properties of the GW data
sqrtSX = 1e-22
tstart = 1126051217
duration = 120 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFO = "H1"  # Interferometers to use

# Parameters for injected signals
depth = 0.02
h0 = sqrtSX / depth
F0_inj = 151.5
F1_inj = -1e-10
F2_inj = -1e-20
Alpha_inj = 0.5
Delta_inj = 1
cosi_inj = 1
psi_inj = 0.0
phi0_inj = 0.0

# Semi-coherent search parameters
tStack = 20 * 86400  # 15 day coherent segments
nStacks = int(duration / tStack)  # Number of segments

# Step 1: Generate SFT data
# print("Generating SFT data with injected signal...")

sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)
os.makedirs(os.path.join(outdir, "dats"), exist_ok=True)
os.makedirs(os.path.join(outdir, "commands"), exist_ok=True)
sft_pattern = os.path.join(sft_dir, "*.sft")

injection_params = (
    f"{{Alpha={Alpha_inj:.15g}; Delta={Delta_inj:.15g}; Freq={F0_inj:.15g}; "
    f"f1dot={F1_inj:.15e}; f2dot={F2_inj:.15e}; refTime={tref:.15g}; "
    f"h0={h0:.15e}; cosi={cosi_inj:.15g}; psi={psi_inj:.15g}; phi0={phi0_inj:.15g};}}"
)

sft_label = "SemiCoh"

makefakedata_cmd = [
    "lalpulsar_Makefakedata_v5",
    f"--IFOs={IFO}",
    f"--sqrtSX={sqrtSX:.15e}",
    f"--startTime={int(tstart)}",
    f"--duration={int(duration)}",
    f"--fmin={F0_inj - 0.1:.15g}",
    f"--Band=0.2",
    "--Tsft=1800",
    f"--outSFTdir={sft_dir}",
    f"--outLabel={sft_label}",
    f"--injectionSources={injection_params}",
    "--randSeed=1234"
]

result = subprocess.run(makefakedata_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error generating SFTs: {result.stderr}")
    raise RuntimeError("Failed to generate SFTs")

# Step 2: Set up grid search parameters
mf = 0.01
mf1 = 0.001
mf2 = 0.08
dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

# Search bands
N1 = 2
N2 = 3
N3 = 3
gamma1 = 11
gamma2 = 91

DeltaF0 = N1 * dF0
DeltaF1 = N2 * dF1
DeltaF2 = N3 * df2

F0_randoms = np.random.uniform(- dF0 / 2.0, dF0 / 2.0, size=N)
F1_randoms = np.random.uniform(- dF1 / 2.0, dF1 / 2.0, size=N)
F2_randoms = np.random.uniform(- df2 / 2.0, df2 / 2.0, size=N)

shared_cmd = [
    f"--DataFiles1={sft_pattern}",
    "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
    f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
    f"--refTime={tref:.15f}",
    f"--tStack={tStack:.15g}",
    f"--nStacksMax={nStacks}",
    "--nCand1=1000",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    "--recalcToplistStats=TRUE",
    "--FstatMethod=DemodBest",
    "--FstatMethodRecalc=DemodOptC",
    "--Dterms=8",
]

def single_run(i):
    F0_min = F0_inj - DeltaF0 / 2.0 + F0_randoms[i]
    F0_max = F0_inj + DeltaF0 / 2.0 + F0_randoms[i]
    F1_min = F1_inj - DeltaF1 / 2.0 + F1_randoms[i]
    F1_max = F1_inj + DeltaF1 / 2.0 + F1_randoms[i]
    F2_min = F2_inj - DeltaF2 / 2.0 + F2_randoms[i]
    F2_max = F2_inj + DeltaF2 / 2.0 + F2_randoms[i]

    output_file = os.path.join(outdir, f"dats/semicoh_results_{i}.dat")
    
    # Build command with proper formatting
    hierarchsearch_cmd = [
        "lalpulsar_HierarchSearchGCT",
        f"--fnameout={output_file}",
        f"--Freq={F0_min:.15g}",
        f"--FreqBand={DeltaF0:.15g}",
        f"--dFreq={dF0:.15e}",
        f"--f1dot={F1_min:.15e}",
        f"--f1dotBand={DeltaF1:.15e}",
        f"--df1dot={dF1:.15e}",
        f"--f2dot={F2_min:.15e}",
        f"--f2dotBand={DeltaF2:.15e}",
        f"--df2dot={df2:.15e}",
        f"--gammaRefine={gamma1:.15g}",
        f"--gamma2Refine={gamma2:.15g}",
    ] + shared_cmd

    result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running HierarchSearchGCT:")
        print(f"stderr: {result.stderr}")
        print(f"stdout: {result.stdout}")
        raise RuntimeError("Failed to run semi-coherent search")

# Run the grid searches
with Progress(
    "[progress.description]{task.description}",
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    "[progress.completed]{task.completed}/{task.total}",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
) as progress:
    
    task = progress.add_task("Processing runs", total=N)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_run, i) for i in range(N)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Wait for each future to complete
                progress.advance(task, 1)
            except Exception as e:
                print(f"Error in thread: {e}")
                progress.advance(task, 1)

# Extract maximum 2F values from each run
results = []
max_twoFs = []

for i in range(N):
    output_file = os.path.join(outdir, f"dats/semicoh_results_{i}.dat")
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist, skipping.")
        continue
    with open(output_file, 'r') as f:
        lines = f.readlines()
    # Look for the data section
        data = []
        in_data = False
        for line in lines:
            if line.strip() and not line.startswith('%'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        freq = float(parts[0])
                        alpha = float(parts[1])
                        delta = float(parts[2])
                        f1dot = float(parts[3])
                        f2dot = float(parts[4])
                        nc = float(parts[5])
                        twoF = float(parts[6])
                        twoFr = float(parts[7])
                        data.append([freq, f1dot, f2dot, twoFr])
                    except ValueError:
                        continue

        if data:
            data = np.array(data)
            F0_vals = data[:, 0]
            F1_vals = data[:, 1]
            F2_vals = data[:, 2]
            twoF_vals = data[:, 3]
            
            # Find maximum
            max_idx = np.argmax(twoF_vals)
            max_twoF = twoF_vals[max_idx]
            max_F0 = F0_vals[max_idx]
            max_F1 = F1_vals[max_idx]
            max_F2 = F2_vals[max_idx]

    max_twoFs.append(max_twoF)

# Run a perfect match search for plotting
print("Running perfect match search for detailed plotting...")
perfect_output_file = os.path.join(outdir, "perfectly_matched_results.dat")

F0_min = F0_inj - DeltaF0 / 2.0
F0_max = F0_inj + DeltaF0 / 2.0
F1_min = F1_inj - DeltaF1 / 2.0
F1_max = F1_inj + DeltaF1 / 2.0
F2_min = F2_inj - DeltaF2 / 2.0
F2_max = F2_inj + DeltaF2 / 2.0

perfect_search_cmd = [
    "lalpulsar_HierarchSearchGCT",
    f"--Freq={F0_min:.15g}",
    f"--FreqBand={DeltaF0:.15g}",
    f"--dFreq={dF0:.15e}",
    f"--f1dot={F1_min:.15e}",
    f"--f1dotBand={DeltaF1:.15e}",
    f"--df1dot={dF1:.15e}",
    f"--f2dot={F2_min:.15e}",
    f"--f2dotBand={DeltaF2:.15e}",
    f"--df2dot={df2:.15e}",
    f"--fnameout={perfect_output_file}",
    f"--gammaRefine={gamma1:.15g}",
    f"--gamma2Refine={gamma2:.15g}",
] + shared_cmd 

# Run command and capture output
result = subprocess.run(perfect_search_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running lalpulsar_HierarchSearchGCT: {result.stderr}")
    raise RuntimeError("Failed to compute perfectly matched 2F value")

# Parse the perfect match results for plotting
with open(perfect_output_file, 'r') as f:
    lines = f.readlines()

# Extract all data points for plotting
plot_data = []
for line in lines:
    if line.strip() and not line.startswith('%'):
        parts = line.split()
        if len(parts) >= 7:
            try:
                freq = float(parts[0])
                alpha = float(parts[1])
                delta = float(parts[2])
                f1dot = float(parts[3])
                f2dot = float(parts[4])
                nc = float(parts[5])
                twoF = float(parts[6])
                twoFr = float(parts[7])
                plot_data.append([freq, f1dot, f2dot, twoFr])
            except ValueError:
                continue

if plot_data:
    plot_data = np.array(plot_data)
    F0_vals = plot_data[:, 0]
    F1_vals = plot_data[:, 1]
    F2_vals = plot_data[:, 2]
    twoF_vals = plot_data[:, 3]
    
    # Find maximum for mismatch calculation
    max_idx = np.argmax(twoF_vals)
    perfect_2F = twoF_vals[max_idx]
    
    print(f"Perfect match 2F = {perfect_2F:.4f}")
    print(f"Found {len(plot_data)} grid points for plotting")
    
    # Create detailed plots similar to the second file
    print("Creating detailed 2F parameter plots...")
    
    # 1D plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 2F vs F0 (at injection F1, F2)
    mask_F1F2 = (np.abs(F1_vals - F1_inj) < dF1/2) & (np.abs(F2_vals - F2_inj) < df2/2)
    if np.any(mask_F1F2):
        F0_slice = F0_vals[mask_F1F2]
        twoF_slice = twoF_vals[mask_F1F2]
        sort_idx = np.argsort(F0_slice)
        axes[0].plot(F0_slice[sort_idx], twoF_slice[sort_idx], 'b-', linewidth=2)
    axes[0].axvline(F0_inj, color='r', linestyle='--', linewidth=2, label='Injection')
    axes[0].set_xlabel('Frequency [Hz]', fontsize=14)
    axes[0].set_ylabel('$2\\mathcal{F}$', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)
    
    # Plot 2F vs F1 (at injection F0, F2)
    mask_F0F2 = (np.abs(F0_vals - F0_inj) < dF0/2) & (np.abs(F2_vals - F2_inj) < df2/2)
    if np.any(mask_F0F2):
        F1_slice = F1_vals[mask_F0F2]
        twoF_slice = twoF_vals[mask_F0F2]
        sort_idx = np.argsort(F1_slice)
        axes[1].plot(F1_slice[sort_idx], twoF_slice[sort_idx], 'b-', linewidth=2)
    axes[1].axvline(F1_inj, color='r', linestyle='--', linewidth=2, label='Injection')
    axes[1].set_xlabel('$\\dot{f}$ [Hz/s]', fontsize=14)
    axes[1].set_ylabel('$2\\mathcal{F}$', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)
    
    # Plot 2F vs F2 (at injection F0, F1)
    mask_F0F1 = (np.abs(F0_vals - F0_inj) < dF0/2) & (np.abs(F1_vals - F1_inj) < dF1/2)
    if np.any(mask_F0F1):
        F2_slice = F2_vals[mask_F0F1]
        twoF_slice = twoF_vals[mask_F0F1]
        sort_idx = np.argsort(F2_slice)
        axes[2].plot(F2_slice[sort_idx], twoF_slice[sort_idx], 'b-', linewidth=2)
    axes[2].axvline(F2_inj, color='r', linestyle='--', linewidth=2, label='Injection')
    axes[2].set_xlabel('$\\ddot{f}$ [Hz/s$^2$]', fontsize=14)
    axes[2].set_ylabel('$2\\mathcal{F}$', fontsize=14)
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "1D_projections_semicoh.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create corner plot if we have enough data points
    print("Creating corner plot...")
    
    # Get unique values for each parameter
    F0_unique = np.sort(np.unique(F0_vals))
    F1_unique = np.sort(np.unique(F1_vals))
    F2_unique = np.sort(np.unique(F2_vals))
    
    # Shifted values for plotting
    F0_shifted = F0_unique - F0_inj
    F1_shifted = F1_unique - F1_inj
    F2_shifted = F2_unique - F2_inj
    
    if len(F0_unique) > 1 and len(F1_unique) > 1 and len(F2_unique) > 1:
        # Reshape twoF values into 3D array
        n_F0 = len(F0_unique)
        n_F1 = len(F1_unique)
        n_F2 = len(F2_unique)
        
        # Create a mapping from parameter values to indices
        tolerance = 1e-10  # Small tolerance for floating point comparison
        def find_index(val, unique_vals):
            idx = np.argmin(np.abs(unique_vals - val))
            return idx
        
        # Initialize 3D array
        twoF_3D = np.zeros((n_F0, n_F1, n_F2))
        
        # Fill the 3D array
        for i in range(len(F0_vals)):
            i_F0 = find_index(F0_vals[i], F0_unique)
            i_F1 = find_index(F1_vals[i], F1_unique)
            i_F2 = find_index(F2_vals[i], F2_unique)
            twoF_3D[i_F0, i_F1, i_F2] = twoF_vals[i]
        
        # Create corner plot
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.1, wspace=0.1)
        
        # 2D projections
        # F0-F1
        ax_F0F1 = fig.add_subplot(gs[1, 0])
        Z_F0F1 = np.max(twoF_3D, axis=2)
        im = ax_F0F1.imshow(Z_F0F1.T, origin='lower', aspect='auto',
                            extent=[F0_shifted[0], F0_shifted[-1], F1_shifted[0], F1_shifted[-1]],
                            cmap='viridis')
        ax_F0F1.set_xlabel('$f - f_0$ [Hz]', fontsize=14)
        ax_F0F1.set_ylabel('$\\dot{f} - \\dot{f}_0$ [Hz/s]', fontsize=14)
        ax_F0F1.tick_params(labelsize=12)
        
        # F0-F2
        ax_F0F2 = fig.add_subplot(gs[2, 0])
        Z_F0F2 = np.max(twoF_3D, axis=1)
        ax_F0F2.imshow(Z_F0F2.T, origin='lower', aspect='auto',
                       extent=[F0_shifted[0], F0_shifted[-1], F2_shifted[0], F2_shifted[-1]],
                       cmap='viridis')
        ax_F0F2.set_xlabel('$f - f_0$ [Hz]', fontsize=14)
        ax_F0F2.set_ylabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]', fontsize=14)
        ax_F0F2.tick_params(labelsize=12)
        
        # F1-F2
        ax_F1F2 = fig.add_subplot(gs[2, 1])
        Z_F1F2 = np.max(twoF_3D, axis=0)
        ax_F1F2.imshow(Z_F1F2.T, origin='lower', aspect='auto',
                       extent=[F1_shifted[0], F1_shifted[-1], F2_shifted[0], F2_shifted[-1]],
                       cmap='viridis')
        ax_F1F2.set_xlabel('$\\dot{f} - \\dot{f}_0$ [Hz/s]', fontsize=14)
        ax_F1F2.set_ylabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]', fontsize=14)
        ax_F1F2.tick_params(labelsize=12)
        
        # 1D projections
        # F0
        ax_F0 = fig.add_subplot(gs[0, 0])
        ax_F0.plot(F0_shifted, np.max(np.max(twoF_3D, axis=2), axis=1), 'b-', linewidth=2)
        ax_F0.set_ylabel('$2\\mathcal{F}$', fontsize=14)
        ax_F0.set_xticklabels([])
        ax_F0.tick_params(labelsize=12)
        ax_F0.grid(True, alpha=0.3)
        
        # F1
        ax_F1 = fig.add_subplot(gs[1, 1])
        ax_F1.plot(F1_shifted, np.max(np.max(twoF_3D, axis=2), axis=0), 'b-', linewidth=2)
        ax_F1.set_ylabel('$2\\mathcal{F}$', fontsize=14)
        ax_F1.set_xticklabels([])
        ax_F1.tick_params(labelsize=12)
        ax_F1.grid(True, alpha=0.3)
        
        # F2
        ax_F2 = fig.add_subplot(gs[2, 2])
        ax_F2.plot(F2_shifted, np.max(np.max(twoF_3D, axis=1), axis=0), 'b-', linewidth=2)
        ax_F2.set_xlabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]', fontsize=14)
        ax_F2.set_ylabel('$2\\mathcal{F}$', fontsize=14)
        ax_F2.tick_params(labelsize=12)
        ax_F2.grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=[ax_F0F1, ax_F0F2, ax_F1F2], label='$2\\mathcal{F}$')
        plt.savefig(os.path.join(outdir, "grid_corner_plot_semicoh.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Corner plot created successfully!")
    else:
        print("Not enough unique parameter values for corner plot")

# Calculate mismatches and create histogram
mismatches = []
for i in range(N):
    max_twoF = max_twoFs[i]
    mismatches.append((perfect_2F - max_twoF) / (perfect_2F - 4))

# Create mismatch histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    mismatches,
    bins=10,
    density=True,
    color="#5B9BD5",  # pleasant blue
    alpha=0.85,
    edgecolor="black",
    linewidth=1.0,
)

# axis labels & limits
ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
ax.set_ylabel("normalized histogram", fontsize=20)
ax.set_xlim(0, 1)

# ticks & grid
ax.tick_params(axis="both", which="major", labelsize=14, length=6)
ax.grid(axis="y", linewidth=0.6, alpha=0.35)

fig.tight_layout()
fig.savefig(f"images/mismatch_distribution_lal-{mf}-{mf1}-{mf2}-{N}-{depth}.pdf")
fig.savefig(os.path.join(outdir, "mismatch_distribution.png"), dpi=300, bbox_inches='tight')

print("mean mismatch: ", np.mean(mismatches))
print(f"\nAll plots saved to {outdir}")
print("Created files:")
print("  - 1D_projections_semicoh.png")
print("  - grid_corner_plot_semicoh.png") 
print("  - mismatch_distribution.png")