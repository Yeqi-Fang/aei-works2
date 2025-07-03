import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Create output directory
print_output = False  # Set to False to suppress output
label = "LALSemiCoherentF0F1F2_analytical"
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
mf = 0.1
mf1 = 0.02
mf2 = 0.001
dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

# Search bands
N1 = 40
N2 = 20
N3 = 15
gamma1 = 19
gamma2 = 69

DeltaF0 = N1 * dF0
DeltaF1 = N2 * dF1
DeltaF2 = N3 * df2

# Single random offset (instead of array)
F0_random = np.random.uniform(- dF0 / 2.0, dF0 / 2.0)
F1_random = np.random.uniform(- dF1 / 2.0, dF1 / 2.0)
F2_random = np.random.uniform(- df2 / 2.0, df2 / 2.0)

shared_cmd = [
    f"--DataFiles1={sft_pattern}",
    "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
    f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
    f"--refTime={tref:.15f}",
    f"--tStack={tStack:.15g}",
    f"--nStacksMax={nStacks}",
    "--nCand1=10000000",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    "--recalcToplistStats=TRUE",
    "--FstatMethod=DemodBest",
    "--FstatMethodRecalc=DemodOptC",
    "--Dterms=8",
]

# Single run (no loop)
F0_min = F0_inj - DeltaF0 / 2.0 
F1_min = F1_inj - DeltaF1 / 2.0 
F2_min = F2_inj - DeltaF2 / 2.0 

output_file = os.path.join(outdir, "dats/semicoh_results.dat")

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

print("Running semi-coherent search...")
result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running HierarchSearchGCT:")
    print(f"stderr: {result.stderr}")
    print(f"stdout: {result.stdout}")
    raise RuntimeError("Failed to run semi-coherent search")

# Process single result file
if not os.path.exists(output_file):
    print(f"Output file {output_file} does not exist.")
    raise RuntimeError("No output file generated")

with open(output_file, 'r') as f:
    lines = f.readlines()

# Look for the data section
data = []
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
    
    print(f"Maximum 2F: {max_twoF}")
    print(f"At F0: {max_F0}, F1: {max_F1}, F2: {max_F2}")

# Calculate perfectly matched result
perfect_output_file = os.path.join(outdir, "perfectly_matched_results.dat")

perfect_search_cmd = [
    "lalpulsar_HierarchSearchGCT",
    f"--Freq={F0_inj:.15g}",
    "--FreqBand=0",
    f"--dFreq={dF0:.15e}",
    f"--f1dot={F1_inj:.15e}",
    "--f1dotBand=0",
    f"--df1dot={dF1:.15e}",
    f"--f2dot={F2_inj:.15e}",
    "--f2dotBand=0",
    f"--df2dot={df2:.15e}",
    f"--fnameout={perfect_output_file}",
    f"--gammaRefine={gamma1:.15g}",
    f"--gamma2Refine={gamma2:.15g}",
] + shared_cmd

print("Running perfectly matched search...")
result = subprocess.run(perfect_search_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running lalpulsar_HierarchSearchGCT: {result.stderr}")
    raise RuntimeError("Failed to compute perfectly matched 2F value")

with open(perfect_output_file, 'r') as f:
    lines = f.readlines()

# Process perfectly matched result
data = []
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
                f3dot = float(parts[5])
                twoF = float(parts[6])
                twoFr = float(parts[7])
                data.append([freq, f1dot, f2dot, twoFr])
            except ValueError:
                continue

if data:
    data = np.array(data)
    twoF_vals = data[:, 3]
    max_idx = np.argmax(twoF_vals)
    perfect_2F = twoF_vals[max_idx]
    
    print(f"Perfect match 2F: {perfect_2F}")

# Calculate mismatch
mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)
print(f"Mismatch: {mismatch}")

# Create simple plot (single value)
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar([0], [mismatch], width=0.5, color="#5B9BD5", alpha=0.85, edgecolor="black")
ax.set_xlabel("Single Run", fontsize=16)
ax.set_ylabel("Mismatch μ", fontsize=16)
ax.set_title(f"Single Run Mismatch: {mismatch:.4f}", fontsize=14)
ax.set_ylim(0, max(1, mismatch * 1.2))
ax.grid(axis="y", linewidth=0.6, alpha=0.35)

fig.tight_layout()
fig.savefig(f"images/single_mismatch-{mf}-{mf1}-{mf2}-{depth}.pdf")
# plt.show()

# Create Corner Plot for 2F vs dF0, dF1, dF2
print("\nCreating corner plot...")

# Re-read the data to get all points (not just maximum)
with open(output_file, 'r') as f:
    lines = f.readlines()

# Parse all data points
all_data = []
for line in lines:
    if line.strip() and not line.startswith('%'):
        parts = line.split()
        if len(parts) >= 8:
            try:
                freq = float(parts[0])
                alpha = float(parts[1])
                delta = float(parts[2])
                f1dot = float(parts[3])
                f2dot = float(parts[4])
                nc = float(parts[5])
                twoF = float(parts[6])
                twoFr = float(parts[7])
                all_data.append([freq, f1dot, f2dot, twoFr])
            except ValueError:
                continue

if all_data:
    all_data = np.array(all_data)
    all_F0 = all_data[:, 0]
    all_F1 = all_data[:, 1]
    all_F2 = all_data[:, 2]
    all_twoF = all_data[:, 3]
    
    # Calculate offsets from injected values (these are our dF0, dF1, dF2)
    dF0_vals = all_F0 - F0_inj
    dF1_vals = all_F1 - F1_inj
    dF2_vals = all_F2 - F2_inj
    
    print(f"Found {len(all_data)} data points")
    print(f"2F range: {np.min(all_twoF):.3f} - {np.max(all_twoF):.3f}")
    print(f"dF0 range: {np.min(dF0_vals):.2e} - {np.max(dF0_vals):.2e}")
    print(f"dF1 range: {np.min(dF1_vals):.2e} - {np.max(dF1_vals):.2e}")
    print(f"dF2 range: {np.min(dF2_vals):.2e} - {np.max(dF2_vals):.2e}")
    
    # Create corner plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    # Define labels and data
    labels = ['dF0 (Hz)', 'dF1 (Hz/s)', 'dF2 (Hz/s²)']
    data_arrays = [dF0_vals, dF1_vals, dF2_vals]
    
    # Main diagonal: 2F vs each parameter
    for i in range(3):
        ax = axes[i, i]
        scatter = ax.scatter(data_arrays[i], all_twoF, c=all_twoF, 
                           cmap='viridis', alpha=0.7, s=20)
        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel('2F', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for the middle plot
        if i == 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('2F', rotation=270, labelpad=15)
    
    # Off-diagonal: parameter correlations with 2F as color
    for i in range(3):
        for j in range(3):
            if i != j:
                ax = axes[i, j]
                scatter = ax.scatter(data_arrays[j], data_arrays[i], 
                                   c=all_twoF, cmap='viridis', alpha=0.7, s=20)
                ax.set_xlabel(labels[j], fontsize=12)
                ax.set_ylabel(labels[i], fontsize=12)
                ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    plt.tight_layout()
    
    # Save corner plot
    corner_filename = f"images/corner_plot_2F-{mf}-{mf1}-{mf2}-{depth}.pdf"
    fig.savefig(corner_filename, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved as: {corner_filename}")
    # plt.show()
    
    # Create a separate 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(dF0_vals, dF1_vals, dF2_vals, c=all_twoF, 
                         cmap='viridis', alpha=0.8, s=30)
    ax1.set_xlabel('dF0 (Hz)', fontsize=12)
    ax1.set_ylabel('dF1 (Hz/s)', fontsize=12)
    ax1.set_zlabel('dF2 (Hz/s²)', fontsize=12)
    ax1.set_title('3D Parameter Space (colored by 2F)', fontsize=12)
    
    # 2F surface plot
    ax2 = fig.add_subplot(122)
    # Create a simple 2D projection of 2F
    im = ax2.tricontourf(dF0_vals, dF1_vals, all_twoF, levels=20, cmap='viridis')
    ax2.set_xlabel('dF0 (Hz)', fontsize=12)
    ax2.set_ylabel('dF1 (Hz/s)', fontsize=12)
    ax2.set_title('2F Contour (dF0 vs dF1)', fontsize=12)
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('2F', rotation=270, labelpad=15)
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('2F', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save 3D plot
    plot3d_filename = f"images/3d_plot_2F-{mf}-{mf1}-{mf2}-{depth}.pdf"
    fig.savefig(plot3d_filename, dpi=300, bbox_inches='tight')
    print(f"3D plot saved as: {plot3d_filename}")
    # plt.show()
    
    # Create three separate 3D surface plots
    try:
        from scipy.interpolate import griddata
        scipy_available = True
    except ImportError:
        print("Warning: scipy not available, using scatter plots instead of surfaces")
        scipy_available = False
    
    # Function to create 3D surface plot
    def create_3d_surface(x_data, y_data, z_data, x_label, y_label, title, filename):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if scipy_available:
            # Create grid for interpolation
            xi = np.linspace(np.min(x_data), np.max(x_data), 50)
            yi = np.linspace(np.min(y_data), np.max(y_data), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate Z values
            Zi = griddata((x_data, y_data), z_data, (Xi, Yi), method='cubic', fill_value=np.nan)
            
            # Create surface plot
            surface = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8, 
                                     linewidth=0, antialiased=True)
            
            # Add colorbar for surface
            cbar = plt.colorbar(surface, ax=ax, shrink=0.8)
            cbar.set_label('2F', rotation=270, labelpad=15)
        
        # Add scatter points (always show these)
        scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', 
                           s=30, alpha=0.9, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel('2F', fontsize=12)
        ax.set_title(f"{title} {'(Surface + Scatter)' if scipy_available else '(Scatter Only)'}", fontsize=14)
        
        # Add colorbar for scatter if no surface
        if not scipy_available:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('2F', rotation=270, labelpad=15)
        
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"3D {'surface' if scipy_available else 'scatter'} plot saved as: {filename}")
        # plt.show()
        
        return fig
    
    print(f"\nCreating 3D {'surface' if scipy_available else 'scatter'} plots...")
    
    # Plot 1: 2F(dF0, dF1)
    create_3d_surface(
        dF0_vals, dF1_vals, all_twoF,
        'dF0 (Hz)', 'dF1 (Hz/s)',
        '2F Surface: 2F(dF0, dF1)',
        f"images/3d_surface_2F_dF0_dF1-{mf}-{mf1}-{mf2}-{depth}.pdf"
    )
    
    # Plot 2: 2F(dF1, dF2)
    create_3d_surface(
        dF1_vals, dF2_vals, all_twoF,
        'dF1 (Hz/s)', 'dF2 (Hz/s²)',
        '2F Surface: 2F(dF1, dF2)',
        f"images/3d_surface_2F_dF1_dF2-{mf}-{mf1}-{mf2}-{depth}.pdf"
    )
    
    # Plot 3: 2F(dF0, dF2)
    create_3d_surface(
        dF0_vals, dF2_vals, all_twoF,
        'dF0 (Hz)', 'dF2 (Hz/s²)',
        '2F Surface: 2F(dF0, dF2)',
        f"images/3d_surface_2F_dF0_dF2-{mf}-{mf1}-{mf2}-{depth}.pdf"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Search parameters:")
    print(f"  mf = {mf}, mf1 = {mf1}, mf2 = {mf2}")
    print(f"  Depth = {depth}")
    print(f"  N1 = {N1}, N2 = {N2}, N3 = {N3}")
    print(f"\nResults:")
    print(f"  Total data points analyzed: {len(all_data)}")
    print(f"  Maximum 2F found: {np.max(all_twoF):.3f}")
    print(f"  Perfect match 2F: {perfect_2F:.3f}")
    print(f"  Mismatch: {mismatch:.6f}")
    print(f"\nParameter space coverage:")
    print(f"  dF0 range: [{np.min(dF0_vals):.2e}, {np.max(dF0_vals):.2e}] Hz")
    print(f"  dF1 range: [{np.min(dF1_vals):.2e}, {np.max(dF1_vals):.2e}] Hz/s")
    print(f"  dF2 range: [{np.min(dF2_vals):.2e}, {np.max(dF2_vals):.2e}] Hz/s²")
    print(f"\nGenerated plots:")
    print(f"  - Corner plot: corner_plot_2F-{mf}-{mf1}-{mf2}-{depth}.pdf")
    print(f"  - 3D overview: 3d_plot_2F-{mf}-{mf1}-{mf2}-{depth}.pdf")
    print(f"  - 3D surface plots:")
    print(f"    * 2F(dF0,dF1): 3d_surface_2F_dF0_dF1-{mf}-{mf1}-{mf2}-{depth}.pdf")
    print(f"    * 2F(dF1,dF2): 3d_surface_2F_dF1_dF2-{mf}-{mf1}-{mf2}-{depth}.pdf")
    print(f"    * 2F(dF0,dF2): 3d_surface_2F_dF0_dF2-{mf}-{mf1}-{mf2}-{depth}.pdf")
    print("="*60)
    
else:
    print("No data found for corner plot")