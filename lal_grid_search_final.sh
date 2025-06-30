#!/bin/bash

# Create output directory
N=100
label="LALSemiCoherentF0F1F2_corrected_fast2"
outdir="LAL_example_data/${label}"
mkdir -p "${outdir}"

# Properties of the GW data
sqrtSX="1e-22"
tstart=1126051217
duration=$((120 * 86400))
tend=$((tstart + duration))
tref=$(awk -v t1="$tstart" -v t2="$tend" 'BEGIN{printf "%.15f", 0.5 * (t1 + t2)}')
IFO="H1"  # Interferometers to use

# Parameters for injected signals
depth="0.02"
h0=$(awk -v s="$sqrtSX" -v d="$depth" 'BEGIN{printf "%.15e", s / d}')
F0_inj="151.5"
F1_inj="-1e-10"
F2_inj="-1e-20"
Alpha_inj="0.5"
Delta_inj="1"
cosi_inj="1"
psi_inj="0.0"
phi0_inj="0.0"

# Semi-coherent search parameters
tStack=$((15 * 86400))  # 15 day coherent segments
nStacks=$((duration / tStack))  # Number of segments

# Step 1: Generate SFT data
echo "Generating SFT data with injected signal..."

sft_dir="${outdir}/sfts"
mkdir -p "${sft_dir}"
mkdir -p "${outdir}/dats"
mkdir -p "${outdir}/commands"
sft_pattern="${sft_dir}/*.sft"

injection_params="{Alpha=${Alpha_inj}; Delta=${Delta_inj}; Freq=${F0_inj}; f1dot=${F1_inj}; f2dot=${F2_inj}; refTime=${tref}; h0=${h0}; cosi=${cosi_inj}; psi=${psi_inj}; phi0=${phi0_inj};}"

sft_label="SemiCoh"

# Calculate fmin
fmin=$(awk -v f="$F0_inj" 'BEGIN{printf "%.15g", f - 0.1}')

lalpulsar_Makefakedata_v5 \
    --IFOs="${IFO}" \
    --sqrtSX="${sqrtSX}" \
    --startTime="${tstart}" \
    --duration="${duration}" \
    --fmin="${fmin}" \
    --Band=0.2 \
    --Tsft=1800 \
    --outSFTdir="${sft_dir}" \
    --outLabel="${sft_label}" \
    --injectionSources="${injection_params}" \
    --randSeed=1234

if [ $? -ne 0 ]; then
    echo "Error generating SFTs"
    exit 1
fi

# Step 3: Set up grid search parameters
mf="0.35"
mf1="0.5"
mf2="0.01"

# Calculate grid spacings using awk for floating point math
# Note: Using 3.14159265359 as approximation for pi
dF0=$(awk -v mf="$mf" -v ts="$tStack" 'BEGIN{pi=3.14159265359; printf "%.15e", sqrt(12 * mf) / (pi * ts)}')
dF1=$(awk -v mf1="$mf1" -v ts="$tStack" 'BEGIN{pi=3.14159265359; printf "%.15e", sqrt(180 * mf1) / (pi * ts * ts)}')
df2=$(awk -v mf2="$mf2" -v ts="$tStack" 'BEGIN{pi=3.14159265359; printf "%.15e", sqrt(25200 * mf2) / (pi * ts * ts * ts)}')

# Search bands
N1=2
N2=3
N3=3
gamma1=9
gamma2=69

DeltaF0=$(awk -v n="$N1" -v d="$dF0" 'BEGIN{printf "%.15e", n * d}')
DeltaF1=$(awk -v n="$N2" -v d="$dF1" 'BEGIN{printf "%.15e", n * d}')
DeltaF2=$(awk -v n="$N3" -v d="$df2" 'BEGIN{printf "%.15e", n * d}')

# Generate random offsets
echo "Generating random offsets..."
for ((i=0; i<N; i++)); do
    # Generate random numbers between -0.5 and 0.5, then scale
    F0_randoms[$i]=$(awk -v seed="$RANDOM$i" -v d="$dF0" 'BEGIN{srand(seed); printf "%.15e", (rand()-0.5) * d}')
    F1_randoms[$i]=$(awk -v seed="$RANDOM$i" -v d="$dF1" 'BEGIN{srand(seed+1000); printf "%.15e", (rand()-0.5) * d}')
    F2_randoms[$i]=$(awk -v seed="$RANDOM$i" -v d="$df2" 'BEGIN{srand(seed+2000); printf "%.15e", (rand()-0.5) * d}')
done

# Function to run a single search
run_single_search() {
    local i=$1
    
    # Access the exported random values
    local F0_rand_var="F0_random_$i"
    local F1_rand_var="F1_random_$i"
    local F2_rand_var="F2_random_$i"
    
    local F0_rand="${!F0_rand_var}"
    local F1_rand="${!F1_rand_var}"
    local F2_rand="${!F2_rand_var}"
    
    local F0_min=$(awk -v f0="$F0_inj" -v df0="$DeltaF0" -v r="$F0_rand" 'BEGIN{printf "%.15g", f0 - df0/2.0 + r}')
    local F1_min=$(awk -v f1="$F1_inj" -v df1="$DeltaF1" -v r="$F1_rand" 'BEGIN{printf "%.15e", f1 - df1/2.0 + r}')
    local F2_min=$(awk -v f2="$F2_inj" -v df2="$DeltaF2" -v r="$F2_rand" 'BEGIN{printf "%.15e", f2 - df2/2.0 + r}')
    
    local output_file="${outdir}/dats/semicoh_results_${i}.dat"
    local log_file="${outdir}/dats/semicoh_results_${i}.log"
    
    # Debug: print parameters to log
    {
        echo "Running search $i"
        echo "F0_min: $F0_min"
        echo "F1_min: $F1_min"
        echo "F2_min: $F2_min"
        echo "F0_rand: $F0_rand"
        echo "F1_rand: $F1_rand"
        echo "F2_rand: $F2_rand"
    } > "$log_file"
    
    lalpulsar_HierarchSearchGCT \
        --fnameout="${output_file}" \
        --Freq="${F0_min}" \
        --FreqBand="${DeltaF0}" \
        --dFreq="${dF0}" \
        --f1dot="${F1_min}" \
        --f1dotBand="${DeltaF1}" \
        --df1dot="${dF1}" \
        --f2dot="${F2_min}" \
        --f2dotBand="${DeltaF2}" \
        --df2dot="${df2}" \
        --gammaRefine="${gamma1}" \
        --gamma2Refine="${gamma2}" \
        --DataFiles1="${sft_pattern}" \
        --gridType1=3 \
        --skyGridFile="{${Alpha_inj} ${Delta_inj}}" \
        --refTime="${tref}" \
        --tStack="${tStack}" \
        --nStacksMax="${nStacks}" \
        --nCand1=10 \
        --printCand1 \
        --semiCohToplist \
        --minStartTime1="${tstart}" \
        --maxStartTime1="${tend}" \
        --recalcToplistStats=TRUE \
        --FstatMethod=DemodBest \
        --FstatMethodRecalc=DemodOptC \
        --Dterms=8 >> "$log_file" 2>&1
    
    local ret_code=$?
    if [ $ret_code -ne 0 ]; then
        echo "Error running HierarchSearchGCT for run $i (exit code: $ret_code)" >> "$log_file"
        return 1
    fi
}

# Export necessary variables for parallel execution
export -f run_single_search
export F0_inj F1_inj F2_inj DeltaF0 DeltaF1 DeltaF2 dF0 dF1 df2 gamma1 gamma2
export outdir tref tStack nStacks tstart tend sft_pattern Alpha_inj Delta_inj

# Run searches sequentially or in parallel
echo "Running $N searches..."

# First, verify we have the random offsets
echo "Sample random offsets:"
echo "F0_randoms[0]: ${F0_randoms[0]}"
echo "F1_randoms[0]: ${F1_randoms[0]}"
echo "F2_randoms[0]: ${F2_randoms[0]}"

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution..."
    # Export the arrays for parallel
    for ((i=0; i<N; i++)); do
        export F0_random_$i="${F0_randoms[$i]}"
        export F1_random_$i="${F1_randoms[$i]}"
        export F2_random_$i="${F2_randoms[$i]}"
    done
    seq 0 $((N-1)) | parallel -j $(nproc) run_single_search {}
else
    echo "Running searches with background jobs..."
    # Fallback to background jobs with job control
    max_jobs=$(nproc 2>/dev/null || echo 4)
    for ((i=0; i<N; i++)); do
        # Export individual array elements for this iteration
        export F0_random_$i="${F0_randoms[$i]}"
        export F1_random_$i="${F1_randoms[$i]}"
        export F2_random_$i="${F2_randoms[$i]}"
        
        while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
            sleep 0.1
        done
        run_single_search $i &
        
        # Show progress
        if [ $((i % 10)) -eq 0 ]; then
            echo "Started $((i+1))/$N searches..."
        fi
    done
    wait
fi

echo "All searches completed."

# Process results and find max 2F values
echo "Processing results..."
max_twoFs=()

for ((i=0; i<N; i++)); do
    output_file="${outdir}/dats/semicoh_results_${i}.dat"
    if [ ! -f "$output_file" ]; then
        echo "Output file $output_file does not exist, skipping."
        continue
    fi
    
    # Extract the maximum 2F value from the file (column 8 is twoFr)
    max_twoF=$(grep -v '^%' "$output_file" | awk 'NF>=8 {print $8}' | sort -g | tail -1)
    if [ -n "$max_twoF" ]; then
        max_twoFs+=($max_twoF)
    fi
done

# Run perfectly matched search
echo "Running perfectly matched search..."
perfect_output_file="${outdir}/perfectly_matched_results.dat"

lalpulsar_HierarchSearchGCT \
    --Freq="${F0_inj}" \
    --FreqBand=0 \
    --dFreq="${dF0}" \
    --f1dot="${F1_inj}" \
    --f1dotBand=0 \
    --df1dot="${dF1}" \
    --f2dot="${F2_inj}" \
    --f2dotBand=0 \
    --df2dot="${df2}" \
    --fnameout="${perfect_output_file}" \
    --gammaRefine="${gamma1}" \
    --gamma2Refine="${gamma2}" \
    --DataFiles1="${sft_pattern}" \
    --gridType1=3 \
    --skyGridFile="{${Alpha_inj} ${Delta_inj}}" \
    --refTime="${tref}" \
    --tStack="${tStack}" \
    --nStacksMax="${nStacks}" \
    --nCand1=10 \
    --printCand1 \
    --semiCohToplist \
    --minStartTime1="${tstart}" \
    --maxStartTime1="${tend}" \
    --recalcToplistStats=TRUE \
    --FstatMethod=DemodBest \
    --FstatMethodRecalc=DemodOptC \
    --Dterms=8

if [ $? -ne 0 ]; then
    echo "Error running perfectly matched search"
    exit 1
fi

# Extract perfect 2F value
perfect_2F=$(grep -v '^%' "$perfect_output_file" | awk 'NF>=8 {print $8}' | sort -g | tail -1)
echo "Perfect 2F value: $perfect_2F"

# Calculate mismatches
echo "Calculating mismatches..."
mismatches=()
sum_mismatch=0

for max_twoF in "${max_twoFs[@]}"; do
    mismatch=$(awk -v p="$perfect_2F" -v m="$max_twoF" 'BEGIN{printf "%.15f", (p - m) / (p - 4)}')
    mismatches+=($mismatch)
    sum_mismatch=$(awk -v s="$sum_mismatch" -v m="$mismatch" 'BEGIN{printf "%.15f", s + m}')
done

# Calculate mean mismatch
if [ ${#mismatches[@]} -gt 0 ]; then
    mean_mismatch=$(awk -v s="$sum_mismatch" -v n="${#mismatches[@]}" 'BEGIN{printf "%.15f", s / n}')
    echo "Mean mismatch: $mean_mismatch"
    echo "Number of successful searches: ${#mismatches[@]}"
else
    echo "No mismatches calculated"
fi

# Save mismatches to file for potential later analysis
mismatch_file="${outdir}/mismatches.txt"
printf "%s\n" "${mismatches[@]}" > "$mismatch_file"
echo "Mismatches saved to: $mismatch_file"

# Also save a summary file
summary_file="${outdir}/summary.txt"
{
    echo "LAL Semi-Coherent Search Summary"
    echo "================================"
    echo "Number of searches: $N"
    echo "Successful searches: ${#mismatches[@]}"
    echo "Perfect 2F: $perfect_2F"
    echo "Mean mismatch: $mean_mismatch"
    echo ""
    echo "Search parameters:"
    echo "F0_inj: $F0_inj"
    echo "F1_inj: $F1_inj"
    echo "F2_inj: $F2_inj"
    echo "mf: $mf"
    echo "mf1: $mf1"
    echo "mf2: $mf2"
    echo "depth: $depth"
} > "$summary_file"

echo "Summary saved to: $summary_file"
echo "Analysis complete."