#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Timing and Cleanup ---
START_TIME=$SECONDS
WORKSPACE="" # Global variable for the workspace path

# This cleanup function will be called automatically when the script exits.
cleanup() {
  if [ -n "${WORKSPACE}" ] && [ -d "${WORKSPACE}" ]; then
    echo "Cleaning up memory workspace: ${WORKSPACE}"
    rm -rf "${WORKSPACE}"
  fi
}
trap cleanup EXIT # Register the cleanup function to run on any exit

# --- Aggressive Memory Optimization Setup ---
echo "Setting up aggressive memory optimization..."
# Use /dev/shm (RAM disk) if it exists, otherwise fall back to the system's default temp directory.
if [ -d "/dev/shm" ]; then
  TEMP_BASE="/dev/shm"
  echo "Using /dev/shm for memory-first workspace."
else
  TEMP_BASE="/tmp"
  echo "Warning: /dev/shm not found. Using ${TEMP_BASE} as workspace."
fi

# Create a unique, secure temporary workspace.
WORKSPACE=$(mktemp -d "${TEMP_BASE}/lal_aggressive.XXXXXX")
SFT_DIR="${WORKSPACE}/sfts"
DATS_DIR="${WORKSPACE}/dats"
mkdir -p "${SFT_DIR}" "${DATS_DIR}"
echo "Workspace created at: ${WORKSPACE}"

# Set LAL environment variables for aggressive optimization
export TMPDIR="${WORKSPACE}"       # Force all temp files into our memory workspace
export LAL_CACHE_SIZE=0            # Disable LAL's internal caching
export OMP_NUM_THREADS=1           # Disable OpenMP to avoid thread contention with our parallelism

# --- Configuration ---
N=500
LABEL="LALSemiCoherentF0F1F2_aggressive_memory"
FINAL_OUTDIR="LAL_example_data/${LABEL}" # Persistent directory for final results
mkdir -p "${FINAL_OUTDIR}"

# GW data properties
SQRT_SX=1e-22
TSTART=1126051217
DURATION=$((120 * 86400))
TEND=$((TSTART + DURATION))
TREF=$(echo "scale=10; ${TSTART} + ${DURATION} / 2" | bc)
IFO="H1"

# Injected signal parameters (using a weaker signal to make the search harder)
DEPTH=0.6
F0_INJ=151.5
F1_INJ=-1e-10
F2_INJ=-1e-20

# Automatic Number Conversion (for bc compatibility)
H0=$(echo "scale=25; $(printf "%.25f" "${SQRT_SX}") / ${DEPTH}" | bc)
F1_INJ_DECIMAL=$(printf "%.25f" "${F1_INJ}")
F2_INJ_DECIMAL=$(printf "%.25f" "${F2_INJ}")

# Build injection string with decimal numbers
INJECTION_PARAMS="{Alpha=0.5; Delta=1.0; Freq=${F0_INJ}; f1dot=${F1_INJ_DECIMAL}; f2dot=${F2_INJ_DECIMAL}; refTime=${TREF}; h0=${H0}; cosi=1.0; psi=0.0; phi0=0.0;}"

# Semi-coherent search parameters
TSTACK=$((20 * 86400)) # 20 day segments
NSTACKS=$((DURATION / TSTACK))

# --- Step 1: Generate SFT data in Memory ---
echo "Generating SFT data in memory (${SFT_DIR})..."
SFT_START_TIME=$SECONDS
lalpulsar_Makefakedata_v5 \
    --IFOs="${IFO}" \
    --sqrtSX="${SQRT_SX}" \
    --startTime="${TSTART}" \
    --duration="${DURATION}" \
    --fmin=$(echo "${F0_INJ} - 0.2" | bc) \
    --Band=0.4 \
    --Tsft=1800 \
    --outSFTdir="${SFT_DIR}" \
    --outLabel="MemOpt" \
    --injectionSources="${INJECTION_PARAMS}" \
    --randSeed=1234
SFT_TIME=$(($SECONDS - SFT_START_TIME))
echo "SFT generation completed in ${SFT_TIME} seconds."

# --- Step 2: Set up grid search parameters ---
MF=0.01
MF1=0.001
MF2=0.08
DF0=$(echo "scale=15; sqrt(12 * ${MF}) / (3.14159 * ${TSTACK})" | bc -l)
DF1=$(echo "scale=20; sqrt(180 * ${MF1}) / (3.14159 * ${TSTACK}^2)" | bc -l)
DF2=$(echo "scale=30; sqrt(25200 * ${MF2}) / (3.14159 * ${TSTACK}^3)" | bc -l)

N1=2; N2=3; N3=3; GAMMA1=11; GAMMA2=91
DELTA_F0=$(echo "${N1} * ${DF0}" | bc -l)
DELTA_F1=$(echo "${N2} * ${DF1}" | bc -l)
DELTA_F2=$(echo "${N3} * ${DF2}" | bc -l)

# --- Step 3: Run Semi-coherent Searches in Parallel ---
echo "Starting ${N} parallel searches..."
SEARCH_START_TIME=$SECONDS

SHARED_CMD=(
    "--DataFiles1=${SFT_DIR}/*.sft"
    "--assumeSqrtSX=${SQRT_SX}"
    "--gridType1=3"
    "--skyGridFile={0.5 1.0}"
    "--refTime=${TREF}"
    "--tStack=${TSTACK}"
    "--nStacksMax=${NSTACKS}"
    "--nCand1=30"
    "--printCand1"
    "--semiCohToplist"
    "--minStartTime1=${TSTART}"
    "--maxStartTime1=${TEND}"
    "--recalcToplistStats=TRUE"
    "--FstatMethod=DemodBest"
    "--FstatMethodRecalc=DemodBest"
)

# Determine number of parallel workers, leaving 2 cores for the system
WORKERS=$(($(nproc) - 2))
[ "$WORKERS" -lt 1 ] && WORKERS=1
echo "Using up to ${WORKERS} parallel workers."

for i in $(seq 0 $((N-1))); do
    F0_RANDOM=$(awk 'BEGIN{srand(); print rand()-0.5}')
    F1_RANDOM=$(awk 'BEGIN{srand(); print rand()-0.5}')
    F2_RANDOM=$(awk 'BEGIN{srand(); print rand()-0.5}')

    F0_MIN=$(echo "${F0_INJ} - ${DELTA_F0}/2 + ${F0_RANDOM}*${DF0}" | bc -l)
    F1_MIN=$(echo "${F1_INJ_DECIMAL} - ${DELTA_F1}/2 + ${F1_RANDOM}*${DF1}" | bc -l)
    F2_MIN=$(echo "${F2_INJ_DECIMAL} - ${DELTA_F2}/2 + ${F2_RANDOM}*${DF2}" | bc -l)
    
    OUTPUT_FILE="${DATS_DIR}/results_${i}.dat"

    # Launch job in the background
    ( # Run in a subshell
        lalpulsar_HierarchSearchGCT \
            --fnameout="${OUTPUT_FILE}" \
            --Freq="${F0_MIN}" --FreqBand="${DELTA_F0}" --dFreq="${DF0}" \
            --f1dot="${F1_MIN}" --f1dotBand="${DELTA_F1}" --df1dot="${DF1}" \
            --f2dot="${F2_MIN}" --f2dotBand="${DELTA_F2}" --df2dot="${DF2}" \
            --gammaRefine="${GAMMA1}" --gamma2Refine="${GAMMA2}" \
            "${SHARED_CMD[@]}" &> "${OUTPUT_FILE}.err"
    ) &

    # Limit the number of concurrent jobs
    if (( (i+1) % WORKERS == 0 )); then
        wait -n # Wait for the next job to finish
    fi
done

wait # Wait for all remaining jobs to complete
SEARCH_TIME=$(($SECONDS - SEARCH_START_TIME))
echo -e "\nParallel search writing completed in ${SEARCH_TIME} seconds."

# --- Step 4: Run Perfectly Matched Search ---
echo "Computing perfect match 2F value..."
PERFECT_OUTPUT_FILE="${DATS_DIR}/perfect_match.dat"
lalpulsar_HierarchSearchGCT \
    --Freq="${F0_INJ}" --FreqBand=0 --dFreq="${DF0}" \
    --f1dot="${F1_INJ_DECIMAL}" --f1dotBand=0 --df1dot="${DF1}" \
    --f2dot="${F2_INJ_DECIMAL}" --f2dotBand=0 --df2dot="${DF2}" \
    --fnameout="${PERFECT_OUTPUT_FILE}" \
    --gammaRefine="${GAMMA1}" --gamma2Refine="${GAMMA2}" \
    "${SHARED_CMD[@]}"

# --- Step 5: Process Results and Calculate Mismatch ---
echo "Processing results and calculating mismatch..."
PERFECT_2F=$(awk '!/^%/ && NF >= 8 {print $8}' "${PERFECT_OUTPUT_FILE}" | sort -gr | head -n1)
echo "Perfect match 2F: ${PERFECT_2F}"

declare -a MAX_TWOFS
for i in $(seq 0 $((N-1))); do
    FILE="${DATS_DIR}/results_${i}.dat"
    if [ -f "$FILE" ]; then
        # Parse the max 2F value from the file
        MAX_2F=$(awk '!/^%/ && NF >= 8 {print $8}' "$FILE" | sort -gr | head -n1)
        if [ -n "$MAX_2F" ]; then
            MAX_TWOFS+=("$MAX_2F")
        fi
        # Immediately delete the file to save memory
        rm "$FILE"
    fi
done
echo "Successfully processed ${#MAX_TWOFS[@]} / ${N} result files."

# Calculate mismatches
MISMATCHES=()
if (( $(echo "${PERFECT_2F} > 4" | bc -l) )); then
    for val in "${MAX_TWOFS[@]}"; do
        MISMATCH=$(echo "scale=10; (${PERFECT_2F} - ${val}) / (${PERFECT_2F} - 4)" | bc -l)
        # Clamp value between 0 and 1
        if (( $(echo "$MISMATCH < 0" | bc -l) )); then MISMATCH=0; fi
        if (( $(echo "$MISMATCH > 1" | bc -l) )); then MISMATCH=1; fi
        MISMATCHES+=("$MISMATCH")
    done
fi

# --- Step 6: Final Reporting ---
if [ ${#MISMATCHES[@]} -gt 0 ]; then
    SUM_MISMATCH=$(IFS=+; echo "scale=10; ${MISMATCHES[*]}" | bc)
    MEAN_MISMATCH=$(echo "scale=10; ${SUM_MISMATCH} / ${#MISMATCHES[@]}" | bc)
    
    SUM_SQ_DIFF=0
    for m in "${MISMATCHES[@]}"; do
        DIFF=$(echo "scale=10; ${m} - ${MEAN_MISMATCH}" | bc)
        SUM_SQ_DIFF=$(echo "scale=10; ${SUM_SQ_DIFF} + ${DIFF}^2" | bc)
    done
    STD_MISMATCH=$(echo "scale=10; sqrt(${SUM_SQ_DIFF} / ${#MISMATCHES[@]})" | bc -l)
    
    echo "Mismatch statistics:"
    echo "  Mean: ${MEAN_MISMATCH}"
    echo "  Std:  ${STD_MISMATCH}"
else
    echo "Warning: No valid mismatch calculations possible."
fi

TOTAL_TIME=$(($SECONDS - START_TIME))
SUMMARY_FILE="${FINAL_OUTDIR}/summary.txt"
{
    echo "Aggressive Memory-Optimized LAL Search Results"
    echo "==============================================="
    echo "Total runs attempted: ${N}"
    echo "Successful runs processed: ${#MISMATCHES[@]}"
    echo "Perfect match 2F: ${PERFECT_2F:-"N/A"}"
    echo "Mean mismatch: ${MEAN_MISMATCH:-"N/A"}"
    echo "Std mismatch: ${STD_MISMATCH:-"N/A"}"
    echo "-----------------------------------------------"
    echo "SFT generation time: ${SFT_TIME} seconds"
    echo "Parallel search time: ${SEARCH_TIME} seconds"
    echo "Total execution time: ${TOTAL_TIME} seconds"
    echo "Performance: $(echo "scale=2; ${N}/${TOTAL_TIME}" | bc) runs/sec"
    echo "Memory workspace used: ${WORKSPACE}"
} > "${SUMMARY_FILE}"

echo "----------------------------------------"
echo "Aggressive optimization completed!"
echo "Total execution time: ${TOTAL_TIME} seconds."
echo "Summary report saved to ${SUMMARY_FILE}"
echo "----------------------------------------"

# The 'trap' will now automatically call the cleanup function to remove the workspace.