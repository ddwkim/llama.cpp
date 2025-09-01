#!/bin/bash

# Nsight Compute Analysis Script for benchmark-conv2d-dw2
# This script runs Nsight Compute profiling and analyzes the results

# Check if correct number of arguments is provided
if [ $# -ne 6 ]; then
    echo "Usage: $0 <input_size> <kernel_size> <stride> <padding> <dilation> <iterations>"
    echo "Example: $0 128 5 2 1 1 100"
    exit 1
fi

# Parse command line arguments
INPUT_SIZE=$1
KERNEL_SIZE=$2
STRIDE=$3
PADDING=$4
DILATION=$5
ITERATIONS=$6

# Set output CSV filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="ncu_benchmark2_${TIMESTAMP}.csv"

echo "=========================================="
echo "Nsight Compute Analysis Script for benchmark-conv2d-dw2"
echo "=========================================="
echo "Configuration:"
echo "  Input size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "  Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
echo "  Stride: (${STRIDE},${STRIDE})"
echo "  Padding: (${PADDING},${PADDING})"
echo "  Dilation: (${DILATION},${DILATION})"
echo "  Iterations: ${ITERATIONS}"
echo "  Output CSV: ${CSV_FILE}"
echo ""

# Check if the executable exists
if [ ! -f "./benchmark-conv2d-dw2" ]; then
    echo "Error: Executable './benchmark-conv2d-dw2' not found!"
    echo "Please compile the program first."
    exit 1
fi

# Check if analyze_ncu_csv.py exists
if [ ! -f "./analyze_ncu_csv.py" ]; then
    echo "Error: Analysis script './analyze_ncu_csv.py' not found!"
    exit 1
fi

echo "Step 1: Running Nsight Compute profiling..."
echo "Command: sudo /usr/local/cuda/bin/ncu --target-processes all --nvtx --nvtx-include \"bench_cur\" --nvtx-include \"bench_backup\" --section SpeedOfLight --launch-count 9999 --csv --log-file ${CSV_FILE} ./benchmark-conv2d-dw2 ${INPUT_SIZE} ${KERNEL_SIZE} ${STRIDE} ${PADDING} ${DILATION} ${ITERATIONS}"
echo ""

# Run Nsight Compute profiling
sudo /usr/local/cuda/bin/ncu --target-processes all --nvtx --nvtx-include "bench_cur" --nvtx-include "bench_backup" --section SpeedOfLight --launch-count 9999 --csv --log-file "${CSV_FILE}" ./benchmark-conv2d-dw2 "${INPUT_SIZE}" "${KERNEL_SIZE}" "${STRIDE}" "${PADDING}" "${DILATION}" "${ITERATIONS}"

# Check if profiling was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Nsight Compute profiling completed successfully!"
    echo "CSV file generated: ${CSV_FILE}"
    
    # Check if CSV file was created and has content
    if [ -f "${CSV_FILE}" ] && [ -s "${CSV_FILE}" ]; then
        echo ""
        echo "Step 2: Analyzing results with Python script..."
        echo "Command: python3 analyze_ncu_csv.py ${CSV_FILE}"
        echo ""
        
        # Run Python analysis script
        python3 analyze_ncu_csv.py "${CSV_FILE}"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Analysis completed successfully!"
            echo ""
            echo "Summary:"
            echo "  - Profiling data saved to: ${CSV_FILE}"
            echo "  - Analysis results displayed above"
        else
            echo ""
            echo "❌ Error: Python analysis script failed!"
            exit 1
        fi
    else
        echo ""
        echo "❌ Error: CSV file was not created or is empty!"
        echo "Please check the Nsight Compute command and try again."
        exit 1
    fi
else
    echo ""
    echo "❌ Error: Nsight Compute profiling failed!"
    echo "Please check the command and try again."
    exit 1
fi

echo ""
echo "=========================================="
echo "Script execution completed!"
echo "==========================================" 