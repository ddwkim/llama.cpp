#!/usr/bin/env python3
"""
Nsight Compute CSV Analysis Script
Calculates average values for each component (kernel) performance metrics.
Uses only standard Python libraries without pandas dependency.
"""

import csv
import sys
import os
import re
from collections import defaultdict

def parse_number(value_str):
    """
    Converts a number string with commas to float.
    
    Args:
        value_str (str): Number string (e.g., "7,492,466,296.59")
    
    Returns:
        float: Converted number
    """
    if value_str is None or value_str == '':
        return None
    
    try:
        # Remove commas and convert to float
        clean_str = str(value_str).replace(',', '')
        return float(clean_str)
    except (ValueError, TypeError):
        return None

def calculate_stats(values):
    """
    Calculates statistics for a list of numbers.
    
    Args:
        values (list): List of numbers
    
    Returns:
        dict: Statistics (mean, min, max, std, count)
    """
    if not values:
        return None
    
    n = len(values)
    mean_val = sum(values) / n
    min_val = min(values)
    max_val = max(values)
    
    # Calculate standard deviation
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = variance ** 0.5
    
    return {
        'count': n,
        'mean': mean_val,
        'min': min_val,
        'max': max_val,
        'std': std_val
    }

def analyze_ncu_csv(csv_file_path):
    """
    Analyzes Nsight Compute CSV file to calculate average performance metrics for each kernel.
    
    Args:
        csv_file_path (str): Path to the CSV file to analyze
    """
    
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found: {csv_file_path}")
        return
    
    try:
        # Read CSV file - skip first two lines as they contain profiling info
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Remove first two lines and parse CSV
        csv_content = ''.join(lines[2:])
        
        # Parse CSV
        reader = csv.DictReader(csv_content.splitlines())
        rows = list(reader)
        
        print(f"CSV file successfully read: {csv_file_path}")
        print(f"Total data rows (excluding 2 profiling info lines): {len(rows)}\n")
        
        if not rows:
            print("No data found in CSV file.")
            return
        
        # Find numeric metric columns
        sample_row = rows[0]
        
        # Modified for Nsight Compute CSV structure
        # Use Metric Name and Metric Value columns
        if 'Metric Name' not in sample_row or 'Metric Value' not in sample_row:
            print("Warning: Metric Name or Metric Value columns not found.")
            print("Available columns:")
            for col in sample_row.keys():
                print(f"  - {col}")
            return
        
        # Target metrics to analyze
        target_metrics = [
            'Duration', 'Elapsed Cycles', 'SM Active Cycles',
            'Memory Throughput', 'DRAM Throughput', 'L1/TEX Cache Throughput',
            'L2 Cache Throughput', 'Compute (SM) Throughput'
        ]
        
        print("Available metrics for analysis:")
        for metric in target_metrics:
            print(f"  - {metric}")
        print()
        
        # Group by kernel and calculate averages
        kernel_stats = defaultdict(lambda: defaultdict(list))
        
        for row in rows:
            kernel_name = row.get('Kernel Name', '')
            metric_name = row.get('Metric Name', '')
            metric_value = row.get('Metric Value', '')
            
            if not kernel_name or not metric_name or not metric_value:
                continue
            
            # Check if metric is in target list
            if metric_name not in target_metrics:
                continue
            
            # Extract core part from kernel name
            kernel_match = re.search(r'(\w+_kernel\w*)', kernel_name)
            if kernel_match:
                kernel_short = kernel_match.group(1)
            else:
                kernel_short = kernel_name
            
            # Extract NVTX marker info
            nvtx_field = row.get('Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg', '')
            nvtx_match = re.search(r'bench_(\w+)', nvtx_field)
            nvtx_marker = nvtx_match.group(1) if nvtx_match else 'unknown'
            
            # Collect metric values
            value = parse_number(metric_value)
            if value is not None:
                kernel_stats[kernel_short][metric_name].append(value)
        
        # Output results - only show summary
        print("=" * 80)
        print("Kernel Performance Metrics Analysis Results")
        print("=" * 80)
        
        # Overall statistics summary
        print("Overall Statistics Summary")
        print("=" * 80)
        
        total_kernels = len(kernel_stats)
        total_executions = sum(len(metrics.get('Duration', [])) for metrics in kernel_stats.values())
        
        print(f"Total kernels: {total_kernels}")
        print(f"Total executions: {total_executions}")
        
        # Calculate overall metric averages for each kernel
        print("\n📈 Overall Metric Averages by Kernel:")
        for kernel_name, metrics in kernel_stats.items():
            print(f"\n🔹 {kernel_name}:")
            print("-" * 30)
            
            for metric_name in target_metrics:
                if metric_name in metrics and metrics[metric_name]:
                    values = metrics[metric_name]
                    overall_mean = sum(values) / len(values)
                    print(f"  {metric_name}: {overall_mean:.4f}")
                    
                    # Add unit information
                    if metric_name == 'Duration':
                        print(f"     Unit: ns (nanoseconds)")
                    elif 'Throughput' in metric_name:
                        print(f"     Unit: %")
                    elif 'Cycles' in metric_name:
                        print(f"     Unit: cycles")
        
    except Exception as e:
        print(f"Error: An error occurred while analyzing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_ncu_csv.py <csv_file_path>")
        print("Example: python3 analyze_ncu_csv.py ncu_iterations.csv")
        return
    
    csv_file_path = sys.argv[1]
    analyze_ncu_csv(csv_file_path)

if __name__ == "__main__":
    main() 