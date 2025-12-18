"""
Script for analyzing ablation study results.

Reads results from Excel files or JSON files and creates comparison tables
similar to Table 3 in the BrainPrompt paper.

Usage:
    # Analyze from JSON (from run_ablation.py)
    python analyze_ablation.py ablation_results_UCLA_20240101_120000.json
    
    # Analyze from Excel files directory
    python analyze_ablation.py --excel_dir ./result/UCLA/1/ --ablation_modes no_prompt,random_prompt,static_prompt,full
"""

import json
import os
import sys
import argparse
import glob
from pathlib import Path
try:
    import xlrd
    HAS_XLRD = True
except ImportError:
    HAS_XLRD = False
    print("Warning: xlrd not installed. Excel file reading disabled. Install with: pip install xlrd")


# Mapping of ablation modes to display names
PROMPT_NAMES = {
    'no_prompt': 'Baseline (w/o Prompts)',
    'random_prompt': 'Random Prompt',
    'static_prompt': 'Static Prompt (BERT only)',
    'full': 'Full Model (BERT + Soft)'
}

MASK_NAMES = {
    'with_mask': 'With Mask',
    'no_mask': 'No Mask'
}

POOLING_NAMES = {
    'attention': 'Attention',
    'mean': 'Mean',
    'max': 'Max'
}


def read_excel_results(excel_path):
    """
    Read results from Excel file saved by train_and_test_baseline_model.
    
    Format: [kfold_index, prec, recall, acc, F1, auc, sen, spe]
    Row 0: Header
    Rows 1-5: Fold results
    Row 6: Average
    
    Args:
        excel_path: Path to Excel file
    
    Returns:
        dict: Dictionary with metrics from average row
    """
    if not HAS_XLRD:
        return None
    
    try:
        workbook = xlrd.open_workbook(excel_path)
        sheet = workbook.sheet_by_index(0)
        
        # Find average row (usually row 6, index 6)
        # Format: ['average', prec, recall, acc, F1, auc, sen, spe]
        for row_idx in range(sheet.nrows):
            row = sheet.row_values(row_idx)
            if len(row) > 0 and str(row[0]).lower() == 'average':
                return {
                    'precision': row[1] * 100 if len(row) > 1 else None,
                    'recall': row[2] * 100 if len(row) > 2 else None,
                    'accuracy': row[3] * 100 if len(row) > 3 else None,
                    'f1': row[4] * 100 if len(row) > 4 else None,
                    'auc': row[5] * 100 if len(row) > 5 else None,
                    'sensitivity': row[6] * 100 if len(row) > 6 else None,
                    'specificity': row[7] * 100 if len(row) > 7 else None
                }
        
        return None
    except Exception as e:
        print(f"Error reading Excel file {excel_path}: {e}")
        return None


def find_latest_excel_files(result_dir, ablation_mode):
    """
    Find the most recent Excel file for a given ablation mode.
    
    Args:
        result_dir: Directory containing result files
        ablation_mode: Ablation mode to search for
    
    Returns:
        str: Path to Excel file or None
    """
    # Excel files are saved with pattern: model-site_result-epoch-lr-batch-weight_decay-timestamp.xlsx
    # We need to find files that match the ablation mode
    # Since ablation_mode is not in filename, we'll need to check all files
    # For now, return the most recent file (user should organize by mode)
    
    pattern = os.path.join(result_dir, '*.xlsx')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Return most recent file
    return max(files, key=os.path.getmtime)


def analyze_from_json(json_path):
    """
    Analyze results from JSON file created by run_ablation.py.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        dict: Results dictionary, site, timestamp, ablation_type
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for result in data['results']:
        # Create key from all ablation components
        mode = result.get('ablation_mode', 'full')
        mask = result.get('ablation_mask', 'with_mask')
        pooling = result.get('ablation_pooling', 'attention')
        
        # Create descriptive key
        key = f"{mode}_{mask}_{pooling}"
        results[key] = {
            'ablation_mode': mode,
            'ablation_mask': mask,
            'ablation_pooling': pooling,
            'accuracy': result.get('accuracy'),
            'status': result.get('status'),
            'precision': None,  # Not captured in JSON yet
            'recall': None,
            'f1': None,
            'auc': None
        }
    
    return results, data.get('site', 'Unknown'), data.get('timestamp', ''), data.get('ablation_type', 'unknown')


def create_comparison_table(results, site='Unknown', ablation_type='unknown'):
    """
    Create a formatted comparison table similar to paper Table 3.
    
    Args:
        results: Dictionary mapping experiment keys to metrics
        site: Site name
        ablation_type: Type of ablation study
    """
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY RESULTS - {site}")
    print(f"Ablation Type: {ablation_type}")
    print(f"{'='*80}\n")
    
    # Group results by ablation type
    if ablation_type == 'prompts':
        # Show only prompt variations
        print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print("-" * 95)
        
        order = ['no_prompt', 'random_prompt', 'static_prompt', 'full']
        for mode in order:
            # Find result with this prompt mode (default mask and pooling)
            key = f"{mode}_with_mask_attention"
            if key not in results:
                continue
            
            metrics = results[key]
            name = PROMPT_NAMES.get(mode, mode)
            
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
            prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
            recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
            f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
            auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
            
            status = metrics.get('status', 'unknown')
            status_marker = "✓" if status == 'success' else "✗"
            
            print(f"{status_marker} {name:<33} {acc:<12} {prec:<12} {recall:<12} {f1:<12} {auc:<12}")
    
    elif ablation_type == 'mask':
        # Show mask variations
        print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print("-" * 95)
        
        for mask in ['with_mask', 'no_mask']:
            key = f"full_{mask}_attention"
            if key not in results:
                continue
            
            metrics = results[key]
            name = MASK_NAMES.get(mask, mask)
            
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
            prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
            recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
            f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
            auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
            
            status = metrics.get('status', 'unknown')
            status_marker = "✓" if status == 'success' else "✗"
            
            print(f"{status_marker} {name:<33} {acc:<12} {prec:<12} {recall:<12} {f1:<12} {auc:<12}")
    
    elif ablation_type == 'pooling':
        # Show pooling variations
        print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print("-" * 95)
        
        for pooling in ['attention', 'mean', 'max']:
            key = f"full_with_mask_{pooling}"
            if key not in results:
                continue
            
            metrics = results[key]
            name = POOLING_NAMES.get(pooling, pooling)
            
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
            prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
            recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
            f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
            auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
            
            status = metrics.get('status', 'unknown')
            status_marker = "✓" if status == 'success' else "✗"
            
            print(f"{status_marker} {name:<33} {acc:<12} {prec:<12} {recall:<12} {f1:<12} {auc:<12}")
    
    else:
        # Show all results
        print(f"{'Prompt':<18} {'Mask':<12} {'Pooling':<12} {'Accuracy':<12} {'Status':<10}")
        print("-" * 75)
        
        # Sort by accuracy (descending)
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1].get('accuracy', 0) or 0, 
                               reverse=True)
        
        for key, metrics in sorted_results:
            mode = metrics.get('ablation_mode', 'unknown')
            mask = metrics.get('ablation_mask', 'unknown')
            pooling = metrics.get('ablation_pooling', 'unknown')
            
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
            status = metrics.get('status', 'unknown')
            status_marker = "✓" if status == 'success' else "✗"
            
            print(f"{status_marker} {mode:<16} {mask:<12} {pooling:<12} {acc:<12} {status:<10}")
    
    print()


def create_markdown_table(results, site='Unknown', ablation_type='unknown'):
    """
    Create markdown table for easy copy-paste into reports.
    
    Args:
        results: Dictionary mapping experiment keys to metrics
        site: Site name
        ablation_type: Type of ablation study
    """
    print(f"\n{'='*80}")
    print("MARKDOWN TABLE (for reports)")
    print(f"{'='*80}\n")
    
    if ablation_type == 'prompts':
        print("| Model | Accuracy | Precision | Recall | F1 | AUC |")
        print("|-------|----------|-----------|--------|-----|-----|")
        
        order = ['no_prompt', 'random_prompt', 'static_prompt', 'full']
        for mode in order:
            key = f"{mode}_with_mask_attention"
            if key not in results:
                continue
            
            metrics = results[key]
            name = PROMPT_NAMES.get(mode, mode)
            
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
            prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
            recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
            f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
            auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
            
            print(f"| {name} | {acc} | {prec} | {recall} | {f1} | {auc} |")
    
    elif ablation_type in ['mask', 'pooling']:
        component = 'Mask' if ablation_type == 'mask' else 'Pooling'
        print(f"| {component} | Accuracy | Precision | Recall | F1 | AUC |")
        print("|-------|----------|-----------|--------|-----|-----|")
        
        if ablation_type == 'mask':
            for mask in ['with_mask', 'no_mask']:
                key = f"full_{mask}_attention"
                if key not in results:
                    continue
                metrics = results[key]
                name = MASK_NAMES.get(mask, mask)
                acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
                prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
                recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
                f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
                auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
                print(f"| {name} | {acc} | {prec} | {recall} | {f1} | {auc} |")
        else:  # pooling
            for pooling in ['attention', 'mean', 'max']:
                key = f"full_with_mask_{pooling}"
                if key not in results:
                    continue
                metrics = results[key]
                name = POOLING_NAMES.get(pooling, pooling)
                acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] is not None else "N/A"
                prec = f"{metrics['precision']:.2f}%" if metrics.get('precision') is not None else "N/A"
                recall = f"{metrics['recall']:.2f}%" if metrics.get('recall') is not None else "N/A"
                f1 = f"{metrics['f1']:.2f}%" if metrics.get('f1') is not None else "N/A"
                auc = f"{metrics['auc']:.2f}%" if metrics.get('auc') is not None else "N/A"
                print(f"| {name} | {acc} | {prec} | {recall} | {f1} | {auc} |")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('input', type=str, nargs='?', default=None,
                        help='Input JSON file from run_ablation.py')
    parser.add_argument('--excel_dir', type=str, default=None,
                        help='Directory containing Excel result files')
    parser.add_argument('--ablation_modes', type=str, default=None,
                        help='Comma-separated list of ablation modes (for Excel mode)')
    parser.add_argument('--markdown', action='store_true',
                        help='Output markdown table format')
    
    args = parser.parse_args()
    
    if args.input and os.path.exists(args.input):
        # Analyze from JSON
        results, site, timestamp, ablation_type = analyze_from_json(args.input)
        create_comparison_table(results, site, ablation_type)
        if args.markdown:
            create_markdown_table(results, site, ablation_type)
    
    elif args.excel_dir and os.path.exists(args.excel_dir):
        # Analyze from Excel files
        if not HAS_XLRD:
            print("Error: xlrd not installed. Cannot read Excel files.")
            print("Install with: pip install xlrd")
            sys.exit(1)
        
        if not args.ablation_modes:
            print("Error: --ablation_modes required when using --excel_dir")
            print("Example: --ablation_modes no_prompt,random_prompt,static_prompt,full")
            sys.exit(1)
        
        modes = [m.strip() for m in args.ablation_modes.split(',')]
        results = {}
        
        # For each mode, find and read Excel file
        # Note: This assumes files are organized by mode somehow
        # User may need to manually specify file paths
        print("Warning: Excel file reading requires manual file organization.")
        print("Consider using JSON output from run_ablation.py instead.")
        
        for mode in modes:
            excel_path = find_latest_excel_files(args.excel_dir, mode)
            if excel_path:
                metrics = read_excel_results(excel_path)
                if metrics:
                    results[mode] = metrics
        
        if results:
            create_comparison_table(results, os.path.basename(args.excel_dir))
            if args.markdown:
                create_markdown_table(results, os.path.basename(args.excel_dir))
        else:
            print("No results found in Excel files.")
    
    else:
        print("Error: Please provide either:")
        print("  1. JSON file: python analyze_ablation.py <json_file>")
        print("  2. Excel directory: python analyze_ablation.py --excel_dir <dir> --ablation_modes <modes>")
        sys.exit(1)


if __name__ == '__main__':
    main()
