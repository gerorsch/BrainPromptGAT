"""
Script for running complete ablation study.

Executes ablation variants for prompts, mask, and pooling components.
Results are saved to a JSON file for later analysis.

Usage:
    # Run all prompt ablations (default, with mask and attention pooling)
    python run_ablation.py --site UCLA
    
    # Run specific ablation components
    python run_ablation.py --site UCLA --ablation_type prompts
    python run_ablation.py --site UCLA --ablation_type mask
    python run_ablation.py --site UCLA --ablation_type pooling
    
    # Run all combinations (24 experiments: 4 prompts × 2 masks × 3 poolings)
    python run_ablation.py --site UCLA --ablation_type all
"""

import subprocess
import sys
import json
import os
from datetime import datetime
import argparse
import itertools

# Ablation configurations
ABLATION_PROMPTS = ['no_prompt', 'random_prompt', 'static_prompt', 'full']
ABLATION_MASKS = ['with_mask', 'no_mask']
ABLATION_POOLINGS = ['attention', 'mean', 'max']

# Base configuration (same for all experiments)
BASE_ARGS = {
    'model': 'gnn_baseline_model',
    'batch_size': '2',
    'accumulation_steps': '16',
    'gat_heads': '2',
    'gat_hidden_dim': '32',
    'gat_num_layers': '2',
    'dropout': '0.3',
    'decay': '0.0005',
    'l1_weight': '0.03',
    'early_stopping_patience': '10',
    'lr': '0.0005',
    'loss_type': 'bce',
    'fc_num_layers': '1',
    'fc_activation': 'elu',
    'fc_use_batchnorm': True,
    'fc_hidden_dim': '128'
}


def run_experiment(site, ablation_mode, ablation_mask, ablation_pooling, base_args):
    """
    Run a single ablation experiment.
    
    Args:
        site: Site name (NYU, UCLA, UM, USM)
        ablation_mode: Prompt ablation mode
        ablation_mask: Mask ablation mode
        ablation_pooling: Pooling ablation mode
        base_args: Base configuration dictionary
    
    Returns:
        dict: Results dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"Running ablation experiment:")
    print(f"  Prompt: {ablation_mode}")
    print(f"  Mask: {ablation_mask}")
    print(f"  Pooling: {ablation_pooling}")
    print(f"  Site: {site}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = ['uv', 'run', 'main.py']
    
    # Add base arguments
    for key, value in base_args.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # Add site and ablation parameters
    cmd.extend(['--site', site])
    cmd.extend(['--ablation_mode', ablation_mode])
    cmd.extend(['--ablation_mask', ablation_mask])
    cmd.extend(['--ablation_pooling', ablation_pooling])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Parse output to extract final accuracy
        output = result.stdout
        accuracy = None
        
        # Look for "Final Accuracy: X.XX" pattern
        for line in output.split('\n'):
            if 'Final Accuracy:' in line:
                try:
                    accuracy = float(line.split('Final Accuracy:')[1].strip().rstrip('%'))
                    break
                except (ValueError, IndexError):
                    pass
        
        experiment_name = f"{ablation_mode}_{ablation_mask}_{ablation_pooling}"
        print(f"✓ Experiment {experiment_name} completed")
        if accuracy is not None:
            print(f"  Final Accuracy: {accuracy:.2f}%")
        
        return {
            'ablation_mode': ablation_mode,
            'ablation_mask': ablation_mask,
            'ablation_pooling': ablation_pooling,
            'site': site,
            'accuracy': accuracy,
            'status': 'success',
            'output': output
        }
        
    except subprocess.CalledProcessError as e:
        experiment_name = f"{ablation_mode}_{ablation_mask}_{ablation_pooling}"
        print(f"✗ Experiment {experiment_name} failed")
        print(f"  Error: {e.stderr}")
        return {
            'ablation_mode': ablation_mode,
            'ablation_mask': ablation_mask,
            'ablation_pooling': ablation_pooling,
            'site': site,
            'accuracy': None,
            'status': 'failed',
            'error': str(e.stderr)
        }


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for BrainPromptGAT')
    parser.add_argument('--site', type=str, choices=['NYU', 'UCLA', 'UM', 'USM'], 
                        default='UCLA', help='Site to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: ablation_results_SITE_TIMESTAMP.json)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with custom configuration (overrides BASE_ARGS)')
    parser.add_argument('--ablation_type', type=str, 
                        choices=['prompts', 'mask', 'pooling', 'all'],
                        default='prompts',
                        help="Type of ablation: 'prompts' (test prompts only), 'mask' (test mask only), 'pooling' (test pooling only), 'all' (all combinations)")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = BASE_ARGS.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
            print(f"Loaded custom configuration from {args.config}")
    
    # Determine output file
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'ablation_results_{args.site}_{args.ablation_type}_{timestamp}.json'
    
    print(f"\n{'='*80}")
    print(f"BrainPromptGAT Ablation Study")
    print(f"Site: {args.site}")
    print(f"Ablation Type: {args.ablation_type}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Determine which experiments to run
    if args.ablation_type == 'prompts':
        # Test prompts only (with default mask and pooling)
        experiments = [(p, 'with_mask', 'attention') for p in ABLATION_PROMPTS]
    elif args.ablation_type == 'mask':
        # Test mask only (with default prompt and pooling)
        experiments = [('full', m, 'attention') for m in ABLATION_MASKS]
    elif args.ablation_type == 'pooling':
        # Test pooling only (with default prompt and mask)
        experiments = [('full', 'with_mask', p) for p in ABLATION_POOLINGS]
    else:  # 'all'
        # Test all combinations
        experiments = list(itertools.product(ABLATION_PROMPTS, ABLATION_MASKS, ABLATION_POOLINGS))
    
    print(f"Total experiments to run: {len(experiments)}\n")
    
    # Run all ablation experiments
    results = []
    for i, (mode, mask, pooling) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        result = run_experiment(args.site, mode, mask, pooling, config)
        results.append(result)
        
        # Small delay between experiments
        import time
        time.sleep(2)
    
    # Save results
    output_data = {
        'site': args.site,
        'ablation_type': args.ablation_type,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Prompt':<18} {'Mask':<12} {'Pooling':<12} {'Status':<10} {'Accuracy':<15}")
    print("-" * 75)
    
    for result in results:
        mode = result['ablation_mode']
        mask = result['ablation_mask']
        pooling = result['ablation_pooling']
        status = result['status']
        acc = result['accuracy']
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        print(f"{mode:<18} {mask:<12} {pooling:<12} {status:<10} {acc_str:<15}")
    
    print(f"\nResults saved to: {args.output}")
    print(f"\nTo analyze results, run: python analyze_ablation.py {args.output}")


if __name__ == '__main__':
    main()
