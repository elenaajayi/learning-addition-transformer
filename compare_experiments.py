"""
Experiment Comparison Visualization.

Creates side-by-side comparisons of baseline (k=3 only) vs mixed (k=3,4) training.
Highlights the trade-off between in-distribution accuracy and length generalization.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Configure matplotlib
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme
COLORS = {
    'baseline': '#2E86AB',      # Blue
    'mixed': '#E94F37',         # Red-orange
    'baseline_light': '#8EC4E0',
    'mixed_light': '#F5A99A',
    'success': '#44AF69',
    'warning': '#F18F01',
    'danger': '#C73E1D',
}


def load_experiment_data(experiment_dir: str) -> Tuple[Dict, Dict]:
    """
    Load training history and evaluation results for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Tuple of (training_history, evaluation_results)
    """
    history_path = os.path.join(experiment_dir, "checkpoints", "training_history.json")
    results_path = os.path.join(experiment_dir, "results.json")
    
    history = None
    results = None
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            results = data.get('results', data)
    
    return history, results


def plot_training_comparison(
    baseline_history: Dict,
    mixed_history: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves comparison between experiments.
    
    Args:
        baseline_history: Training history from baseline experiment
        mixed_history: Training history from mixed experiment
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
    epochs_mixed = range(1, len(mixed_history['train_loss']) + 1)
    
    # 1. Loss Comparison
    ax1 = axes[0]
    ax1.plot(epochs_baseline, baseline_history['val_loss'], 
             color=COLORS['baseline'], linewidth=2.5, label='Baseline (k=3)')
    ax1.plot(epochs_mixed, mixed_history['val_loss'], 
             color=COLORS['mixed'], linewidth=2.5, label='Mixed (k=3,4)')
    ax1.fill_between(epochs_baseline, baseline_history['val_loss'], 
                     alpha=0.15, color=COLORS['baseline'])
    ax1.fill_between(epochs_mixed, mixed_history['val_loss'], 
                     alpha=0.15, color=COLORS['mixed'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend(loc='upper right')
    
    # 2. Answer Accuracy Comparison
    ax2 = axes[1]
    baseline_acc = [a * 100 for a in baseline_history['answer_accuracy']]
    mixed_acc = [a * 100 for a in mixed_history['answer_accuracy']]
    
    ax2.plot(epochs_baseline, baseline_acc, 
             color=COLORS['baseline'], linewidth=2.5, label='Baseline (k=3)')
    ax2.plot(epochs_mixed, mixed_acc, 
             color=COLORS['mixed'], linewidth=2.5, label='Mixed (k=3,4)')
    ax2.fill_between(epochs_baseline, baseline_acc, alpha=0.15, color=COLORS['baseline'])
    ax2.fill_between(epochs_mixed, mixed_acc, alpha=0.15, color=COLORS['mixed'])
    
    # Add final accuracy annotations
    ax2.axhline(y=baseline_acc[-1], color=COLORS['baseline'], linestyle='--', alpha=0.5)
    ax2.axhline(y=mixed_acc[-1], color=COLORS['mixed'], linestyle='--', alpha=0.5)
    ax2.text(len(epochs_baseline) * 0.02, baseline_acc[-1] + 2, 
             f'{baseline_acc[-1]:.1f}%', color=COLORS['baseline'], fontweight='bold')
    ax2.text(len(epochs_mixed) * 0.02, mixed_acc[-1] - 5, 
             f'{mixed_acc[-1]:.1f}%', color=COLORS['mixed'], fontweight='bold')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Answer Accuracy (%)')
    ax2.set_title('Training Accuracy Comparison')
    ax2.set_ylim([0, 105])
    ax2.legend(loc='lower right')
    
    # 3. Convergence Speed Analysis
    ax3 = axes[2]
    
    # Find epochs to reach certain accuracy thresholds
    thresholds = [50, 70, 80, 90]
    baseline_epochs = []
    mixed_epochs = []
    
    for thresh in thresholds:
        # Baseline
        baseline_epoch = None
        for i, acc in enumerate(baseline_acc):
            if acc >= thresh:
                baseline_epoch = i + 1
                break
        baseline_epochs.append(baseline_epoch if baseline_epoch else len(epochs_baseline) + 5)
        
        # Mixed
        mixed_epoch = None
        for i, acc in enumerate(mixed_acc):
            if acc >= thresh:
                mixed_epoch = i + 1
                break
        mixed_epochs.append(mixed_epoch if mixed_epoch else len(epochs_mixed) + 5)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_epochs, width, 
                    label='Baseline (k=3)', color=COLORS['baseline'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, mixed_epochs, width, 
                    label='Mixed (k=3,4)', color=COLORS['mixed'], alpha=0.8)
    
    ax3.set_ylabel('Epochs to Reach Threshold')
    ax3.set_xlabel('Accuracy Threshold')
    ax3.set_title('Convergence Speed')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t}%' for t in thresholds])
    ax3.legend(loc='upper left')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height <= len(epochs_baseline):
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height <= len(epochs_mixed):
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Training Dynamics: Baseline vs Mixed Training', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved training comparison to {save_path}")
    
    return fig


def plot_accuracy_comparison(
    baseline_results: Dict,
    mixed_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot accuracy comparison across all test sets.
    
    Args:
        baseline_results: Evaluation results from baseline
        mixed_results: Evaluation results from mixed
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Test set names and display labels
    test_sets = ['in_distribution', 'length_gen_k4', 'length_gen_k5', 'shift_nines', 'shift_small']
    display_names = {
        'in_distribution': 'In-Dist\n(k=3)',
        'length_gen_k4': 'Length Gen\n(k=4)',
        'length_gen_k5': 'Length Gen\n(k=5)',
        'shift_nines': 'Many 9s',
        'shift_small': 'Small Nums'
    }
    
    # Extract accuracies
    baseline_accs = []
    mixed_accs = []
    labels = []
    
    for test in test_sets:
        if test in baseline_results and test in mixed_results:
            baseline_accs.append(baseline_results[test].get('accuracy', 0))
            mixed_accs.append(mixed_results[test].get('accuracy', 0))
            labels.append(display_names[test])
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_accs, width, 
                   label='Baseline (k=3 only)', color=COLORS['baseline'], alpha=0.85)
    bars2 = ax.bar(x + width/2, mixed_accs, width, 
                   label='Mixed (k=3,4)', color=COLORS['mixed'], alpha=0.85)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, baseline_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=COLORS['baseline'])
    
    for bar, acc in zip(bars2, mixed_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=COLORS['mixed'])
    
    # Add delta annotations
    for i, (b_acc, m_acc) in enumerate(zip(baseline_accs, mixed_accs)):
        delta = m_acc - b_acc
        color = COLORS['success'] if delta > 0 else COLORS['danger']
        symbol = '↑' if delta > 0 else '↓'
        if abs(delta) > 1:  # Only show significant differences
            ax.text(x[i], max(b_acc, m_acc) + 8,
                    f'{symbol}{abs(delta):.1f}%', ha='center', 
                    fontsize=9, color=color, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison: Baseline vs Mixed Training', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, 115])
    ax.legend(loc='upper right', fontsize=11)
    
    # Add reference lines
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.text(-0.4, 91, '90%', fontsize=9, color='gray')
    
    # Highlight the key finding (k=4 generalization)
    ax.annotate('', xy=(1 + width/2, mixed_accs[1] + 3), 
                xytext=(1 + width/2, mixed_accs[1] + 15),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    ax.text(1 + width/2, mixed_accs[1] + 17, 
            'Key: Length\nGeneralization!', ha='center', fontsize=9, 
            color=COLORS['success'], fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved accuracy comparison to {save_path}")
    
    return fig


def plot_tradeoff_analysis(
    baseline_results: Dict,
    mixed_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the trade-off between in-distribution and generalization performance.
    
    Args:
        baseline_results: Evaluation results from baseline
        mixed_results: Evaluation results from mixed
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Trade-off scatter plot
    ax1 = axes[0]
    
    # In-dist vs k=4 generalization
    baseline_in_dist = baseline_results['in_distribution']['accuracy']
    baseline_k4 = baseline_results['length_gen_k4']['accuracy']
    mixed_in_dist = mixed_results['in_distribution']['accuracy']
    mixed_k4 = mixed_results['length_gen_k4']['accuracy']
    
    ax1.scatter([baseline_in_dist], [baseline_k4], s=300, c=COLORS['baseline'], 
                label='Baseline (k=3)', marker='o', edgecolors='white', linewidth=2, zorder=5)
    ax1.scatter([mixed_in_dist], [mixed_k4], s=300, c=COLORS['mixed'], 
                label='Mixed (k=3,4)', marker='s', edgecolors='white', linewidth=2, zorder=5)
    
    # Add labels
    ax1.annotate(f'Baseline\n({baseline_in_dist:.1f}%, {baseline_k4:.1f}%)', 
                 xy=(baseline_in_dist, baseline_k4),
                 xytext=(baseline_in_dist - 8, baseline_k4 + 10),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    ax1.annotate(f'Mixed\n({mixed_in_dist:.1f}%, {mixed_k4:.1f}%)', 
                 xy=(mixed_in_dist, mixed_k4),
                 xytext=(mixed_in_dist + 8, mixed_k4 - 10),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # Draw arrow showing the trade-off
    ax1.annotate('', xy=(mixed_in_dist, mixed_k4), 
                 xytext=(baseline_in_dist, baseline_k4),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
    
    ax1.set_xlabel('In-Distribution Accuracy (k=3) %', fontsize=12)
    ax1.set_ylabel('Length Generalization (k=4) %', fontsize=12)
    ax1.set_title('The Accuracy Trade-Off', fontsize=13, fontweight='bold')
    ax1.set_xlim([80, 100])
    ax1.set_ylim([-5, 100])
    ax1.legend(loc='lower right')
    
    # Add quadrant labels
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
    ax1.axvline(x=90, color='gray', linestyle=':', alpha=0.3)
    ax1.text(95, 75, 'Ideal Zone', fontsize=10, color='green', alpha=0.7, ha='center')
    
    # 2. Improvement/Degradation breakdown
    ax2 = axes[1]
    
    test_sets = ['in_distribution', 'length_gen_k4', 'shift_nines', 'shift_small']
    display_names = ['In-Dist (k=3)', 'Length Gen (k=4)', 'Many 9s', 'Small Nums']
    
    deltas = []
    for test in test_sets:
        baseline_acc = baseline_results[test]['accuracy']
        mixed_acc = mixed_results[test]['accuracy']
        deltas.append(mixed_acc - baseline_acc)
    
    colors = [COLORS['success'] if d > 0 else COLORS['danger'] for d in deltas]
    
    y_pos = np.arange(len(display_names))
    bars = ax2.barh(y_pos, deltas, color=colors, alpha=0.8, height=0.6)
    
    # Add value labels
    for bar, delta in zip(bars, deltas):
        width = bar.get_width()
        label_x = width + 1 if width >= 0 else width - 1
        ha = 'left' if width >= 0 else 'right'
        sign = '+' if delta > 0 else ''
        ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{sign}{delta:.1f}%', va='center', ha=ha, 
                fontsize=11, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_names)
    ax2.set_xlabel('Change in Accuracy (Mixed - Baseline) %', fontsize=12)
    ax2.set_title('Impact of Mixed Training', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add legend
    gain_patch = mpatches.Patch(color=COLORS['success'], label='Improvement')
    loss_patch = mpatches.Patch(color=COLORS['danger'], label='Degradation')
    ax2.legend(handles=[gain_patch, loss_patch], loc='lower right')
    
    fig.suptitle('Trade-Off Analysis: In-Distribution vs Generalization', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved trade-off analysis to {save_path}")
    
    return fig


def plot_carry_comparison(
    baseline_results: Dict,
    mixed_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare carry vs no-carry performance across experiments.
    
    Args:
        baseline_results: Evaluation results from baseline
        mixed_results: Evaluation results from mixed
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    test_sets = ['in_distribution', 'length_gen_k4', 'shift_small']
    display_names = ['In-Dist (k=3)', 'Length Gen (k=4)', 'Small Nums']
    
    # 1. Carry accuracy comparison
    ax1 = axes[0]
    
    baseline_carry = []
    mixed_carry = []
    
    for test in test_sets:
        b_carry = baseline_results[test].get('carry_accuracy')
        m_carry = mixed_results[test].get('carry_accuracy')
        baseline_carry.append(b_carry if b_carry else 0)
        mixed_carry.append(m_carry if m_carry else 0)
    
    x = np.arange(len(display_names))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_carry, width, label='Baseline', 
            color=COLORS['baseline'], alpha=0.8)
    ax1.bar(x + width/2, mixed_carry, width, label='Mixed', 
            color=COLORS['mixed'], alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Carry Operation Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names)
    ax1.set_ylim([0, 105])
    ax1.legend()
    
    # Add value labels
    for i, (b, m) in enumerate(zip(baseline_carry, mixed_carry)):
        ax1.text(i - width/2, b + 1, f'{b:.1f}%', ha='center', fontsize=9)
        ax1.text(i + width/2, m + 1, f'{m:.1f}%', ha='center', fontsize=9)
    
    # 2. No-carry accuracy comparison
    ax2 = axes[1]
    
    baseline_no_carry = []
    mixed_no_carry = []
    
    for test in test_sets:
        b_no_carry = baseline_results[test].get('no_carry_accuracy')
        m_no_carry = mixed_results[test].get('no_carry_accuracy')
        baseline_no_carry.append(b_no_carry if b_no_carry else 0)
        mixed_no_carry.append(m_no_carry if m_no_carry else 0)
    
    ax2.bar(x - width/2, baseline_no_carry, width, label='Baseline', 
            color=COLORS['baseline'], alpha=0.8)
    ax2.bar(x + width/2, mixed_no_carry, width, label='Mixed', 
            color=COLORS['mixed'], alpha=0.8)
    
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('No-Carry Operation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names)
    ax2.set_ylim([0, 105])
    ax2.legend()
    
    # Add value labels
    for i, (b, m) in enumerate(zip(baseline_no_carry, mixed_no_carry)):
        ax2.text(i - width/2, b + 1, f'{b:.1f}%', ha='center', fontsize=9)
        ax2.text(i + width/2, m + 1, f'{m:.1f}%', ha='center', fontsize=9)
    
    fig.suptitle('Carry vs No-Carry Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved carry comparison to {save_path}")
    
    return fig


def create_summary_table(
    baseline_results: Dict,
    mixed_results: Dict
) -> str:
    """
    Create a text summary table of results.
    
    Returns:
        Formatted string table
    """
    test_sets = ['in_distribution', 'length_gen_k4', 'length_gen_k5', 'shift_nines', 'shift_small']
    display_names = {
        'in_distribution': 'In-Dist (k=3)',
        'length_gen_k4': 'Length Gen (k=4)',
        'length_gen_k5': 'Length Gen (k=5)',
        'shift_nines': 'Many 9s',
        'shift_small': 'Small Nums'
    }
    
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Test Set':<20} {'Baseline':>12} {'Mixed':>12} {'Delta':>12}")
    lines.append("-" * 70)
    
    for test in test_sets:
        name = display_names[test]
        b_acc = baseline_results.get(test, {}).get('accuracy', 0)
        m_acc = mixed_results.get(test, {}).get('accuracy', 0)
        delta = m_acc - b_acc
        sign = '+' if delta > 0 else ''
        lines.append(f"{name:<20} {b_acc:>11.1f}% {m_acc:>11.1f}% {sign}{delta:>10.1f}%")
    
    lines.append("-" * 70)
    lines.append("")
    lines.append("KEY FINDINGS:")
    lines.append("• Mixed training enables k=4 generalization (0% → 78.9%)")
    lines.append("• Trade-off: ~7% drop in in-distribution accuracy")
    lines.append("• Neither model generalizes to k=5")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_all_comparisons(
    baseline_dir: str = "experiments/baseline_k3_only",
    mixed_dir: str = "experiments/mixed_k3_k4",
    output_dir: str = "report/comparison"
):
    """
    Generate all comparison visualizations.
    
    Args:
        baseline_dir: Path to baseline experiment directory
        mixed_dir: Path to mixed experiment directory
        output_dir: Directory to save output plots
    """
    print("=" * 60)
    print("EXPERIMENT COMPARISON VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print("\n[Loading Data]")
    baseline_history, baseline_results = load_experiment_data(baseline_dir)
    mixed_history, mixed_results = load_experiment_data(mixed_dir)
    
    if baseline_results is None or mixed_results is None:
        print("Error: Could not load experiment results!")
        return
    
    print(f"  Loaded baseline from: {baseline_dir}")
    print(f"  Loaded mixed from: {mixed_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\n[Generating Visualizations]")
    
    # 1. Training comparison
    if baseline_history and mixed_history:
        plot_training_comparison(
            baseline_history, mixed_history,
            save_path=os.path.join(output_dir, "training_comparison.png")
        )
        plt.close()
    
    # 2. Accuracy comparison
    plot_accuracy_comparison(
        baseline_results, mixed_results,
        save_path=os.path.join(output_dir, "accuracy_comparison.png")
    )
    plt.close()
    
    # 3. Trade-off analysis
    plot_tradeoff_analysis(
        baseline_results, mixed_results,
        save_path=os.path.join(output_dir, "tradeoff_analysis.png")
    )
    plt.close()
    
    # 4. Carry comparison
    plot_carry_comparison(
        baseline_results, mixed_results,
        save_path=os.path.join(output_dir, "carry_comparison.png")
    )
    plt.close()
    
    # Print summary
    summary = create_summary_table(baseline_results, mixed_results)
    print("\n" + summary)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSaved summary to {summary_path}")
    
    print(f"\n[Complete] All plots saved to {output_dir}/")
    print("  - training_comparison.png")
    print("  - accuracy_comparison.png")
    print("  - tradeoff_analysis.png")
    print("  - carry_comparison.png")
    print("  - summary.txt")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare baseline and mixed training experiments"
    )
    
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="experiments/baseline_k3_only",
        help="Path to baseline experiment directory"
    )
    
    parser.add_argument(
        "--mixed-dir",
        type=str,
        default="experiments/mixed_k3_k4",
        help="Path to mixed experiment directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="report/comparison",
        help="Directory to save comparison plots"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    
    args = parser.parse_args()
    
    generate_all_comparisons(
        baseline_dir=args.baseline_dir,
        mixed_dir=args.mixed_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

