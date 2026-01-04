"""
Visualization utilities for the Addition Transformer project.

Comprehensive plotting module for analyzing:
- Training dynamics (loss, accuracy, learning rate)
- Evaluation performance across test sets
- Model predictions and error analysis
- Attention patterns (if extracted during evaluation)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional and distinct
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red-orange
    'neutral': '#3B1F2B',      # Dark purple
    'train': '#2E86AB',
    'val': '#E94F37',
    'accuracy': '#44AF69',
    'answer_acc': '#9B5DE5',
    'lr': '#F18F01',
}


class TrainingVisualizer:
    """Visualizer for training history and metrics."""
    
    def __init__(self, history_path: str = "checkpoints/training_history.json"):
        """
        Initialize with training history.
        
        Args:
            history_path: Path to training history JSON file
        """
        self.history_path = history_path
        self.history = None
        self._load_history()
    
    def _load_history(self):
        """Load training history from JSON file."""
        if not os.path.exists(self.history_path):
            print(f"Warning: Training history not found at {self.history_path}")
            return
        
        with open(self.history_path, 'r') as f:
            self.history = json.load(f)
    
    def plot_loss_curves(self, ax: Optional[plt.Axes] = None, 
                         save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot training and validation loss curves.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib axes object
        """
        if self.history is None:
            raise ValueError("No training history loaded")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot with confidence-inspiring styling
        ax.plot(epochs, self.history['train_loss'], 
                color=COLORS['train'], linewidth=2.5, label='Training Loss',
                marker='o', markersize=4, markevery=max(1, len(epochs)//10))
        ax.plot(epochs, self.history['val_loss'], 
                color=COLORS['val'], linewidth=2.5, label='Validation Loss',
                marker='s', markersize=4, markevery=max(1, len(epochs)//10))
        
        # Add fill between for visual effect
        ax.fill_between(epochs, self.history['train_loss'], alpha=0.1, color=COLORS['train'])
        ax.fill_between(epochs, self.history['val_loss'], alpha=0.1, color=COLORS['val'])
        
        # Mark minimum validation loss
        min_val_idx = np.argmin(self.history['val_loss'])
        min_val_loss = self.history['val_loss'][min_val_idx]
        ax.axvline(x=min_val_idx + 1, color='gray', linestyle=':', alpha=0.7)
        ax.scatter([min_val_idx + 1], [min_val_loss], color=COLORS['val'], 
                   s=100, zorder=5, edgecolors='white', linewidth=2)
        ax.annotate(f'Best: {min_val_loss:.4f}', 
                    xy=(min_val_idx + 1, min_val_loss),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color=COLORS['val'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend(loc='upper right', framealpha=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved loss curves to {save_path}")
        
        return ax
    
    def plot_accuracy_curves(self, ax: Optional[plt.Axes] = None,
                             save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot validation and answer accuracy curves.
        
        Args:
            ax: Matplotlib axes to plot on
            save_path: Path to save figure
            
        Returns:
            Matplotlib axes object
        """
        if self.history is None:
            raise ValueError("No training history loaded")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.history['val_accuracy']) + 1)
        
        # Convert to percentages
        val_acc = [acc * 100 for acc in self.history['val_accuracy']]
        answer_acc = [acc * 100 for acc in self.history['answer_accuracy']]
        
        ax.plot(epochs, val_acc, color=COLORS['accuracy'], linewidth=2.5,
                label='Sequence Accuracy', marker='o', markersize=4,
                markevery=max(1, len(epochs)//10))
        ax.plot(epochs, answer_acc, color=COLORS['answer_acc'], linewidth=2.5,
                label='Answer Accuracy', marker='s', markersize=4,
                markevery=max(1, len(epochs)//10))
        
        # Fill to show progress
        ax.fill_between(epochs, answer_acc, alpha=0.15, color=COLORS['answer_acc'])
        
        # Mark final accuracy
        final_answer_acc = answer_acc[-1]
        ax.axhline(y=final_answer_acc, color=COLORS['answer_acc'], 
                   linestyle='--', alpha=0.5)
        ax.text(len(epochs) * 0.02, final_answer_acc + 2, 
                f'Final: {final_answer_acc:.1f}%', 
                color=COLORS['answer_acc'], fontsize=10)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Over Training')
        ax.set_ylim([0, 105])
        ax.legend(loc='lower right', framealpha=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved accuracy curves to {save_path}")
        
        return ax
    
    def plot_learning_rate(self, ax: Optional[plt.Axes] = None,
                           save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot learning rate schedule.
        
        Args:
            ax: Matplotlib axes to plot on
            save_path: Path to save figure
            
        Returns:
            Matplotlib axes object
        """
        if self.history is None:
            raise ValueError("No training history loaded")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        epochs = range(1, len(self.history['learning_rate']) + 1)
        
        ax.plot(epochs, self.history['learning_rate'], 
                color=COLORS['lr'], linewidth=2.5)
        ax.fill_between(epochs, self.history['learning_rate'], 
                        alpha=0.2, color=COLORS['lr'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        
        # Add annotations for start and end LR
        start_lr = self.history['learning_rate'][0]
        end_lr = self.history['learning_rate'][-1]
        ax.annotate(f'{start_lr:.1e}', xy=(1, start_lr), 
                    xytext=(-5, 5), textcoords='offset points', fontsize=9)
        ax.annotate(f'{end_lr:.1e}', xy=(len(epochs), end_lr),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved learning rate plot to {save_path}")
        
        return ax
    
    def plot_training_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive training dashboard with all metrics.
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if self.history is None:
            raise ValueError("No training history loaded")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Loss curves (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_loss_curves(ax=ax1)
        
        # Accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        self.plot_accuracy_curves(ax=ax2)
        
        # Learning rate
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_learning_rate(ax=ax3)
        ax3.set_title('LR Schedule', fontsize=12)
        
        # Training summary stats
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_summary_stats(ax4)
        
        fig.suptitle('Training Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved training dashboard to {save_path}")
        
        return fig
    
    def _plot_summary_stats(self, ax: plt.Axes):
        """Plot summary statistics as text."""
        ax.axis('off')
        
        # Calculate stats
        final_train_loss = self.history['train_loss'][-1]
        best_val_loss = min(self.history['val_loss'])
        final_answer_acc = self.history['answer_accuracy'][-1] * 100
        total_epochs = len(self.history['train_loss'])
        
        stats_text = (
            f"Training Summary\n"
            f"{'─' * 25}\n\n"
            f"Total Epochs: {total_epochs}\n\n"
            f"Final Train Loss: {final_train_loss:.4f}\n\n"
            f"Best Val Loss: {best_val_loss:.4f}\n\n"
            f"Final Answer Acc: {final_answer_acc:.1f}%\n"
        )
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))


class EvaluationVisualizer:
    """Visualizer for evaluation results."""
    
    def __init__(self, results_path: str = "results/evaluation_results.json"):
        """
        Initialize with evaluation results.
        
        Args:
            results_path: Path to evaluation results JSON file
        """
        self.results_path = results_path
        self.results = None
        self._load_results()
        
        # Display names for test sets
        self.display_names = {
            "in_distribution": "In-Dist (k=3)",
            "length_gen_k4": "Length Gen (k=4)",
            "length_gen_k5": "Length Gen (k=5)",
            "shift_nines": "Many 9s",
            "shift_small": "Small Nums"
        }
    
    def _load_results(self):
        """Load evaluation results from JSON file."""
        if not os.path.exists(self.results_path):
            print(f"Warning: Evaluation results not found at {self.results_path}")
            return
        
        with open(self.results_path, 'r') as f:
            data = json.load(f)
        
        self.results = data.get('results', data)
    
    def plot_accuracy_comparison(self, ax: Optional[plt.Axes] = None,
                                  save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot accuracy comparison across test sets as horizontal bars.
        
        Args:
            ax: Matplotlib axes to plot on
            save_path: Path to save figure
            
        Returns:
            Matplotlib axes object
        """
        if self.results is None:
            raise ValueError("No evaluation results loaded")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        test_names = []
        accuracies = []
        
        for key, result in self.results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                test_names.append(self.display_names.get(key, key))
                accuracies.append(result['accuracy'])
        
        if not test_names:
            ax.text(0.5, 0.5, 'No evaluation results available',
                    ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Create horizontal bar chart
        y_pos = np.arange(len(test_names))
        
        # Color bars by performance
        colors = []
        for acc in accuracies:
            if acc >= 90:
                colors.append('#44AF69')  # Green
            elif acc >= 70:
                colors.append('#F18F01')  # Orange
            else:
                colors.append('#E94F37')  # Red
        
        bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.8, height=0.6)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1f}%', va='center', fontweight='bold', fontsize=11)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Model Accuracy by Test Set')
        ax.set_xlim([0, 110])
        
        # Add reference lines
        ax.axvline(x=90, color='#44AF69', linestyle='--', alpha=0.5, label='90% threshold')
        ax.axvline(x=70, color='#F18F01', linestyle='--', alpha=0.5, label='70% threshold')
        
        ax.legend(loc='lower right', fontsize=9)
        ax.invert_yaxis()  # Top to bottom
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved accuracy comparison to {save_path}")
        
        return ax
    
    def plot_carry_analysis(self, ax: Optional[plt.Axes] = None,
                            save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot carry vs no-carry accuracy breakdown.
        
        Args:
            ax: Matplotlib axes to plot on
            save_path: Path to save figure
            
        Returns:
            Matplotlib axes object
        """
        if self.results is None:
            raise ValueError("No evaluation results loaded")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract carry/no-carry data
        test_names = []
        carry_accs = []
        no_carry_accs = []
        
        for key, result in self.results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                carry_acc = result.get('carry_accuracy')
                no_carry_acc = result.get('no_carry_accuracy')
                
                if carry_acc is not None or no_carry_acc is not None:
                    test_names.append(self.display_names.get(key, key))
                    carry_accs.append(carry_acc if carry_acc else 0)
                    no_carry_accs.append(no_carry_acc if no_carry_acc else 0)
        
        if not test_names:
            ax.text(0.5, 0.5, 'No carry/no-carry data available',
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=12, style='italic')
            ax.set_title('Carry vs No-Carry Analysis')
            return ax
        
        x = np.arange(len(test_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, carry_accs, width, label='With Carry',
                       color='#E94F37', alpha=0.8)
        bars2 = ax.bar(x + width/2, no_carry_accs, width, label='No Carry',
                       color='#44AF69', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Carry vs No-Carry Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.set_ylim([0, 110])
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved carry analysis to {save_path}")
        
        return ax
    
    def plot_generalization_gap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the generalization gap between in-distribution and OOD tests.
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if self.results is None:
            raise ValueError("No evaluation results loaded")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get in-distribution accuracy as baseline
        in_dist_acc = None
        if 'in_distribution' in self.results:
            in_dist_acc = self.results['in_distribution'].get('accuracy', 0)
        
        if in_dist_acc is None:
            ax.text(0.5, 0.5, 'In-distribution baseline not available',
                    ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate gaps
        test_names = []
        gaps = []
        accuracies = []
        
        for key, result in self.results.items():
            if key == 'in_distribution':
                continue
            if isinstance(result, dict) and 'accuracy' in result:
                acc = result['accuracy']
                gap = in_dist_acc - acc
                test_names.append(self.display_names.get(key, key))
                gaps.append(gap)
                accuracies.append(acc)
        
        if not test_names:
            ax.text(0.5, 0.5, 'No OOD test results available',
                    ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create waterfall-style chart
        y_pos = np.arange(len(test_names))
        
        # Color by gap magnitude
        colors = ['#E94F37' if g > 20 else '#F18F01' if g > 10 else '#44AF69' for g in gaps]
        
        bars = ax.barh(y_pos, gaps, color=colors, alpha=0.8, height=0.6)
        
        # Add baseline reference
        ax.axvline(x=0, color='black', linewidth=2)
        
        # Labels
        for i, (bar, gap, acc) in enumerate(zip(bars, gaps, accuracies)):
            width = bar.get_width()
            label_x = width + 0.5 if width >= 0 else width - 0.5
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'Δ{gap:+.1f}% (→{acc:.1f}%)', va='center', ha=ha, fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Accuracy Drop from In-Distribution (%)')
        ax.set_title(f'Generalization Gap (Baseline: {in_dist_acc:.1f}%)')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved generalization gap plot to {save_path}")
        
        return fig
    
    def plot_evaluation_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive evaluation dashboard.
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if self.results is None:
            raise ValueError("No evaluation results loaded")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_accuracy_comparison(ax=ax1)
        
        # Carry analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_carry_analysis(ax=ax2)
        
        # Generalization gap (spans bottom row)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Recreate generalization gap logic for this axes
        in_dist_acc = None
        if 'in_distribution' in self.results:
            in_dist_acc = self.results['in_distribution'].get('accuracy', 0)
        
        if in_dist_acc is not None:
            test_names = []
            gaps = []
            accuracies = []
            
            for key, result in self.results.items():
                if key == 'in_distribution':
                    continue
                if isinstance(result, dict) and 'accuracy' in result:
                    acc = result['accuracy']
                    gap = in_dist_acc - acc
                    test_names.append(self.display_names.get(key, key))
                    gaps.append(gap)
                    accuracies.append(acc)
            
            if test_names:
                y_pos = np.arange(len(test_names))
                colors = ['#E94F37' if g > 20 else '#F18F01' if g > 10 else '#44AF69' for g in gaps]
                bars = ax3.barh(y_pos, gaps, color=colors, alpha=0.8, height=0.5)
                ax3.axvline(x=0, color='black', linewidth=2)
                
                for i, (bar, gap, acc) in enumerate(zip(bars, gaps, accuracies)):
                    width = bar.get_width()
                    label_x = width + 0.5 if width >= 0 else width - 0.5
                    ha = 'left' if width >= 0 else 'right'
                    ax3.text(label_x, bar.get_y() + bar.get_height()/2,
                            f'Δ{gap:+.1f}% (→{acc:.1f}%)', va='center', ha=ha, fontsize=10)
                
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(test_names)
                ax3.set_xlabel('Accuracy Drop from In-Distribution (%)')
                ax3.set_title(f'Generalization Gap Analysis (Baseline: {in_dist_acc:.1f}%)')
                ax3.invert_yaxis()
        
        fig.suptitle('Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved evaluation dashboard to {save_path}")
        
        return fig


def plot_training_curves(
    history_path: str = "checkpoints/training_history.json",
    save_path: Optional[str] = None
):
    """
    Convenience function to plot training curves.
    
    Args:
        history_path: Path to training history JSON file
        save_path: Path to save the plot
    """
    viz = TrainingVisualizer(history_path)
    if viz.history is not None:
        fig = viz.plot_training_dashboard(save_path)
        if save_path is None:
            plt.show()
        plt.close()


def plot_evaluation_results(
    results_path: str = "results/evaluation_results.json",
    save_path: Optional[str] = None
):
    """
    Convenience function to plot evaluation results.
    
    Args:
        results_path: Path to evaluation results JSON file
        save_path: Path to save the plot
    """
    viz = EvaluationVisualizer(results_path)
    if viz.results is not None:
        fig = viz.plot_evaluation_dashboard(save_path)
        if save_path is None:
            plt.show()
        plt.close()


def plot_all(
    history_path: str = "checkpoints/training_history.json",
    results_path: str = "results/evaluation_results.json",
    output_dir: str = "report"
):
    """
    Generate all visualization plots and save to output directory.
    
    Args:
        history_path: Path to training history JSON
        results_path: Path to evaluation results JSON
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Generating Visualizations")
    print("=" * 50)
    
    # Training visualizations
    if os.path.exists(history_path):
        print("\n[Training Curves]")
        viz = TrainingVisualizer(history_path)
        
        viz.plot_training_dashboard(
            save_path=os.path.join(output_dir, "training_dashboard.png")
        )
        plt.close()
        
        # Individual plots
        fig, ax = plt.subplots(figsize=(10, 6))
        viz.plot_loss_curves(ax=ax, save_path=os.path.join(output_dir, "loss_curves.png"))
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        viz.plot_accuracy_curves(ax=ax, save_path=os.path.join(output_dir, "accuracy_curves.png"))
        plt.close()
    else:
        print(f"\nWarning: Training history not found at {history_path}")
    
    # Evaluation visualizations
    if os.path.exists(results_path):
        print("\n[Evaluation Results]")
        viz = EvaluationVisualizer(results_path)
        
        viz.plot_evaluation_dashboard(
            save_path=os.path.join(output_dir, "evaluation_dashboard.png")
        )
        plt.close()
        
        # Individual plots
        fig, ax = plt.subplots(figsize=(12, 6))
        viz.plot_accuracy_comparison(ax=ax, save_path=os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()
        
        viz.plot_generalization_gap(
            save_path=os.path.join(output_dir, "generalization_gap.png")
        )
        plt.close()
    else:
        print(f"\nWarning: Evaluation results not found at {results_path}")
    
    print("\n" + "=" * 50)
    print(f"All plots saved to {output_dir}/")
    print("=" * 50)


def main():
    """Main entry point for visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate visualizations for Addition Transformer training and evaluation"
    )
    
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/training_history.json",
        help="Path to training history JSON file"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="results/evaluation_results.json",
        help="Path to evaluation results JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="report",
        help="Directory to save plots"
    )
    
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only generate training visualizations"
    )
    
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Only generate evaluation visualizations"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.training_only:
        save_path = None if args.show else os.path.join(args.output_dir, "training_dashboard.png")
        plot_training_curves(history_path=args.history, save_path=save_path)
    elif args.evaluation_only:
        save_path = None if args.show else os.path.join(args.output_dir, "evaluation_dashboard.png")
        plot_evaluation_results(results_path=args.results, save_path=save_path)
    else:
        plot_all(
            history_path=args.history,
            results_path=args.results,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
