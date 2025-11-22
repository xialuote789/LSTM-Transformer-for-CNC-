"""
Module 5: Comprehensive Evaluation and Visualization
Generate all figures and tables for academic publication
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================
# SECTION 1: LOAD RESULTS
# ============================================

def load_training_results():
    """Load all training results"""

    results = {
        'Teacher': {
            'val_acc': 0.9996,
            'test_acc': None,  # Teacher wasn't tested
            'parameters': 1337218,
            'training_time': 35  # minutes
        },
        'Light': {
            'val_acc': 0.9997,
            'test_acc': 0.9996,
            'parameters': 605186,
            'training_time': 51
        },
        'Ultra-Light': {
            'val_acc': 0.9997,
            'test_acc': 0.9993,
            'parameters': 114786,
            'training_time': 31
        },
        'Nano': {
            'val_acc': 0.9995,
            'test_acc': 0.9993,
            'parameters': 23620,
            'training_time': 34
        }
    }

    return results

# ============================================
# SECTION 2: PERFORMANCE COMPARISON TABLE
# ============================================

def create_comparison_table(results):
    """Create comprehensive comparison table"""

    print("="*80)
    print("TABLE 1: COMPREHENSIVE MODEL COMPARISON")
    print("="*80)

    # Create DataFrame
    data = []
    teacher_params = results['Teacher']['parameters']

    for model_name, metrics in results.items():
        if model_name == 'Teacher':
            continue

        compression_ratio = teacher_params / metrics['parameters']
        param_reduction = (1 - metrics['parameters']/teacher_params) * 100

        data.append({
            'Model': model_name,
            'Parameters': f"{metrics['parameters']:,}",
            'Compression': f"{compression_ratio:.1f}×",
            'Reduction': f"{param_reduction:.1f}%",
            'Val Acc': f"{metrics['val_acc']:.4f}",
            'Test Acc': f"{metrics['test_acc']:.4f}",
            'Training (min)': metrics['training_time']
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE FOR PAPER")
    print("="*80)

    print("""
\\begin{table}[h]
\\centering
\\caption{Performance comparison of lightweight LSTM-Transformer variants}
\\begin{tabular}{lrrrrr}
\\hline
Model & Parameters & Compression & Val Acc & Test Acc & Time (min) \\\\
\\hline""")

    for _, row in df.iterrows():
        print(f"{row['Model']} & {row['Parameters']} & {row['Compression']} & "
              f"{row['Val Acc']} & {row['Test Acc']} & {row['Training (min)']} \\\\")

    print("""\\hline
\\end{tabular}
\\label{tab:model_comparison}
\\end{table}""")

    return df

# ============================================
# SECTION 3: VISUALIZATION FUNCTIONS
# ============================================

def plot_model_comparison(results):
    """Create comprehensive model comparison figure"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Prepare data
    models = ['Light', 'Ultra-Light', 'Nano']
    teacher_params = results['Teacher']['parameters']

    # Colors for each model
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 1. Parameters
    ax = axes[0, 0]
    params = [results[m]['parameters'] for m in models]
    bars = ax.bar(models, params, color=colors)
    ax.axhline(y=teacher_params, color='red', linestyle='--', alpha=0.5, label='Teacher')
    ax.set_ylabel('Parameters')
    ax.set_title('(a) Model Size')
    ax.legend()

    for bar, param in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{param/1e6:.2f}M', ha='center', va='bottom')

    # 2. Compression Ratio
    ax = axes[0, 1]
    compressions = [teacher_params/results[m]['parameters'] for m in models]
    bars = ax.bar(models, compressions, color=colors)
    ax.set_ylabel('Compression Ratio')
    ax.set_title('(b) Compression vs Teacher')

    for bar, comp in zip(bars, compressions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{comp:.1f}×', ha='center', va='bottom')

    # 3. Test Accuracy
    ax = axes[0, 2]
    test_accs = [results[m]['test_acc'] for m in models]
    bars = ax.bar(models, test_accs, color=colors)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim([0.998, 1.0])
    ax.set_title('(c) Test Performance')

    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{acc:.4f}', ha='center', va='bottom')

    # 4. Parameter Reduction
    ax = axes[1, 0]
    reductions = [(1 - results[m]['parameters']/teacher_params)*100 for m in models]
    bars = ax.bar(models, reductions, color=colors)
    ax.set_ylabel('Parameter Reduction (%)')
    ax.set_title('(d) Parameter Reduction')

    for bar, red in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{red:.1f}%', ha='center', va='bottom')

    # 5. Training Time
    ax = axes[1, 1]
    times = [results[m]['training_time'] for m in models]
    bars = ax.bar(models, times, color=colors)
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('(e) Training Efficiency')

    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time}m', ha='center', va='bottom')

    # 6. Accuracy vs Parameters (scatter)
    ax = axes[1, 2]
    for i, m in enumerate(models):
        ax.scatter(results[m]['parameters']/1e6, results[m]['test_acc'],
                  s=200, c=colors[i], label=m, edgecolors='black', linewidth=2)

    ax.scatter(teacher_params/1e6, results['Teacher']['val_acc'],
              s=200, c='red', marker='s', label='Teacher', edgecolors='black', linewidth=2)

    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Accuracy')
    ax.set_title('(f) Accuracy vs Model Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    save_path = '/content/drive/MyDrive/CNC_Machining/figures/'
    Path(save_path).mkdir(exist_ok=True)
    plt.savefig(f'{save_path}model_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}model_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Figure saved to {save_path}")

def plot_efficiency_analysis(results):
    """Create efficiency analysis figure"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Teacher', 'Light', 'Ultra-Light', 'Nano']

    # 1. Pareto frontier: Accuracy vs Parameters
    ax = axes[0]
    params_list = [results[m]['parameters']/1e6 for m in models]
    acc_list = [results[m]['val_acc'] if m == 'Teacher' else results[m]['test_acc'] for m in models]
    colors_list = ['red', '#2E86AB', '#A23B72', '#F18F01']

    for i, m in enumerate(models):
        ax.scatter(params_list[i], acc_list[i], s=200, c=colors_list[i],
                  label=m, edgecolors='black', linewidth=2, alpha=0.7)

    # Draw Pareto frontier
    ax.plot(params_list, acc_list, 'k--', alpha=0.3)

    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('(a) Pareto Frontier: Accuracy vs Model Size', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.5])
    ax.set_ylim([0.998, 1.0])

    # Add annotations
    for i, m in enumerate(models):
        if m != 'Teacher':
            reduction = (1 - results[m]['parameters']/results['Teacher']['parameters'])*100
            ax.annotate(f'-{reduction:.0f}%',
                       xy=(params_list[i], acc_list[i]),
                       xytext=(params_list[i]+0.05, acc_list[i]-0.0002),
                       fontsize=9)

    # 2. Speedup analysis
    ax = axes[1]
    teacher_time = results['Teacher']['training_time']
    speedups = [teacher_time/results[m]['training_time'] for m in models[1:]]

    bars = ax.bar(models[1:], speedups, color=colors_list[1:])
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('(b) Training Speedup vs Teacher', fontsize=13)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{speedup:.2f}×', ha='center', va='bottom')

    plt.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    save_path = '/content/drive/MyDrive/CNC_Machining/figures/'
    plt.savefig(f'{save_path}efficiency_analysis.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}efficiency_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_summary_report(results):
    """Create final summary report"""

    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)

    print("\n1. BEST PERFORMING MODEL:")
    print("-" * 40)
    print(f"   Light Model: 99.96% test accuracy with 54.7% parameter reduction")

    print("\n2. MOST EFFICIENT MODEL:")
    print("-" * 40)
    print(f"   Nano Model: 99.93% test accuracy with 98.2% parameter reduction")

    print("\n3. BEST BALANCE:")
    print("-" * 40)
    print(f"   Ultra-Light Model: 99.93% test accuracy with 91.4% parameter reduction")

    print("\n4. KEY ACHIEVEMENTS:")
    print("-" * 40)
    print("   • All models maintain >99.9% accuracy")
    print("   • Up to 98.2% parameter reduction achieved")
    print("   • Knowledge distillation highly effective")
    print("   • Training time comparable or better than teacher")

    print("\n5. RECOMMENDATION:")
    print("-" * 40)
    print("   For deployment: Ultra-Light model (best accuracy/size trade-off)")
    print("   For edge devices: Nano model (minimal size, excellent performance)")
    print("   For high-accuracy needs: Light model (highest accuracy)")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution for comprehensive evaluation"""

    print("="*80)
    print("MODULE 5: COMPREHENSIVE EVALUATION AND VISUALIZATION")
    print("="*80)

    # Load results
    results = load_training_results()

    # Create comparison table
    print("\nStep 1: Creating comparison table...")
    df_comparison = create_comparison_table(results)

    # Create visualizations
    print("\nStep 2: Generating visualizations...")
    plot_model_comparison(results)
    plot_efficiency_analysis(results)

    # Create summary report
    print("\nStep 3: Generating summary report...")
    create_summary_report(results)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nAll figures and tables have been generated for your paper!")
    print("Check /content/drive/MyDrive/CNC_Machining/figures/ for outputs")

    return results, df_comparison

if __name__ == "__main__":
    results, comparison_table = main()