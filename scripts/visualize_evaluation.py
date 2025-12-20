"""
Evaluation Results Visualization

Creates charts and visualizations from model evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_latest_results(results_dir: str = "./evaluation_results"):
    """Load the most recent evaluation results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Run evaluation first: python scripts/evaluate_models.py")
        return None, None, None
    
    # Find latest comparison file
    comparison_files = list(results_path.glob("model_comparison_*.csv"))
    if not comparison_files:
        print(f"❌ No comparison files found in {results_dir}")
        return None, None, None
    
    latest_comparison = sorted(comparison_files)[-1]
    comparison_df = pd.read_csv(latest_comparison)
    
    # Load individual model results
    timestamp = latest_comparison.stem.split("_")[-1]
    
    results_dict = {}
    for model_name in comparison_df['model'].unique():
        model_file = results_path / f"{model_name}_results_{timestamp}.csv"
        if model_file.exists():
            results_dict[model_name] = pd.read_csv(model_file)
    
    # Load summary
    summary_file = results_path / f"evaluation_summary_{timestamp}.json"
    summary = None
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    
    print(f"✓ Loaded results from: {latest_comparison.parent}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Models: {list(results_dict.keys())}")
    
    return comparison_df, results_dict, summary


def plot_metric_comparison(comparison_df: pd.DataFrame, output_dir: str = "./evaluation_results"):
    """Plot side-by-side comparison of metrics across models."""
    
    metrics = [col for col in comparison_df.columns 
              if col not in ['model', 'n_users'] and '@' in col]
    
    # Group by metric type
    precision_metrics = [m for m in metrics if 'precision' in m]
    recall_metrics = [m for m in metrics if 'recall' in m]
    ndcg_metrics = [m for m in metrics if 'ndcg' in m]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision
    ax = axes[0, 0]
    comparison_df[['model'] + precision_metrics].set_index('model').plot(
        kind='bar', ax=ax, rot=45
    )
    ax.set_title('Precision@K Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision')
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Recall
    ax = axes[0, 1]
    comparison_df[['model'] + recall_metrics].set_index('model').plot(
        kind='bar', ax=ax, rot=45
    )
    ax.set_title('Recall@K Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall')
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # NDCG
    ax = axes[1, 0]
    comparison_df[['model'] + ndcg_metrics].set_index('model').plot(
        kind='bar', ax=ax, rot=45
    )
    ax.set_title('NDCG@K Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('NDCG')
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Average Precision
    ax = axes[1, 1]
    comparison_df[['model', 'avg_precision']].set_index('model').plot(
        kind='bar', ax=ax, rot=45, legend=False
    )
    ax.set_title('Average Precision', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Precision')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "metrics_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_metric_heatmap(comparison_df: pd.DataFrame, output_dir: str = "./evaluation_results"):
    """Create heatmap of all metrics across models."""
    
    metrics = [col for col in comparison_df.columns 
              if col not in ['model', 'n_users']]
    
    # Prepare data for heatmap
    heatmap_data = comparison_df[['model'] + metrics].set_index('model')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Score'},
        ax=ax
    )
    
    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "metrics_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_per_user_distribution(results_dict: dict, output_dir: str = "./evaluation_results"):
    """Plot distribution of metrics across users for each model."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for model_name, results_df in results_dict.items():
            if metric in results_df.columns:
                ax.hist(
                    results_df[metric],
                    bins=20,
                    alpha=0.5,
                    label=model_name,
                    edgecolor='black'
                )
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "per_user_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_method_breakdown(results_dict: dict, output_dir: str = "./evaluation_results"):
    """Plot breakdown of recommendation methods used."""
    
    fig, axes = plt.subplots(1, len(results_dict), figsize=(15, 5))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, results_df) in enumerate(results_dict.items()):
        if 'method' in results_df.columns:
            method_counts = results_df['method'].value_counts()
            
            ax = axes[idx]
            method_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{model_name}\nMethod Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "method_breakdown.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_report(
    comparison_df: pd.DataFrame,
    results_dict: dict,
    summary: dict,
    output_dir: str = "./evaluation_results"
):
    """Generate markdown evaluation report."""
    
    report_lines = [
        "# Model Evaluation Report",
        "",
        f"**Generated:** {summary['timestamp'] if summary else 'N/A'}",
        "",
        "## Models Evaluated",
        ""
    ]
    
    for model in comparison_df['model']:
        report_lines.append(f"- **{model}**")
    
    report_lines.extend([
        "",
        "## Performance Summary",
        "",
        "### Overall Metrics",
        ""
    ])
    
    # Add comparison table
    report_lines.append(comparison_df.to_markdown(index=False))
    
    report_lines.extend([
        "",
        "## Best Performing Models",
        ""
    ])
    
    # Find best for each metric
    metrics = [col for col in comparison_df.columns if col not in ['model', 'n_users']]
    for metric in metrics:
        best_idx = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_idx, 'model']
        best_score = comparison_df.loc[best_idx, metric]
        report_lines.append(f"- **{metric}**: {best_model} ({best_score:.4f})")
    
    report_lines.extend([
        "",
        "## Visualizations",
        "",
        "![Metrics Comparison](metrics_comparison.png)",
        "",
        "![Metrics Heatmap](metrics_heatmap.png)",
        "",
        "![Per-User Distribution](per_user_distribution.png)",
        "",
        "![Method Breakdown](method_breakdown.png)",
        "",
        "## Recommendations",
        ""
    ])
    
    # Add recommendations based on results
    best_overall = comparison_df.loc[comparison_df['ndcg@10'].idxmax(), 'model']
    report_lines.extend([
        f"- **Best Overall Model**: {best_overall}",
        "- **Production Deployment**: Use Hybrid model for best balance",
        "- **Cold Start**: Hybrid falls back to popularity automatically",
        "- **Next Steps**: A/B test different weight combinations",
        ""
    ])
    
    report_content = "\n".join(report_lines)
    
    output_file = Path(output_dir) / "EVALUATION_REPORT.md"
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Saved: {output_file}")


def main():
    """Generate all visualizations and report."""
    
    print("="*60)
    print("Mousiki Evaluation Visualizations")
    print("="*60)
    print()
    
    # Load results
    comparison_df, results_dict, summary = load_latest_results()
    
    if comparison_df is None:
        sys.exit(1)
    
    output_dir = "./evaluation_results"
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metric_comparison(comparison_df, output_dir)
    plot_metric_heatmap(comparison_df, output_dir)
    
    if results_dict:
        plot_per_user_distribution(results_dict, output_dir)
        plot_method_breakdown(results_dict, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(comparison_df, results_dict, summary, output_dir)
    
    print("\n" + "="*60)
    print("✓ Visualizations Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"View report: {output_dir}/EVALUATION_REPORT.md")


if __name__ == "__main__":
    main()
