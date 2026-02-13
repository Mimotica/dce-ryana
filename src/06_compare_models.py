#!/usr/bin/env python3
"""
Comprehensive comparison of MNL vs Neural Network models.
Generates complete analysis report with performance metrics, WTP comparisons, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def run_mnl_model(data_path, output_dir):
    """Run MNL model using pylogit."""
    print("\n" + "="*80)
    print("STEP 1: Running MNL Model (pylogit)")
    print("="*80)
    
    cmd = [
        sys.executable, 
        'src/02_fit_mnl_pylogit.py',
        '--input', data_path,
        '--mapping', str(output_dir / 'coding_map.csv'),
        '--coef_output', str(output_dir / 'coefficients_mnl.csv'),
        '--wtp_output', str(output_dir / 'wtp_levels_mnl.csv'),
        '--range_output', str(output_dir / 'attribute_ranges_mnl.csv')
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        return None
    
    # Load results
    coef_df = pd.read_csv(output_dir / 'coefficients_mnl.csv')
    wtp_df = pd.read_csv(output_dir / 'wtp_levels_mnl.csv')
    
    return {
        'coefficients': coef_df,
        'wtp': wtp_df,
        'name': 'MNL (pylogit)'
    }


def run_neural_networks(data_path, output_dir, epochs=100):
    """Run both Linear and Deep Neural Network models."""
    print("\n" + "="*80)
    print("STEP 2: Running Neural Network Models")
    print("="*80)
    
    cmd = [
        sys.executable,
        'src/05_neural_network.py',
        '--input', data_path,
        '--model_type', 'both',
        '--epochs', str(epochs),
        '--output_dir', str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        return None, None
    
    # Load results
    perf_df = pd.read_csv(output_dir / 'nn_performance.csv')
    linear_wtp = pd.read_csv(output_dir / 'wtp_linear_nn.csv')
    deep_wtp = pd.read_csv(output_dir / 'wtp_deep_nn.csv')
    
    linear_results = {
        'wtp': linear_wtp,
        'performance': perf_df[perf_df['model'] == 'linear'].iloc[0],
        'name': 'Linear NN'
    }
    
    deep_results = {
        'wtp': deep_wtp,
        'performance': perf_df[perf_df['model'] == 'deep'].iloc[0],
        'name': 'Deep NN'
    }
    
    return linear_results, deep_results


def create_performance_comparison(mnl_results, linear_results, deep_results, output_dir):
    """Create performance comparison table."""
    print("\n" + "="*80)
    print("STEP 3: Creating Performance Comparison")
    print("="*80)
    
    # For MNL, we need to calculate accuracy on validation set
    # For now, we'll use log-likelihood and other metrics
    
    comparison = pd.DataFrame({
        'Model': ['MNL (pylogit)', 'Linear NN', 'Deep NN'],
        'Type': ['Statistical', 'Machine Learning', 'Machine Learning'],
        'Complexity': ['Linear', 'Linear', 'Non-linear'],
        'Hidden Layers': [0, 0, 2],
        'Validation Accuracy (%)': [
            np.nan,  # MNL doesn't have train/val split in our setup
            linear_results['performance']['val_accuracy'],
            deep_results['performance']['val_accuracy']
        ],
        'Log-Likelihood': [
            np.nan,  # Would need to extract from MNL output
            linear_results['performance']['log_likelihood'],
            deep_results['performance']['log_likelihood']
        ]
    })
    
    comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    print("\n✓ Model Comparison Table:")
    print(comparison.to_string(index=False))
    
    return comparison


def create_wtp_comparison(mnl_results, linear_results, deep_results, output_dir):
    """Create comprehensive WTP comparison across all models."""
    print("\n" + "="*80)
    print("STEP 4: Creating WTP Comparison")
    print("="*80)
    
    # Prepare MNL WTP data
    mnl_wtp = mnl_results['wtp'].copy()
    mnl_wtp['model'] = 'MNL'
    mnl_wtp = mnl_wtp[['attribute', 'level', 'wtp_vs_baseline', 'model']]
    mnl_wtp = mnl_wtp.rename(columns={'wtp_vs_baseline': 'wtp'})
    
    # Neural network WTP data needs to be processed differently
    # They output feature-level WTP, not level-level
    # We'll create a separate comparison for that
    
    # Save MNL level-wise WTP
    mnl_wtp.to_csv(output_dir / 'wtp_comparison_by_level.csv', index=False)
    
    # Create feature-level comparison for NN models
    linear_wtp_df = linear_results['wtp'].copy()
    linear_wtp_df['model'] = 'Linear NN'
    
    deep_wtp_df = deep_results['wtp'].copy()
    deep_wtp_df['model'] = 'Deep NN'
    
    # Combine NN results
    nn_comparison = pd.concat([linear_wtp_df, deep_wtp_df], ignore_index=True)
    nn_comparison.to_csv(output_dir / 'wtp_comparison_nn_features.csv', index=False)
    
    print("\n✓ WTP Comparison (MNL attribute levels):")
    print(mnl_wtp.head(15).to_string(index=False))
    
    print("\n✓ WTP Comparison (NN features):")
    print(nn_comparison.head(10).to_string(index=False))
    
    return mnl_wtp, nn_comparison


def create_visualizations(mnl_wtp, nn_comparison, comparison_df, output_dir):
    """Create visualization plots."""
    print("\n" + "="*80)
    print("STEP 5: Creating Visualizations")
    print("="*80)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Plot 1: MNL WTP by Attribute
    fig, ax = plt.subplots(figsize=(14, 8))
    
    attributes = mnl_wtp['attribute'].unique()
    colors = sns.color_palette("husl", len(attributes))
    
    for i, attr in enumerate(attributes):
        attr_data = mnl_wtp[mnl_wtp['attribute'] == attr].copy()
        attr_data = attr_data.sort_values('wtp')
        ax.barh(range(len(attr_data)), attr_data['wtp'], 
                label=attr, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('WTP (EUR)', fontsize=12)
    ax.set_title('MNL Model: Willingness to Pay by Attribute Level', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'mnl_wtp_by_attribute.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: mnl_wtp_by_attribute.png")
    
    # Plot 2: WTP by Attribute (grouped bar chart)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for i, attr in enumerate(attributes):
        attr_data = mnl_wtp[mnl_wtp['attribute'] == attr].copy()
        attr_data = attr_data.sort_values('level')
        
        x_positions = np.arange(len(attr_data)) + i * (len(attr_data) + 0.5)
        ax.bar(x_positions, attr_data['wtp'], width=0.4, 
               label=attr, color=colors[i], alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(attr_data['level'], rotation=45, ha='right')
    
    ax.set_ylabel('WTP (EUR)', fontsize=12)
    ax.set_title('MNL Model: WTP by Level within Each Attribute', fontsize=14, fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'mnl_wtp_grouped.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: mnl_wtp_grouped.png")
    
    # Plot 3: Neural Network Feature WTP Comparison
    if len(nn_comparison) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Pivot for side-by-side comparison
        nn_pivot = nn_comparison.pivot(index='feature', columns='model', values='wtp')
        
        x = np.arange(len(nn_pivot))
        width = 0.35
        
        ax.barh(x - width/2, nn_pivot['Linear NN'], width, label='Linear NN', alpha=0.8)
        ax.barh(x + width/2, nn_pivot['Deep NN'], width, label='Deep NN', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(nn_pivot.index, fontsize=9)
        ax.set_xlabel('WTP (EUR)', fontsize=12)
        ax.set_title('Neural Network Models: Feature-level WTP Comparison', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'nn_wtp_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: nn_wtp_comparison.png")
    
    # Plot 4: Model Performance Comparison
    if 'Validation Accuracy (%)' in comparison_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy comparison (only for NN models)
        nn_models = comparison_df[comparison_df['Model'].str.contains('NN')]
        if len(nn_models) > 0 and nn_models['Validation Accuracy (%)'].notna().any():
            ax1.bar(nn_models['Model'], nn_models['Validation Accuracy (%)'], 
                   color=['steelblue', 'coral'], alpha=0.7)
            ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
            ax1.set_title('Model Validation Accuracy', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, 100)
            ax1.grid(axis='y', alpha=0.3)
            
            for i, (idx, row) in enumerate(nn_models.iterrows()):
                ax1.text(i, row['Validation Accuracy (%)'] + 2, 
                        f"{row['Validation Accuracy (%)']:.2f}%",
                        ha='center', va='bottom', fontweight='bold')
        
        # Log-likelihood comparison
        models_with_ll = comparison_df[comparison_df['Log-Likelihood'].notna()]
        if len(models_with_ll) > 0:
            ax2.bar(models_with_ll['Model'], models_with_ll['Log-Likelihood'],
                   color=['steelblue', 'coral'], alpha=0.7)
            ax2.set_ylabel('Log-Likelihood', fontsize=12)
            ax2.set_title('Model Log-Likelihood', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            for i, (idx, row) in enumerate(models_with_ll.iterrows()):
                ax2.text(i, row['Log-Likelihood'] + abs(row['Log-Likelihood'])*0.02,
                        f"{row['Log-Likelihood']:.1f}",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: model_performance.png")
    
    print(f"\n✓ All visualizations saved to {vis_dir}/")


def generate_summary_report(mnl_results, linear_results, deep_results,
                            comparison_df, mnl_wtp, nn_comparison, output_dir):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("STEP 6: Generating Summary Report")
    print("="*80)
    
    report_path = output_dir / 'COMPREHENSIVE_ANALYSIS_REPORT.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE DISCRETE CHOICE EXPERIMENT ANALYSIS\n")
        f.write("MNL vs Neural Network Models Comparison\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. MODEL OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write("Three models were estimated and compared:\n\n")
        f.write("  a) Multinomial Logit (MNL) - Traditional statistical approach\n")
        f.write("     - Estimated using pylogit package\n")
        f.write("     - Assumes linear utility and IIA property\n")
        f.write("     - Effects-coded categorical variables\n\n")
        f.write("  b) Linear Neural Network - Machine learning equivalent to MNL\n")
        f.write("     - Single linear layer (no hidden layers)\n")
        f.write("     - Should produce similar results to MNL\n")
        f.write("     - Trained with 80/20 train-validation split\n\n")
        f.write("  c) Deep Neural Network - Non-linear machine learning\n")
        f.write("     - 2 hidden layers (64 and 32 units)\n")
        f.write("     - ReLU activation with 20% dropout\n")
        f.write("     - Captures non-linear preference patterns\n\n")
        
        f.write("\n2. MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("\n3. MNL MODEL COEFFICIENTS\n")
        f.write("-" * 80 + "\n")
        f.write(mnl_results['coefficients'].to_string(index=False))
        f.write("\n\n")
        
        f.write("\n4. WILLINGNESS TO PAY (WTP) - MNL Model\n")
        f.write("-" * 80 + "\n")
        
        for attr in mnl_wtp['attribute'].unique():
            f.write(f"\n{attr}:\n")
            attr_wtp = mnl_wtp[mnl_wtp['attribute'] == attr]
            attr_wtp_display = attr_wtp[['level', 'wtp']].copy()
            attr_wtp_display['wtp'] = attr_wtp_display['wtp'].apply(lambda x: f"€{x:.2f}")
            f.write(attr_wtp_display.to_string(index=False))
            f.write("\n")
        
        f.write("\n\n5. NEURAL NETWORK WTP COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(nn_comparison.to_string(index=False))
        f.write("\n\n")
        
        f.write("\n6. KEY INSIGHTS\n")
        f.write("-" * 80 + "\n")
        
        # Compare Linear NN vs Deep NN accuracy
        if 'Validation Accuracy (%)' in comparison_df.columns:
            linear_acc = comparison_df[comparison_df['Model'] == 'Linear NN']['Validation Accuracy (%)'].values
            deep_acc = comparison_df[comparison_df['Model'] == 'Deep NN']['Validation Accuracy (%)'].values
            
            if len(linear_acc) > 0 and len(deep_acc) > 0:
                linear_acc = linear_acc[0]
                deep_acc = deep_acc[0]
                
                f.write(f"\n• Validation Accuracy:\n")
                f.write(f"  - Linear NN: {linear_acc:.2f}%\n")
                f.write(f"  - Deep NN: {deep_acc:.2f}%\n")
                
                if deep_acc > linear_acc:
                    f.write(f"  → Deep NN performs {deep_acc - linear_acc:.2f}% better\n")
                    f.write(f"  → This suggests NON-LINEAR preference patterns exist\n")
                else:
                    f.write(f"  → Models perform similarly\n")
                    f.write(f"  → Preferences appear primarily LINEAR\n")
        
        # WTP insights
        f.write(f"\n• MNL Model WTP Range:\n")
        for attr in mnl_wtp['attribute'].unique():
            attr_wtp = mnl_wtp[mnl_wtp['attribute'] == attr]['wtp']
            wtp_range = attr_wtp.max() - attr_wtp.min()
            f.write(f"  - {attr}: €{wtp_range:.2f}\n")
        
        f.write("\n\n7. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("• If Deep NN significantly outperforms Linear NN:\n")
        f.write("  → Consider mixed logit or latent class models\n")
        f.write("  → Preference heterogeneity may be important\n\n")
        f.write("• If models perform similarly:\n")
        f.write("  → MNL assumptions appear valid\n")
        f.write("  → Standard MNL interpretation is reliable\n\n")
        f.write("• For publication:\n")
        f.write("  → Report MNL as main model (interpretability)\n")
        f.write("  → Use NN results as robustness check\n")
        f.write("  → Discuss any divergence in WTP estimates\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Comprehensive report saved to: {report_path}")
    
    # Also print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n📊 Models compared: 3 (MNL, Linear NN, Deep NN)")
    print(f"📁 Output directory: {output_dir}")
    print(f"📈 Visualizations created: 4 plots")
    print(f"📝 Full report: {report_path.name}")


def main():
    """Main execution function."""
    print("\n" + "🚀 " + "="*76)
    print("🚀 COMPREHENSIVE DCE ANALYSIS: MNL vs NEURAL NETWORKS")
    print("🚀 " + "="*76)
    
    # Setup paths
    data_path = 'Data/dce_long_model.csv'
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Run MNL
    mnl_results = run_mnl_model(data_path, output_dir)
    if mnl_results is None:
        print("\n❌ MNL model failed. Stopping analysis.")
        return
    
    # Step 2: Run Neural Networks
    linear_results, deep_results = run_neural_networks(data_path, output_dir, epochs=100)
    if linear_results is None or deep_results is None:
        print("\n❌ Neural network models failed. Stopping analysis.")
        return
    
    # Step 3: Create performance comparison
    comparison_df = create_performance_comparison(mnl_results, linear_results, 
                                                   deep_results, output_dir)
    
    # Step 4: Create WTP comparison
    mnl_wtp, nn_comparison = create_wtp_comparison(mnl_results, linear_results,
                                                    deep_results, output_dir)
    
    # Step 5: Create visualizations
    create_visualizations(mnl_wtp, nn_comparison, comparison_df, output_dir)
    
    # Step 6: Generate summary report
    generate_summary_report(mnl_results, linear_results, deep_results,
                           comparison_df, mnl_wtp, nn_comparison, output_dir)
    
    print("\n" + "✅ " + "="*76)
    print("✅ ANALYSIS COMPLETE!")
    print("✅ " + "="*76)
    print(f"\n📂 All results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
