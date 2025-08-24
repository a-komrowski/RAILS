#!/usr/bin/env python3
"""
Visual Report Generator for Clustering Experiments
Analyzes clustering results with focus on individual metrics rather than composite scores.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusteringVisualReport:
    """Generates comprehensive visual reports for clustering experiments."""
    
    def __init__(self, results_file):
        """Load and prepare the results data."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Prepare data for analysis
        self.df = self._prepare_dataframe()
        self.metrics_info = {
            'silhouette': {
                'name': 'Silhouette Score',
                'description': 'Higher is better (max: 1.0)',
                'direction': 'maximize',
                'range': (-1, 1),
                'optimal_range': (0.5, 1.0),
                'good_range': (0.3, 0.5),
                'poor_range': (-1, 0.3)
            },
            'davies_bouldin': {
                'name': 'Davies-Bouldin Index',
                'description': 'Lower is better (min: 0.0)',
                'direction': 'minimize',
                'range': (0, np.inf),
                'optimal_range': (0.0, 1.0),
                'good_range': (1.0, 1.5),
                'poor_range': (1.5, np.inf)
            },
            'calinski_harabasz': {
                'name': 'Calinski-Harabasz Index',
                'description': 'Higher is better',
                'direction': 'maximize',
                'range': (0, np.inf),
                'optimal_range': None,  # Relative metric
                'good_range': None,
                'poor_range': None
            }
        }
    
    def _prepare_dataframe(self):
        """Convert results to a pandas DataFrame for easier analysis."""
        data = []
        for result in self.results:
            base_info = {
                'model': result['model'],
                'level': result['level'],
                'preprocessing': result['preprocessing'],
                'feature_dim': result['feature_dim'],
                'best_k': result['best_k'],
                'silhouette': result['silhouette'],
                'davies_bouldin': result['davies_bouldin'],
                'calinski_harabasz': result['calinski_harabasz'],
                'composite_score': result['composite_score'],
                'pca_explained_variance': result['pca_explained_variance']
            }
            
            # Create experiment identifier
            base_info['experiment'] = f"{result['model']}_{result['level']}_{result['preprocessing']}"
            data.append(base_info)
        
        return pd.DataFrame(data)
    
    def create_comprehensive_report(self, output_dir="./clustering_report"):
        """Generate the complete visual report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating clustering analysis report...")
        print(f"Total experiments analyzed: {len(self.df)}")
        
        # Create all visualizations
        self._create_metrics_overview(output_path)
        self._create_metrics_comparison(output_path)
        self._create_model_comparison(output_path)
        self._create_preprocessing_analysis(output_path)
        self._create_level_analysis(output_path)
        self._create_top_performers(output_path)
        self._create_detailed_heatmaps(output_path)
        self._create_metric_distributions(output_path)
        
        # Generate summary report
        self._generate_summary_report(output_path)
        
        print(f"Report generated successfully in: {output_path}")
        return output_path
    
    def _create_metrics_overview(self, output_path):
        """Create overview of all three metrics across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Clustering Metrics Overview - Individual Analysis', fontsize=20, fontweight='bold')
        
        # Silhouette Score
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(self.df)), self.df['silhouette'], 
                       color=['green' if x >= 0.5 else 'orange' if x >= 0.3 else 'red' 
                             for x in self.df['silhouette']], alpha=0.7)
        ax1.set_title('Silhouette Score\n(Higher = Better Separated Clusters)', fontweight='bold')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_xlabel('Experiments')
        ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.5)')
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.3)')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Davies-Bouldin Index
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(self.df)), self.df['davies_bouldin'], 
                       color=['green' if x <= 1.0 else 'orange' if x <= 1.5 else 'red' 
                             for x in self.df['davies_bouldin']], alpha=0.7)
        ax2.set_title('Davies-Bouldin Index\n(Lower = Better Cluster Separation)', fontweight='bold')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_xlabel('Experiments')
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Excellent (≤1.0)')
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Good (≤1.5)')
        ax2.legend()
        
        # Calinski-Harabasz Index (normalized for better visualization)
        ax3 = axes[1, 0]
        ch_normalized = (self.df['calinski_harabasz'] - self.df['calinski_harabasz'].min()) / \
                       (self.df['calinski_harabasz'].max() - self.df['calinski_harabasz'].min())
        bars3 = ax3.bar(range(len(self.df)), ch_normalized, 
                       color=plt.cm.viridis(ch_normalized), alpha=0.7)
        ax3.set_title('Calinski-Harabasz Index (Normalized)\n(Higher = Better Defined Clusters)', fontweight='bold')
        ax3.set_ylabel('Normalized Score')
        ax3.set_xlabel('Experiments')
        ax3.set_ylim(0, 1)
        
        # Combined ranking visualization
        ax4 = axes[1, 1]
        # Calculate rankings for each metric (considering direction)
        sil_rank = self.df['silhouette'].rank(ascending=False)
        db_rank = self.df['davies_bouldin'].rank(ascending=True)  # Lower is better
        ch_rank = self.df['calinski_harabasz'].rank(ascending=False)
        
        # Average rank (lower is better overall)
        avg_rank = (sil_rank + db_rank + ch_rank) / 3
        
        colors = plt.cm.RdYlGn_r(avg_rank / len(self.df))
        bars4 = ax4.bar(range(len(self.df)), avg_rank, color=colors, alpha=0.7)
        ax4.set_title('Overall Performance Ranking\n(Lower = Better Overall Performance)', fontweight='bold')
        ax4.set_ylabel('Average Rank')
        ax4.set_xlabel('Experiments')
        
        # Add experiment labels
        for ax in axes.flat:
            ax.set_xticks(range(len(self.df)))
            ax.set_xticklabels([f"{row['model'][:3]}\n{row['level'][:3]}\n{row['preprocessing'][:4]}" 
                               for _, row in self.df.iterrows()], 
                              rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_comparison(self, output_path):
        """Create detailed comparison of metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Clustering Metrics: Detailed Comparison', fontsize=16, fontweight='bold')
        
        # Silhouette vs Davies-Bouldin
        ax1 = axes[0]
        scatter = ax1.scatter(self.df['silhouette'], self.df['davies_bouldin'], 
                            c=self.df['calinski_harabasz'], s=100, alpha=0.7, 
                            cmap='viridis')
        ax1.set_xlabel('Silhouette Score (Higher is Better)')
        ax1.set_ylabel('Davies-Bouldin Index (Lower is Better)')
        ax1.set_title('Silhouette vs Davies-Bouldin\nColor = Calinski-Harabasz')
        
        # Add quadrant lines
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=0.4, color='red', linestyle='--', alpha=0.5)
        
        # Add annotations for best performers
        best_idx = ((self.df['silhouette'] > 0.4) & (self.df['davies_bouldin'] < 1.0))
        for idx, row in self.df[best_idx].iterrows():
            ax1.annotate(f"{row['model'][:3]}_{row['level'][:1]}", 
                        (row['silhouette'], row['davies_bouldin']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax1, label='Calinski-Harabasz')
        
        # Metrics correlation heatmap
        ax2 = axes[1]
        corr_data = self.df[['silhouette', 'davies_bouldin', 'calinski_harabasz']].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, ax=ax2,
                   square=True, fmt='.3f')
        ax2.set_title('Metrics Correlation')
        
        # Performance distribution by model
        ax3 = axes[2]
        model_performance = []
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            # Normalize each metric to 0-1 scale for fair comparison
            sil_norm = model_data['silhouette'].mean()
            db_norm = 1 / (1 + model_data['davies_bouldin'].mean())  # Invert since lower is better
            ch_norm = (model_data['calinski_harabasz'] - self.df['calinski_harabasz'].min()) / \
                     (self.df['calinski_harabasz'].max() - self.df['calinski_harabasz'].min())
            ch_norm = ch_norm.mean()
            
            model_performance.append({
                'Model': model,
                'Silhouette': sil_norm,
                'Davies-Bouldin (inv)': db_norm,
                'Calinski-Harabasz (norm)': ch_norm
            })
        
        perf_df = pd.DataFrame(model_performance)
        perf_df.set_index('Model').plot(kind='bar', ax=ax3, alpha=0.8)
        ax3.set_title('Average Performance by Model\n(All metrics normalized 0-1)')
        ax3.set_ylabel('Normalized Score')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_comparison(self, output_path):
        """Compare performance across different models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Silhouette by model
        ax1 = axes[0, 0]
        sns.boxplot(data=self.df, x='model', y='silhouette', ax=ax1)
        ax1.set_title('Silhouette Score by Model')
        ax1.set_ylabel('Silhouette Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin by model
        ax2 = axes[0, 1]
        sns.boxplot(data=self.df, x='model', y='davies_bouldin', ax=ax2)
        ax2.set_title('Davies-Bouldin Index by Model')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz by model
        ax3 = axes[1, 0]
        sns.boxplot(data=self.df, x='model', y='calinski_harabasz', ax=ax3)
        ax3.set_title('Calinski-Harabasz Index by Model')
        ax3.set_ylabel('Calinski-Harabasz Index')
        ax3.tick_params(axis='x', rotation=45)
        
        # Model summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            summary_stats.append({
                'Model': model,
                'Experiments': len(model_data),
                'Avg Silhouette': f"{model_data['silhouette'].mean():.3f}",
                'Avg Davies-B': f"{model_data['davies_bouldin'].mean():.3f}",
                'Avg Calinski-H': f"{model_data['calinski_harabasz'].mean():.0f}",
                'Best Sil': f"{model_data['silhouette'].max():.3f}",
                'Best DB': f"{model_data['davies_bouldin'].min():.3f}"
            })
        
        summary_df = pd.DataFrame(summary_stats)
        table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Model Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_preprocessing_analysis(self, output_path):
        """Analyze the effect of preprocessing methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Preprocessing Method Analysis', fontsize=16, fontweight='bold')
        
        # Group by preprocessing
        prep_groups = self.df.groupby('preprocessing')
        
        # Silhouette by preprocessing
        ax1 = axes[0, 0]
        sns.boxplot(data=self.df, x='preprocessing', y='silhouette', ax=ax1)
        ax1.set_title('Silhouette Score by Preprocessing')
        ax1.set_ylabel('Silhouette Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin by preprocessing
        ax2 = axes[0, 1]
        sns.boxplot(data=self.df, x='preprocessing', y='davies_bouldin', ax=ax2)
        ax2.set_title('Davies-Bouldin Index by Preprocessing')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.tick_params(axis='x', rotation=45)
        
        # Effect of preprocessing on different models
        ax3 = axes[1, 0]
        pivot_sil = self.df.pivot_table(values='silhouette', index='model', 
                                       columns='preprocessing', aggfunc='mean')
        sns.heatmap(pivot_sil, annot=True, cmap='RdYlGn', ax=ax3, fmt='.3f')
        ax3.set_title('Silhouette: Model vs Preprocessing')
        
        # Preprocessing effectiveness ranking
        ax4 = axes[1, 1]
        prep_effectiveness = []
        for prep in self.df['preprocessing'].unique():
            prep_data = self.df[self.df['preprocessing'] == prep]
            # Calculate average normalized performance
            sil_rank = prep_data['silhouette'].mean()
            db_rank = 1 / (1 + prep_data['davies_bouldin'].mean())
            ch_rank = (prep_data['calinski_harabasz'] - self.df['calinski_harabasz'].min()) / \
                     (self.df['calinski_harabasz'].max() - self.df['calinski_harabasz'].min())
            ch_rank = ch_rank.mean()
            
            avg_performance = (sil_rank + db_rank + ch_rank) / 3
            prep_effectiveness.append({
                'Preprocessing': prep,
                'Performance': avg_performance,
                'Count': len(prep_data)
            })
        
        prep_df = pd.DataFrame(prep_effectiveness).sort_values('Performance', ascending=False)
        bars = ax4.bar(prep_df['Preprocessing'], prep_df['Performance'], 
                      color=plt.cm.viridis(prep_df['Performance'] / prep_df['Performance'].max()))
        ax4.set_title('Preprocessing Method Effectiveness')
        ax4.set_ylabel('Average Normalized Performance')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, prep_df['Count']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'preprocessing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_level_analysis(self, output_path):
        """Analyze performance across different feature extraction levels."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Extraction Level Analysis', fontsize=16, fontweight='bold')
        
        # Performance by level
        levels = ['top', 'mid', 'low']
        
        for i, metric in enumerate(['silhouette', 'davies_bouldin', 'calinski_harabasz']):
            if i < 3:
                ax = axes[i//2, i%2]
                sns.boxplot(data=self.df, x='level', y=metric, order=levels, ax=ax)
                ax.set_title(f'{self.metrics_info[metric]["name"]} by Level')
                ax.set_ylabel(self.metrics_info[metric]["name"])
        
        # Combined level effectiveness
        ax4 = axes[1, 1]
        level_effectiveness = []
        for level in levels:
            level_data = self.df[self.df['level'] == level]
            if len(level_data) > 0:
                sil_avg = level_data['silhouette'].mean()
                db_avg = 1 / (1 + level_data['davies_bouldin'].mean())
                ch_norm = (level_data['calinski_harabasz'] - self.df['calinski_harabasz'].min()) / \
                         (self.df['calinski_harabasz'].max() - self.df['calinski_harabasz'].min())
                ch_avg = ch_norm.mean()
                
                level_effectiveness.append({
                    'Level': level,
                    'Silhouette': sil_avg,
                    'Davies-B (inv)': db_avg,
                    'Calinski-H (norm)': ch_avg,
                    'Count': len(level_data)
                })
        
        level_df = pd.DataFrame(level_effectiveness)
        if not level_df.empty:
            level_df.set_index('Level')[['Silhouette', 'Davies-B (inv)', 'Calinski-H (norm)']].plot(
                kind='bar', ax=ax4, alpha=0.8)
            ax4.set_title('Performance by Feature Level')
            ax4.set_ylabel('Normalized Score')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path / 'level_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_top_performers(self, output_path):
        """Identify and visualize top performing experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Performing Experiments', fontsize=16, fontweight='bold')
        
        # Top 5 by each metric
        top_silhouette = self.df.nlargest(5, 'silhouette')
        top_davies_bouldin = self.df.nsmallest(5, 'davies_bouldin')
        top_calinski = self.df.nlargest(5, 'calinski_harabasz')
        
        # Top silhouette performers
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(top_silhouette)), top_silhouette['silhouette'], 
                       color='lightgreen', alpha=0.8)
        ax1.set_title('Top 5: Silhouette Score')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_xticks(range(len(top_silhouette)))
        ax1.set_xticklabels([f"{row['experiment'][:15]}..." for _, row in top_silhouette.iterrows()], 
                           rotation=45, fontsize=8)
        
        # Add value labels
        for bar, val in zip(bars1, top_silhouette['silhouette']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Top Davies-Bouldin performers
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(top_davies_bouldin)), top_davies_bouldin['davies_bouldin'], 
                       color='lightcoral', alpha=0.8)
        ax2.set_title('Top 5: Davies-Bouldin Index (Lower = Better)')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_xticks(range(len(top_davies_bouldin)))
        ax2.set_xticklabels([f"{row['experiment'][:15]}..." for _, row in top_davies_bouldin.iterrows()], 
                           rotation=45, fontsize=8)
        
        for bar, val in zip(bars2, top_davies_bouldin['davies_bouldin']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Overall best performers (balanced across metrics)
        ax3 = axes[1, 0]
        # Calculate balanced score
        self.df['balanced_score'] = (
            self.df['silhouette'] / self.df['silhouette'].max() +
            (1 - self.df['davies_bouldin'] / self.df['davies_bouldin'].max()) +
            self.df['calinski_harabasz'] / self.df['calinski_harabasz'].max()
        ) / 3
        
        top_balanced = self.df.nlargest(8, 'balanced_score')
        bars3 = ax3.bar(range(len(top_balanced)), top_balanced['balanced_score'], 
                       color=plt.cm.viridis(top_balanced['balanced_score'] / top_balanced['balanced_score'].max()),
                       alpha=0.8)
        ax3.set_title('Top 8: Balanced Performance')
        ax3.set_ylabel('Balanced Score (0-1)')
        ax3.set_xticks(range(len(top_balanced)))
        ax3.set_xticklabels([f"{row['experiment'][:12]}..." for _, row in top_balanced.iterrows()], 
                           rotation=45, fontsize=8)
        
        # Performance comparison table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Show top 5 balanced performers with all metrics
        top_5_balanced = top_balanced.head(5)
        comparison_data = []
        for _, row in top_5_balanced.iterrows():
            comparison_data.append({
                'Experiment': row['experiment'][:20] + '...' if len(row['experiment']) > 20 else row['experiment'],
                'Silhouette': f"{row['silhouette']:.3f}",
                'Davies-B': f"{row['davies_bouldin']:.3f}",
                'Calinski-H': f"{row['calinski_harabasz']:.0f}",
                'Balanced': f"{row['balanced_score']:.3f}"
            })
        
        comp_df = pd.DataFrame(comparison_data)
        table = ax4.table(cellText=comp_df.values, colLabels=comp_df.columns,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)
        ax4.set_title('Top 5 Balanced Performers - Detailed Metrics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_performers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_heatmaps(self, output_path):
        """Create detailed heatmaps for pattern analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Detailed Performance Heatmaps', fontsize=16, fontweight='bold')
        
        # Model vs Level heatmap for each metric
        for i, metric in enumerate(['silhouette', 'davies_bouldin']):
            ax = axes[i//2, i*2%(len(axes[0]))]
            pivot = self.df.pivot_table(values=metric, index='model', 
                                      columns='level', aggfunc='mean')
            
            cmap = 'RdYlGn' if metric == 'silhouette' else 'RdYlGn_r'
            sns.heatmap(pivot, annot=True, cmap=cmap, ax=ax, fmt='.3f',
                       cbar_kws={'label': self.metrics_info[metric]['name']})
            ax.set_title(f'{self.metrics_info[metric]["name"]}: Model vs Level')
        
        # Model vs Preprocessing heatmap
        ax3 = axes[0, 1]
        pivot_prep = self.df.pivot_table(values='silhouette', index='model', 
                                        columns='preprocessing', aggfunc='mean')
        sns.heatmap(pivot_prep, annot=True, cmap='RdYlGn', ax=ax3, fmt='.3f')
        ax3.set_title('Silhouette Score: Model vs Preprocessing')
        
        # Comprehensive experiment ranking
        ax4 = axes[1, 1]
        
        # Create ranking matrix
        ranking_data = []
        for _, row in self.df.iterrows():
            ranking_data.append({
                'Experiment': row['experiment'][:25] + '...' if len(row['experiment']) > 25 else row['experiment'],
                'Silhouette_Rank': self.df['silhouette'].rank(ascending=False)[row.name],
                'Davies_B_Rank': self.df['davies_bouldin'].rank(ascending=True)[row.name],
                'Calinski_H_Rank': self.df['calinski_harabasz'].rank(ascending=False)[row.name],
                'Avg_Rank': (self.df['silhouette'].rank(ascending=False)[row.name] + 
                           self.df['davies_bouldin'].rank(ascending=True)[row.name] + 
                           self.df['calinski_harabasz'].rank(ascending=False)[row.name]) / 3
            })
        
        ranking_df = pd.DataFrame(ranking_data).sort_values('Avg_Rank')
        
        # Show top 10 in heatmap format
        top_10_ranking = ranking_df.head(10)
        rank_matrix = top_10_ranking[['Silhouette_Rank', 'Davies_B_Rank', 'Calinski_H_Rank']].values
        
        im = ax4.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto')
        ax4.set_xticks(range(3))
        ax4.set_xticklabels(['Silhouette\nRank', 'Davies-B\nRank', 'Calinski-H\nRank'])
        ax4.set_yticks(range(10))
        ax4.set_yticklabels([exp[:20] + '...' if len(exp) > 20 else exp 
                            for exp in top_10_ranking['Experiment']], fontsize=8)
        ax4.set_title('Top 10 Experiments: Individual Metric Rankings')
        
        # Add ranking numbers to heatmap
        for i in range(10):
            for j in range(3):
                ax4.text(j, i, f'{int(rank_matrix[i, j])}', ha='center', va='center', 
                        color='white' if rank_matrix[i, j] > len(self.df)/2 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Rank (Lower = Better)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'detailed_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_distributions(self, output_path):
        """Analyze the distribution of metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Metrics: Distribution Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        
        for i, metric in enumerate(metrics):
            # Distribution histogram
            ax_hist = axes[0, i]
            self.df[metric].hist(bins=15, alpha=0.7, ax=ax_hist, color=f'C{i}')
            ax_hist.axvline(self.df[metric].mean(), color='red', linestyle='--', 
                           label=f'Mean: {self.df[metric].mean():.3f}')
            ax_hist.axvline(self.df[metric].median(), color='orange', linestyle='--', 
                           label=f'Median: {self.df[metric].median():.3f}')
            ax_hist.set_title(f'{self.metrics_info[metric]["name"]} Distribution')
            ax_hist.set_xlabel(self.metrics_info[metric]["name"])
            ax_hist.set_ylabel('Frequency')
            ax_hist.legend()
            
            # Quality assessment
            info = self.metrics_info[metric]
            if info['optimal_range']:
                if metric == 'silhouette':
                    excellent = (self.df[metric] >= info['optimal_range'][0]).sum()
                    good = ((self.df[metric] >= info['good_range'][0]) & 
                           (self.df[metric] < info['optimal_range'][0])).sum()
                    poor = (self.df[metric] < info['good_range'][0]).sum()
                else:  # davies_bouldin
                    excellent = (self.df[metric] <= info['optimal_range'][1]).sum()
                    good = ((self.df[metric] > info['optimal_range'][1]) & 
                           (self.df[metric] <= info['good_range'][1])).sum()
                    poor = (self.df[metric] > info['good_range'][1]).sum()
                
                # Quality breakdown pie chart
                ax_pie = axes[1, i]
                sizes = [excellent, good, poor]
                labels = ['Excellent', 'Good', 'Poor']
                colors = ['green', 'orange', 'red']
                wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, 
                                                     autopct='%1.1f%%', startangle=90)
                ax_pie.set_title(f'{self.metrics_info[metric]["name"]}\nQuality Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, output_path):
        """Generate a text summary of key findings."""
        summary_file = output_path / 'summary_report.txt'
        
        with open(summary_file, 'w') as f:
            f.write("CLUSTERING EXPERIMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments analyzed: {len(self.df)}\n\n")
            
            # Overall statistics
            f.write("OVERALL PERFORMANCE STATISTICS\n")
            f.write("-" * 30 + "\n")
            for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
                f.write(f"{self.metrics_info[metric]['name']}:\n")
                f.write(f"  Mean: {self.df[metric].mean():.4f}\n")
                f.write(f"  Median: {self.df[metric].median():.4f}\n")
                f.write(f"  Std: {self.df[metric].std():.4f}\n")
                f.write(f"  Min: {self.df[metric].min():.4f}\n")
                f.write(f"  Max: {self.df[metric].max():.4f}\n\n")
            
            # Top performers
            f.write("TOP PERFORMERS BY METRIC\n")
            f.write("-" * 25 + "\n")
            
            # Best silhouette
            best_sil = self.df.loc[self.df['silhouette'].idxmax()]
            f.write(f"Best Silhouette Score: {best_sil['silhouette']:.4f}\n")
            f.write(f"  Experiment: {best_sil['experiment']}\n")
            f.write(f"  Davies-Bouldin: {best_sil['davies_bouldin']:.4f}\n")
            f.write(f"  Calinski-Harabasz: {best_sil['calinski_harabasz']:.1f}\n\n")
            
            # Best davies-bouldin
            best_db = self.df.loc[self.df['davies_bouldin'].idxmin()]
            f.write(f"Best Davies-Bouldin Index: {best_db['davies_bouldin']:.4f}\n")
            f.write(f"  Experiment: {best_db['experiment']}\n")
            f.write(f"  Silhouette: {best_db['silhouette']:.4f}\n")
            f.write(f"  Calinski-Harabasz: {best_db['calinski_harabasz']:.1f}\n\n")
            
            # Best calinski-harabasz
            best_ch = self.df.loc[self.df['calinski_harabasz'].idxmax()]
            f.write(f"Best Calinski-Harabasz Index: {best_ch['calinski_harabasz']:.1f}\n")
            f.write(f"  Experiment: {best_ch['experiment']}\n")
            f.write(f"  Silhouette: {best_ch['silhouette']:.4f}\n")
            f.write(f"  Davies-Bouldin: {best_ch['davies_bouldin']:.4f}\n\n")
            
            # Model performance
            f.write("PERFORMANCE BY MODEL\n")
            f.write("-" * 20 + "\n")
            for model in self.df['model'].unique():
                model_data = self.df[self.df['model'] == model]
                f.write(f"{model.upper()}:\n")
                f.write(f"  Experiments: {len(model_data)}\n")
                f.write(f"  Avg Silhouette: {model_data['silhouette'].mean():.4f}\n")
                f.write(f"  Avg Davies-Bouldin: {model_data['davies_bouldin'].mean():.4f}\n")
                f.write(f"  Avg Calinski-Harabasz: {model_data['calinski_harabasz'].mean():.1f}\n\n")
            
            # Preprocessing impact
            f.write("PREPROCESSING METHOD IMPACT\n")
            f.write("-" * 28 + "\n")
            for prep in self.df['preprocessing'].unique():
                prep_data = self.df[self.df['preprocessing'] == prep]
                f.write(f"{prep.upper()}:\n")
                f.write(f"  Experiments: {len(prep_data)}\n")
                f.write(f"  Avg Silhouette: {prep_data['silhouette'].mean():.4f}\n")
                f.write(f"  Avg Davies-Bouldin: {prep_data['davies_bouldin'].mean():.4f}\n")
                f.write(f"  Avg Calinski-Harabasz: {prep_data['calinski_harabasz'].mean():.1f}\n\n")
            
            # Key recommendations
            f.write("KEY RECOMMENDATIONS\n")
            f.write("-" * 19 + "\n")
            
            # Calculate balanced performance
            self.df['balanced_score'] = (
                self.df['silhouette'] / self.df['silhouette'].max() +
                (1 - self.df['davies_bouldin'] / self.df['davies_bouldin'].max()) +
                self.df['calinski_harabasz'] / self.df['calinski_harabasz'].max()
            ) / 3
            
            best_overall = self.df.loc[self.df['balanced_score'].idxmax()]
            f.write(f"1. Best Overall Performer: {best_overall['experiment']}\n")
            f.write(f"   Balanced Score: {best_overall['balanced_score']:.4f}\n")
            f.write(f"   All metrics: Sil={best_overall['silhouette']:.3f}, ")
            f.write(f"DB={best_overall['davies_bouldin']:.3f}, CH={best_overall['calinski_harabasz']:.0f}\n\n")
            
            # Best model
            model_performance = self.df.groupby('model')['balanced_score'].mean()
            best_model = model_performance.idxmax()
            f.write(f"2. Best Model Overall: {best_model.upper()}\n")
            f.write(f"   Average Balanced Score: {model_performance[best_model]:.4f}\n\n")
            
            # Best preprocessing
            prep_performance = self.df.groupby('preprocessing')['balanced_score'].mean()
            best_prep = prep_performance.idxmax()
            f.write(f"3. Best Preprocessing Method: {best_prep.upper()}\n")
            f.write(f"   Average Balanced Score: {prep_performance[best_prep]:.4f}\n\n")
            
            # Quality assessment
            excellent_sil = (self.df['silhouette'] >= 0.5).sum()
            good_sil = ((self.df['silhouette'] >= 0.3) & (self.df['silhouette'] < 0.5)).sum()
            
            f.write("QUALITY ASSESSMENT\n")
            f.write("-" * 18 + "\n")
            f.write(f"Silhouette Score Quality:\n")
            f.write(f"  Excellent (≥0.5): {excellent_sil} experiments ({excellent_sil/len(self.df)*100:.1f}%)\n")
            f.write(f"  Good (0.3-0.5): {good_sil} experiments ({good_sil/len(self.df)*100:.1f}%)\n")
            f.write(f"  Poor (<0.3): {len(self.df)-excellent_sil-good_sil} experiments ({(len(self.df)-excellent_sil-good_sil)/len(self.df)*100:.1f}%)\n\n")
        
        print(f"Summary report generated: {summary_file}")


def main():
    """Main function to run the report generator."""
    parser = argparse.ArgumentParser(description='Generate visual clustering analysis report')
    parser.add_argument('results_file', help='Path to the JSON results file')
    parser.add_argument('--output', '-o', default='./clustering_report', 
                       help='Output directory for the report (default: ./clustering_report)')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file {args.results_file} not found!")
        return
    
    try:
        # Generate the report
        report_generator = ClusteringVisualReport(args.results_file)
        output_path = report_generator.create_comprehensive_report(args.output)
        
        print(f"\n✅ Report generation completed successfully!")
        print(f"📁 Report location: {output_path}")
        print(f"\nGenerated files:")
        for file in sorted(output_path.glob('*.png')):
            print(f"  📊 {file.name}")
        print(f"  📄 summary_report.txt")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()