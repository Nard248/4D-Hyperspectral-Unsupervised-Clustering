"""
Wavelength Analysis Visualization Module

This module provides comprehensive visualization tools for wavelength analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class WavelengthVisualizer:
    """
    Comprehensive visualization tools for wavelength analysis results
    """
    
    def __init__(self, selected_bands: List[Dict], influence_matrix: Dict, output_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            selected_bands: List of selected wavelength combinations
            influence_matrix: Matrix of influence scores by excitation
            output_dir: Directory to save visualizations
        """
        self.selected_bands = selected_bands
        self.influence_matrix = influence_matrix
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_influence_heatmap(self):
        """Create heatmap of influence scores across all wavelengths"""
        print("  Creating influence heatmap...")
        
        # Prepare data for heatmap
        excitations = sorted(self.influence_matrix.keys())
        max_bands = max(len(self.influence_matrix[ex]) for ex in excitations)
        
        influence_array = np.zeros((len(excitations), max_bands))
        for i, ex in enumerate(excitations):
            influences = self.influence_matrix[ex]
            influence_array[i, :len(influences)] = influences
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        
        # Use log scale for better visualization of wide range
        influence_log = np.log10(influence_array + 1e-10)  # Add small constant to avoid log(0)
        
        heatmap = plt.imshow(influence_log, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Customize plot
        plt.colorbar(heatmap, label='Log10(Influence Score)', shrink=0.8)
        plt.xlabel('Emission Band Index', fontsize=12)
        plt.ylabel('Excitation Wavelength (nm)', fontsize=12)
        plt.title('Wavelength Influence Heatmap (Log Scale)', fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        plt.yticks(range(len(excitations)), [f"{ex:.0f}" for ex in excitations])
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "influence_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_wavelength_scatter(self):
        """Create scatter plot of selected wavelength combinations"""
        print("  Creating wavelength scatter plot...")
        
        # Extract data for plotting
        ex_values = [band['excitation'] for band in self.selected_bands[:50]]  # Limit for readability
        em_values = [band['emission_wavelength'] for band in self.selected_bands[:50]]
        influences = [band['influence'] for band in self.selected_bands[:50]]
        ranks = [band['rank'] for band in self.selected_bands[:50]]
        
        # Create scatter plot
        plt.figure(figsize=(14, 10))
        
        # Create scatter with color mapping
        scatter = plt.scatter(ex_values, em_values, 
                            c=influences, 
                            s=[200 - (rank-1)*3 for rank in ranks],  # Size decreases with rank
                            cmap='plasma', 
                            edgecolors='black', 
                            linewidth=1, 
                            alpha=0.8)
        
        # Customize plot
        plt.colorbar(scatter, label='Influence Score', shrink=0.8)
        plt.xlabel('Excitation Wavelength (nm)', fontsize=12)
        plt.ylabel('Emission Wavelength (nm)', fontsize=12) 
        plt.title('Top Wavelength Combinations\n(Size ∝ Ranking, Color ∝ Influence)', 
                 fontsize=14, fontweight='bold')
        
        # Add rank annotations
        for i, (ex, em, rank) in enumerate(zip(ex_values[:20], em_values[:20], ranks[:20])):
            plt.annotate(f'{rank}', (ex, em), 
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "wavelength_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_excitation_distribution(self):
        """Create bar chart showing distribution across excitation wavelengths"""
        print("  Creating excitation distribution chart...")
        
        # Count selections per excitation
        excitation_counts = {}
        for band in self.selected_bands:
            ex = band['excitation']
            excitation_counts[ex] = excitation_counts.get(ex, 0) + 1
        
        # Sort by wavelength
        excitations = sorted(excitation_counts.keys())
        counts = [excitation_counts[ex] for ex in excitations]
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        
        bars = plt.bar(range(len(excitations)), counts, 
                      color='skyblue', edgecolor='navy', alpha=0.7, linewidth=1.5)
        
        # Customize plot
        plt.xlabel('Excitation Wavelength (nm)', fontsize=12)
        plt.ylabel('Number of Selected Bands', fontsize=12)
        plt.title('Distribution of Selected Bands Across Excitation Wavelengths', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(excitations)), [f"{ex:.0f}" for ex in excitations], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "excitation_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_influence_ranking_plot(self):
        """Create plot showing influence scores by rank"""
        print("  Creating influence ranking plot...")
        
        # Extract ranks and influences
        ranks = [band['rank'] for band in self.selected_bands]
        influences = [band['influence'] for band in self.selected_bands]
        
        plt.figure(figsize=(12, 8))
        
        # Create line plot with markers
        plt.plot(ranks, influences, 'o-', linewidth=2, markersize=6, 
                markerfacecolor='red', markeredgecolor='darkred', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Band Rank', fontsize=12)
        plt.ylabel('Influence Score', fontsize=12)
        plt.title('Influence Score vs. Band Ranking', fontsize=14, fontweight='bold')
        
        # Use log scale for y-axis if range is large
        if max(influences) / min(influences) > 100:
            plt.yscale('log')
            plt.ylabel('Influence Score (Log Scale)', fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "influence_ranking.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_wavelength_coverage_plot(self):
        """Create 2D coverage plot showing selected vs available wavelength space"""
        print("  Creating wavelength coverage plot...")
        
        plt.figure(figsize=(14, 10))
        
        # Plot all possible combinations as light background
        all_ex = sorted(self.influence_matrix.keys())
        for ex in all_ex:
            n_em_bands = len(self.influence_matrix[ex])
            em_indices = range(n_em_bands)
            # Plot as light gray points
            plt.scatter([ex] * len(em_indices), em_indices, 
                       c='lightgray', s=20, alpha=0.3, marker='s')
        
        # Plot selected combinations
        ex_selected = [band['excitation'] for band in self.selected_bands]
        em_selected = [band['emission_idx'] for band in self.selected_bands]
        influences = [band['influence'] for band in self.selected_bands]
        
        scatter = plt.scatter(ex_selected, em_selected, 
                            c=influences, s=100, cmap='viridis', 
                            edgecolors='black', linewidth=1)
        
        plt.colorbar(scatter, label='Influence Score')
        plt.xlabel('Excitation Wavelength (nm)', fontsize=12)
        plt.ylabel('Emission Band Index', fontsize=12)
        plt.title('Wavelength Space Coverage\n(Gray: Available, Colored: Selected)', 
                 fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "wavelength_coverage.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        print("  Creating summary dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top 10 influence scores (bar chart)
        ax1 = plt.subplot(2, 3, 1)
        top_10 = self.selected_bands[:10]
        labels = [f"Ex{band['excitation']:.0f}/Em{band['emission_wavelength']:.0f}" 
                 for band in top_10]
        influences = [band['influence'] for band in top_10]
        
        bars = ax1.bar(range(len(labels)), influences, color='coral', alpha=0.7)
        ax1.set_xlabel('Wavelength Combinations')
        ax1.set_ylabel('Influence Score')
        ax1.set_title('Top 10 Influence Scores')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Excitation distribution (pie chart)
        ax2 = plt.subplot(2, 3, 2)
        excitation_counts = {}
        for band in self.selected_bands:
            ex = band['excitation']
            excitation_counts[ex] = excitation_counts.get(ex, 0) + 1
        
        labels_pie = [f"{ex:.0f}nm" for ex in sorted(excitation_counts.keys())]
        sizes = [excitation_counts[ex] for ex in sorted(excitation_counts.keys())]
        
        ax2.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution Across Excitations')
        
        # 3. Influence vs Rank (line plot)
        ax3 = plt.subplot(2, 3, 3)
        ranks = [band['rank'] for band in self.selected_bands]
        influences_all = [band['influence'] for band in self.selected_bands]
        
        ax3.semilogy(ranks, influences_all, 'o-', linewidth=2, markersize=4)
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Influence Score (Log)')
        ax3.set_title('Influence Decay by Rank')
        ax3.grid(True, alpha=0.3)
        
        # 4. Wavelength scatter (simplified)
        ax4 = plt.subplot(2, 3, 4)
        ex_vals = [band['excitation'] for band in self.selected_bands[:30]]
        em_vals = [band['emission_wavelength'] for band in self.selected_bands[:30]]
        sizes = [50 - band['rank'] for band in self.selected_bands[:30]]
        
        ax4.scatter(ex_vals, em_vals, s=sizes, alpha=0.6, c='purple')
        ax4.set_xlabel('Excitation (nm)')
        ax4.set_ylabel('Emission (nm)')
        ax4.set_title('Top 30 Wavelength Pairs')
        ax4.grid(True, alpha=0.3)
        
        # 5. Statistics summary (text)
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        # Calculate statistics
        total_selected = len(self.selected_bands)
        max_influence = max(band['influence'] for band in self.selected_bands)
        min_influence = min(band['influence'] for band in self.selected_bands)
        unique_excitations = len(set(band['excitation'] for band in self.selected_bands))
        
        stats_text = f"""
ANALYSIS SUMMARY
{'='*20}
Total Bands Selected: {total_selected}
Unique Excitations: {unique_excitations}
Max Influence: {max_influence:.2e}
Min Influence: {min_influence:.2e}
Influence Range: {max_influence/min_influence:.1f}x

Top 3 Combinations:
1. Ex{self.selected_bands[0]['excitation']:.0f}/Em{self.selected_bands[0]['emission_wavelength']:.0f}: {self.selected_bands[0]['influence']:.3f}
2. Ex{self.selected_bands[1]['excitation']:.0f}/Em{self.selected_bands[1]['emission_wavelength']:.0f}: {self.selected_bands[1]['influence']:.3f}
3. Ex{self.selected_bands[2]['excitation']:.0f}/Em{self.selected_bands[2]['emission_wavelength']:.0f}: {self.selected_bands[2]['influence']:.3f}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 6. Heatmap (mini version)
        ax6 = plt.subplot(2, 3, 6)
        
        # Create mini heatmap of top excitations
        top_excitations = sorted(set(band['excitation'] for band in self.selected_bands[:20]))[:8]
        heatmap_data = []
        
        for ex in top_excitations:
            ex_influences = [band['influence'] for band in self.selected_bands if band['excitation'] == ex][:5]
            while len(ex_influences) < 5:
                ex_influences.append(0)
            heatmap_data.append(ex_influences)
        
        if heatmap_data:
            im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax6.set_xlabel('Top 5 Bands per Excitation')
            ax6.set_ylabel('Excitation (nm)')
            ax6.set_title('Influence Mini-Heatmap')
            ax6.set_yticks(range(len(top_excitations)))
            ax6.set_yticklabels([f"{ex:.0f}" for ex in top_excitations])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_all_visualizations(self):
        """Create all available visualizations"""
        print("Creating comprehensive visualizations...")
        
        self.create_influence_heatmap()
        self.create_wavelength_scatter()
        self.create_excitation_distribution()
        self.create_influence_ranking_plot()
        self.create_wavelength_coverage_plot()
        self.create_summary_dashboard()
        
        print("All visualizations completed!")