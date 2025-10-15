"""
Performance Timing Tracker
==========================
Track and visualize timing data for autoencoder training, wavelength selection, and clustering.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from contextlib import contextmanager


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        return False

    def get_elapsed(self):
        """Get elapsed time in seconds"""
        return self.elapsed


class TimingTracker:
    """Track timing data for the wavelength analysis pipeline"""

    def __init__(self):
        self.timings = {
            'autoencoder_training': [],
            'wavelength_selection': [],
            'clustering_full': [],
            'clustering_subset': [],
            'total_pipeline': []
        }
        self.metadata = []

    def record_autoencoder_training(self, time_seconds, n_features, config_name=""):
        """Record autoencoder training time"""
        self.timings['autoencoder_training'].append({
            'time': time_seconds,
            'n_features': n_features,
            'config': config_name,
            'stage': 'Training'
        })

    def record_wavelength_selection(self, time_seconds, n_selected, config_name=""):
        """Record wavelength selection time"""
        self.timings['wavelength_selection'].append({
            'time': time_seconds,
            'n_selected': n_selected,
            'config': config_name,
            'stage': 'Selection'
        })

    def record_clustering_full(self, time_seconds, n_features, config_name=""):
        """Record full dataset clustering time"""
        self.timings['clustering_full'].append({
            'time': time_seconds,
            'n_features': n_features,
            'config': config_name,
            'stage': 'Clustering (Full)'
        })

    def record_clustering_subset(self, time_seconds, n_features, config_name=""):
        """Record subset clustering time"""
        self.timings['clustering_subset'].append({
            'time': time_seconds,
            'n_features': n_features,
            'config': config_name,
            'stage': 'Clustering (Subset)'
        })

    def record_total_pipeline(self, time_seconds, config_name=""):
        """Record total pipeline time"""
        self.timings['total_pipeline'].append({
            'time': time_seconds,
            'config': config_name,
            'stage': 'Total'
        })

    def get_summary_stats(self):
        """Get summary statistics for all timings"""
        stats = {}

        for key, values in self.timings.items():
            if values:
                times = [v['time'] for v in values]
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }

        return stats

    def to_dataframe(self):
        """Convert timings to DataFrame"""
        all_records = []

        for stage, records in self.timings.items():
            for record in records:
                all_records.append({
                    'stage': stage,
                    **record
                })

        return pd.DataFrame(all_records)

    def save_to_csv(self, output_path):
        """Save timings to CSV"""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"✅ Saved timing data to: {output_path}")


def create_speed_comparison_visualization(timing_tracker, output_path=None, dpi=300,
                                          save_individual_panels=True, show_title=True,
                                          background_color='white'):
    """
    Create comprehensive speed comparison visualization.

    Parameters:
    -----------
    timing_tracker : TimingTracker
        TimingTracker instance with recorded timings
    output_path : str or Path, optional
        Where to save the figure
    dpi : int
        Resolution
    save_individual_panels : bool
        If True, saves each panel as a separate image
    show_title : bool
        If True, shows titles on panels and main figure
    background_color : str
        Background color for figures ('white', 'black', 'grey', etc.)
    """

    # Get data
    df = timing_tracker.to_dataframe()
    stats = timing_tracker.get_summary_stats()

    if df.empty:
        print("⚠️ No timing data available")
        return

    # Create figure with specified background
    fig = plt.figure(figsize=(18, 12), facecolor=background_color)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Setup for individual panel saving
    if save_individual_panels and output_path:
        individual_dir = Path(output_path).parent / f"{Path(output_path).stem}_panels"
        individual_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to save individual panel
    def save_panel(fig_panel, panel_name):
        if save_individual_panels and output_path:
            panel_path = individual_dir / f"{panel_name}.png"
            fig_panel.savefig(panel_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
            plt.close(fig_panel)
            return panel_path
        return None

    # ========================================================================
    # Panel 1: Average Time by Stage
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2], facecolor=background_color)

    stage_means = []
    stage_names = []
    stage_stds = []

    for stage, stat in stats.items():
        if stat['count'] > 0:
            stage_names.append(stage.replace('_', ' ').title())
            stage_means.append(stat['mean'])
            stage_stds.append(stat['std'])

    bars = ax1.barh(range(len(stage_names)), stage_means, xerr=stage_stds,
                    color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(stage_names)],
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_yticks(range(len(stage_names)))
    ax1.set_yticklabels(stage_names, fontsize=10)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    if show_title:
        ax1.set_title('Average Processing Time by Stage', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, stage_means, stage_stds)):
        ax1.text(mean + std + 0.5, bar.get_y() + bar.get_height()/2,
                f'{mean:.2f}s ± {std:.2f}s',
                va='center', fontsize=9, fontweight='bold')

    # ========================================================================
    # Panel 2: Speedup Factor (Full vs Subset)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2], facecolor=background_color)

    if 'clustering_full' in stats and 'clustering_subset' in stats:
        full_time = stats['clustering_full']['mean']
        subset_time = stats['clustering_subset']['mean']
        speedup = full_time / subset_time if subset_time > 0 else 0

        # Create gauge-style visualization
        ax2.barh([0], [speedup], color='#2ECC71', alpha=0.8, height=0.5,
                edgecolor='black', linewidth=2)
        ax2.set_xlim([0, max(speedup * 1.2, 2)])
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Speedup\nFactor'], fontsize=11, fontweight='bold')
        ax2.set_xlabel('Factor', fontsize=11)
        if show_title:
            ax2.set_title(f'Clustering Speedup: {speedup:.2f}x', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add reference line at 1x
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='1x (no speedup)')
        ax2.legend(fontsize=9)

        # Add value annotation
        ax2.text(speedup + 0.1, 0, f'{speedup:.2f}x',
                va='center', fontsize=14, fontweight='bold', color='darkgreen')

    # ========================================================================
    # Panel 3: Time vs Features (Clustering)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, :2], facecolor=background_color)

    # Combine clustering data
    clustering_data = []
    if 'clustering_full' in timing_tracker.timings:
        for record in timing_tracker.timings['clustering_full']:
            clustering_data.append({
                'n_features': record.get('n_features', 0),
                'time': record['time'],
                'type': 'Full Dataset'
            })

    if 'clustering_subset' in timing_tracker.timings:
        for record in timing_tracker.timings['clustering_subset']:
            clustering_data.append({
                'n_features': record.get('n_features', 0),
                'time': record['time'],
                'type': 'Selected Wavelengths'
            })

    if clustering_data:
        cluster_df = pd.DataFrame(clustering_data)

        # Plot scatter
        for ctype in cluster_df['type'].unique():
            subset = cluster_df[cluster_df['type'] == ctype]
            color = '#E74C3C' if 'Full' in ctype else '#2ECC71'
            ax3.scatter(subset['n_features'], subset['time'],
                       label=ctype, s=100, alpha=0.7, color=color,
                       edgecolors='black', linewidth=1)

        ax3.set_xlabel('Number of Features (Wavelengths)', fontsize=11)
        ax3.set_ylabel('Clustering Time (seconds)', fontsize=11)
        if show_title:
            ax3.set_title('Clustering Time vs Number of Features', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper left')
        ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 4: Total Time Breakdown (Pie Chart)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 2], facecolor=background_color)

    # Calculate total time per stage
    stage_totals = []
    stage_labels = []

    for stage, stat in stats.items():
        if stat['total'] > 0 and stage != 'total_pipeline':
            stage_labels.append(stage.replace('_', '\n').title())
            stage_totals.append(stat['total'])

    if stage_totals:
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(stage_totals)]
        wedges, texts, autotexts = ax4.pie(stage_totals, labels=stage_labels,
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90,
                                           textprops={'fontsize': 9, 'fontweight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)

        if show_title:
            ax4.set_title('Total Time Distribution', fontsize=12, fontweight='bold')

    # ========================================================================
    # Panel 5: Time per Configuration
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, :], facecolor=background_color)

    # Get all configurations
    all_configs = set()
    for records in timing_tracker.timings.values():
        for record in records:
            if 'config' in record and record['config']:
                all_configs.add(record['config'])

    if all_configs and len(all_configs) > 1:
        config_times = []

        for config in sorted(all_configs):
            total_time = 0
            for records in timing_tracker.timings.values():
                for record in records:
                    if record.get('config') == config:
                        total_time += record['time']

            config_times.append({
                'config': config[:20] + '...' if len(config) > 20 else config,
                'time': total_time
            })

        if config_times:
            config_df = pd.DataFrame(config_times).sort_values('time', ascending=True)

            bars = ax5.barh(range(len(config_df)), config_df['time'],
                           color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)

            ax5.set_yticks(range(len(config_df)))
            ax5.set_yticklabels(config_df['config'], fontsize=8)
            ax5.set_xlabel('Total Time (seconds)', fontsize=11)
            if show_title:
                ax5.set_title('Total Processing Time per Configuration', fontsize=13, fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, time) in enumerate(zip(bars, config_df['time'])):
                ax5.text(time + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{time:.2f}s', va='center', fontsize=8)

    # ========================================================================
    # Add summary statistics text box
    # ========================================================================
    if stats:
        summary_text = "PERFORMANCE SUMMARY\n" + "="*40 + "\n\n"

        if 'autoencoder_training' in stats:
            s = stats['autoencoder_training']
            summary_text += f"Autoencoder Training:\n"
            summary_text += f"  Mean: {s['mean']:.2f}s ± {s['std']:.2f}s\n"
            summary_text += f"  Range: {s['min']:.2f}s - {s['max']:.2f}s\n\n"

        if 'wavelength_selection' in stats:
            s = stats['wavelength_selection']
            summary_text += f"Wavelength Selection:\n"
            summary_text += f"  Mean: {s['mean']:.2f}s ± {s['std']:.2f}s\n"
            summary_text += f"  Total: {s['total']:.2f}s ({s['count']} runs)\n\n"

        if 'clustering_full' in stats and 'clustering_subset' in stats:
            full = stats['clustering_full']
            subset = stats['clustering_subset']
            speedup = full['mean'] / subset['mean'] if subset['mean'] > 0 else 0

            summary_text += f"Clustering Performance:\n"
            summary_text += f"  Full: {full['mean']:.2f}s ± {full['std']:.2f}s\n"
            summary_text += f"  Subset: {subset['mean']:.2f}s ± {subset['std']:.2f}s\n"
            summary_text += f"  Speedup: {speedup:.2f}x\n"
            summary_text += f"  Time Saved: {full['mean'] - subset['mean']:.2f}s\n\n"

        if 'total_pipeline' in stats:
            s = stats['total_pipeline']
            summary_text += f"Total Pipeline Time:\n"
            summary_text += f"  Mean: {s['mean']:.2f}s ({s['mean']/60:.2f} min)\n"
            summary_text += f"  Total: {s['total']:.2f}s ({s['total']/60:.2f} min)\n"

        fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Main title
    if show_title:
        plt.suptitle('Performance Timing Analysis - Speed Comparison',
                    fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save combined figure
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
        plt.savefig(str(output_path).replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight', facecolor=background_color)
        print(f"✅ Saved speed comparison to: {output_path}")

    plt.show()

    # ========================================================================
    # Save individual panels
    # ========================================================================
    if save_individual_panels and output_path:
        # Panel: Average time by stage
        fig_p1 = plt.figure(figsize=(10, 6), facecolor=background_color)
        ax_p1 = fig_p1.add_subplot(111, facecolor=background_color)
        bars_p1 = ax_p1.barh(range(len(stage_names)), stage_means, xerr=stage_stds,
                            color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(stage_names)],
                            alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_p1.set_yticks(range(len(stage_names)))
        ax_p1.set_yticklabels(stage_names, fontsize=10)
        ax_p1.set_xlabel('Time (seconds)', fontsize=11)
        if show_title:
            ax_p1.set_title('Average Processing Time by Stage', fontsize=13, fontweight='bold')
        ax_p1.grid(axis='x', alpha=0.3)
        for i, (mean, std) in enumerate(zip(stage_means, stage_stds)):
            ax_p1.text(mean + std + 0.5, i, f'{mean:.2f}s ± {std:.2f}s',
                      va='center', fontsize=9, fontweight='bold')
        save_panel(fig_p1, "01_average_time_by_stage")

        # Panel: Speedup factor
        if 'clustering_full' in stats and 'clustering_subset' in stats:
            fig_p2 = plt.figure(figsize=(8, 4), facecolor=background_color)
            ax_p2 = fig_p2.add_subplot(111, facecolor=background_color)
            full_time = stats['clustering_full']['mean']
            subset_time = stats['clustering_subset']['mean']
            speedup = full_time / subset_time if subset_time > 0 else 0
            ax_p2.barh([0], [speedup], color='#2ECC71', alpha=0.8, height=0.5,
                      edgecolor='black', linewidth=2)
            ax_p2.set_xlim([0, max(speedup * 1.2, 2)])
            ax_p2.set_ylim([-0.5, 0.5])
            ax_p2.set_yticks([0])
            ax_p2.set_yticklabels(['Speedup\nFactor'], fontsize=11, fontweight='bold')
            ax_p2.set_xlabel('Factor', fontsize=11)
            if show_title:
                ax_p2.set_title(f'Clustering Speedup: {speedup:.2f}x', fontsize=12, fontweight='bold')
            ax_p2.grid(axis='x', alpha=0.3)
            ax_p2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7,
                         label='1x (no speedup)')
            ax_p2.text(speedup + 0.1, 0, f'{speedup:.2f}x',
                      va='center', fontsize=14, fontweight='bold', color='darkgreen')
            ax_p2.legend(fontsize=9)
            save_panel(fig_p2, "02_speedup_factor")

        # Panel 3: Time vs Features
        if clustering_data:
            fig_p3 = plt.figure(figsize=(10, 6), facecolor=background_color)
            ax_p3 = fig_p3.add_subplot(111, facecolor=background_color)
            for ctype in cluster_df['type'].unique():
                subset = cluster_df[cluster_df['type'] == ctype]
                color = '#E74C3C' if 'Full' in ctype else '#2ECC71'
                ax_p3.scatter(subset['n_features'], subset['time'],
                             label=ctype, s=100, alpha=0.7, color=color,
                             edgecolors='black', linewidth=1)
            ax_p3.set_xlabel('Number of Features (Wavelengths)', fontsize=11)
            ax_p3.set_ylabel('Clustering Time (seconds)', fontsize=11)
            if show_title:
                ax_p3.set_title('Clustering Time vs Number of Features', fontsize=13, fontweight='bold')
            ax_p3.legend(fontsize=10, loc='upper left')
            ax_p3.grid(True, alpha=0.3)
            save_panel(fig_p3, "03_time_vs_features")

        # Panel 4: Total time breakdown pie chart
        if stage_totals:
            fig_p4 = plt.figure(figsize=(8, 8), facecolor=background_color)
            ax_p4 = fig_p4.add_subplot(111, facecolor=background_color)
            colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(stage_totals)]
            wedges, texts, autotexts = ax_p4.pie(stage_totals, labels=stage_labels,
                                                 colors=colors, autopct='%1.1f%%',
                                                 startangle=90,
                                                 textprops={'fontsize': 10, 'fontweight': 'bold'})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
            if show_title:
                ax_p4.set_title('Total Time Distribution', fontsize=14, fontweight='bold')
            save_panel(fig_p4, "04_time_distribution")

        # Panel 5: Time per configuration
        if all_configs and len(all_configs) > 1 and config_times:
            fig_p5 = plt.figure(figsize=(10, 8), facecolor=background_color)
            ax_p5 = fig_p5.add_subplot(111, facecolor=background_color)
            config_df_sorted = pd.DataFrame(config_times).sort_values('time', ascending=True)
            bars_p5 = ax_p5.barh(range(len(config_df_sorted)), config_df_sorted['time'],
                                color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
            ax_p5.set_yticks(range(len(config_df_sorted)))
            ax_p5.set_yticklabels(config_df_sorted['config'], fontsize=9)
            ax_p5.set_xlabel('Total Time (seconds)', fontsize=11)
            if show_title:
                ax_p5.set_title('Total Processing Time per Configuration', fontsize=13, fontweight='bold')
            ax_p5.grid(axis='x', alpha=0.3)
            for i, (bar, time) in enumerate(zip(bars_p5, config_df_sorted['time'])):
                ax_p5.text(time + 0.5, bar.get_y() + bar.get_height()/2,
                          f'{time:.2f}s', va='center', fontsize=9)
            save_panel(fig_p5, "05_time_per_configuration")

        if save_individual_panels:
            print(f"✅ Saved individual panels to: {individual_dir}")
            print(f"   Total panels saved: {len(list(individual_dir.glob('*.png')))}")


def create_simple_speed_comparison(full_time, subset_time, full_features, subset_features,
                                  output_path=None, save_individual_panels=True,
                                  show_title=True, background_color='white'):
    """
    Create a simple before/after speed comparison.

    Parameters:
    -----------
    full_time : float
        Time for full dataset (seconds)
    subset_time : float
        Time for subset dataset (seconds)
    full_features : int
        Number of features in full dataset
    subset_features : int
        Number of features in subset
    output_path : str or Path, optional
        Save path
    save_individual_panels : bool
        If True, saves each panel as a separate image
    show_title : bool
        If True, shows titles on panels and main figure
    background_color : str
        Background color for figures ('white', 'black', 'grey', etc.)
    """

    speedup = full_time / subset_time if subset_time > 0 else 0
    time_saved = full_time - subset_time
    time_saved_pct = (time_saved / full_time) * 100 if full_time > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=background_color)

    # Setup for individual panel saving
    if save_individual_panels and output_path:
        individual_dir = Path(output_path).parent / f"{Path(output_path).stem}_panels"
        individual_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to save individual panel
    def save_panel(fig_panel, panel_name):
        if save_individual_panels and output_path:
            panel_path = individual_dir / f"{panel_name}.png"
            fig_panel.savefig(panel_path, dpi=300, bbox_inches='tight', facecolor=background_color)
            plt.close(fig_panel)
            return panel_path
        return None

    # Panel 1: Time comparison
    times = [full_time, subset_time]
    labels = ['Full\nDataset', 'Optimized\nSubset']
    colors = ['#E74C3C', '#2ECC71']

    axes[0].set_facecolor(background_color)
    bars1 = axes[0].bar(labels, times, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    if show_title:
        axes[0].set_title('Processing Time Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    for bar, time in zip(bars1, times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    # Panel 2: Speedup factor
    axes[1].set_facecolor(background_color)
    axes[1].barh([0], [speedup], color='#2ECC71', alpha=0.8, height=0.5,
                edgecolor='black', linewidth=2)
    axes[1].axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_xlim([0, max(speedup * 1.2, 2)])
    axes[1].set_ylim([-0.5, 0.5])
    axes[1].set_yticks([])
    axes[1].set_xlabel('Speedup Factor', fontsize=12)
    if show_title:
        axes[1].set_title(f'Speedup: {speedup:.2f}x Faster', fontsize=13, fontweight='bold')
    axes[1].text(speedup/2, 0, f'{speedup:.2f}x', va='center', ha='center',
                fontsize=16, fontweight='bold', color='white')

    # Panel 3: Features vs Time
    axes[2].set_facecolor(background_color)
    axes[2].scatter([full_features], [full_time], s=200, color='#E74C3C',
                   label='Full Dataset', edgecolors='black', linewidth=2, alpha=0.8)
    axes[2].scatter([subset_features], [subset_time], s=200, color='#2ECC71',
                   label='Optimized Subset', edgecolors='black', linewidth=2, alpha=0.8)
    axes[2].plot([full_features, subset_features], [full_time, subset_time],
                'k--', alpha=0.5, linewidth=2)
    axes[2].set_xlabel('Number of Features', fontsize=12)
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    if show_title:
        axes[2].set_title('Efficiency Gain', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    if show_title:
        plt.suptitle(f'Speed Performance: {time_saved:.2f}s saved ({time_saved_pct:.1f}% reduction)',
                    fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=background_color)
        print(f"✅ Saved simple speed comparison to: {output_path}")

    # Save individual panels
    if save_individual_panels and output_path:
        # Panel 1: Time comparison
        fig_p1 = plt.figure(figsize=(6, 5), facecolor=background_color)
        ax_p1 = fig_p1.add_subplot(111, facecolor=background_color)
        bars_p1 = ax_p1.bar(labels, times, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=2)
        ax_p1.set_ylabel('Time (seconds)', fontsize=12)
        if show_title:
            ax_p1.set_title('Processing Time Comparison', fontsize=13, fontweight='bold')
        ax_p1.grid(axis='y', alpha=0.3)
        for bar, time in zip(bars_p1, times):
            height = bar.get_height()
            ax_p1.text(bar.get_x() + bar.get_width()/2., height,
                      f'{time:.2f}s', ha='center', va='bottom',
                      fontsize=11, fontweight='bold')
        save_panel(fig_p1, "01_time_comparison")

        # Panel 2: Speedup factor
        fig_p2 = plt.figure(figsize=(8, 4), facecolor=background_color)
        ax_p2 = fig_p2.add_subplot(111, facecolor=background_color)
        ax_p2.barh([0], [speedup], color='#2ECC71', alpha=0.8, height=0.5,
                  edgecolor='black', linewidth=2)
        ax_p2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_p2.set_xlim([0, max(speedup * 1.2, 2)])
        ax_p2.set_ylim([-0.5, 0.5])
        ax_p2.set_yticks([])
        ax_p2.set_xlabel('Speedup Factor', fontsize=12)
        if show_title:
            ax_p2.set_title(f'Speedup: {speedup:.2f}x Faster', fontsize=13, fontweight='bold')
        ax_p2.text(speedup/2, 0, f'{speedup:.2f}x', va='center', ha='center',
                  fontsize=16, fontweight='bold', color='white')
        save_panel(fig_p2, "02_speedup_factor")

        # Panel 3: Features vs Time
        fig_p3 = plt.figure(figsize=(8, 6), facecolor=background_color)
        ax_p3 = fig_p3.add_subplot(111, facecolor=background_color)
        ax_p3.scatter([full_features], [full_time], s=200, color='#E74C3C',
                     label='Full Dataset', edgecolors='black', linewidth=2, alpha=0.8)
        ax_p3.scatter([subset_features], [subset_time], s=200, color='#2ECC71',
                     label='Optimized Subset', edgecolors='black', linewidth=2, alpha=0.8)
        ax_p3.plot([full_features, subset_features], [full_time, subset_time],
                  'k--', alpha=0.5, linewidth=2)
        ax_p3.set_xlabel('Number of Features', fontsize=12)
        ax_p3.set_ylabel('Time (seconds)', fontsize=12)
        if show_title:
            ax_p3.set_title('Efficiency Gain', fontsize=13, fontweight='bold')
        ax_p3.legend(fontsize=10)
        ax_p3.grid(True, alpha=0.3)
        save_panel(fig_p3, "03_efficiency_gain")

        print(f"✅ Saved individual panels to: {individual_dir}")

    plt.show()


if __name__ == "__main__":
    print("Performance Timing Tracker Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from performance_timing_tracker import TimingTracker, PerformanceTimer")
    print("\n  tracker = TimingTracker()")
    print("  with PerformanceTimer('training') as timer:")
    print("      # your training code")
    print("  tracker.record_autoencoder_training(timer.elapsed, n_features)")
