import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the provided tables
data = {
    'Dataset': ['Reddit', 'Reddit', 'Reddit', 'Protein', 'Protein', 'Protein', 
                'Flickr', 'Flickr', 'Flickr', 'Yelp', 'Yelp', 'Yelp'],
    'Model': ['GCN', 'GIN', 'SAGE', 'GCN', 'GIN', 'SAGE', 
              'GCN', 'GIN', 'SAGE', 'GCN', 'GIN', 'SAGE'],
    'Without_AIA': [56.992, 54.346, 53.704, 33.708, 31.068, 30.682,
                    11.895, 10.318, 10.116, 78.721, 74.601, 70.456],
    'With_AIA': [47.809, 45.861, 45.274, 29.783, 27.598, 27.248,
                 9.128, 8.080, 7.901, 62.553, 59.667, 56.383],
    'CuSparse': [117.052, 114.736, 115.946, 71.378, 70.140, 70.554,
                 11.846, 10.934, 11.174, 82.568, 80.399, 82.845]
}

df = pd.DataFrame(data)

# Calculate improvement percentages
# Pruning+AIA over Pruning = (Without_AIA - With_AIA) / Without_AIA * 100
df['PruningAIA_over_Pruning'] = ((df['Without_AIA'] - df['With_AIA']) / df['Without_AIA']) * 100

# Pruning+AIA over Baseline = (CuSparse - With_AIA) / CuSparse * 100
df['PruningAIA_over_Baseline'] = ((df['CuSparse'] - df['With_AIA']) / df['CuSparse']) * 100

# Create x positions for bars
datasets = ['Reddit', 'Protein', 'Flickr', 'Yelp']
models = ['GCN', 'GIN', 'SAGE']
n_datasets = len(datasets)
n_models = len(models)

# Width of bars and spacing
bar_width = 0.25
group_spacing = 0.15
dataset_spacing = 1.2

# Calculate x positions
x_positions = []
labels_positions = []

for i, dataset in enumerate(datasets):
    dataset_start = i * (n_models * bar_width + group_spacing + dataset_spacing)
    
    for j, model in enumerate(models):
        x_pos = dataset_start + j * bar_width
        x_positions.append(x_pos)
        
    # Store center position for dataset label
    dataset_center = dataset_start + (n_models * bar_width) / 2
    labels_positions.append(dataset_center)

# Figure 1: Pruning+AIA over Pruning
fig1, ax1 = plt.subplots(figsize=(14, 8))

# Plot bars for Pruning+AIA over Pruning
colors_models = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green for GCN, GIN, SAGE
x_pos_idx = 0

for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        value = row['PruningAIA_over_Pruning'].iloc[0]
        
        x_pos = x_positions[x_pos_idx]
        bar = ax1.bar(x_pos, value, bar_width, color=colors_models[j], 
                     label=model if i == 0 else "", 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        ax1.text(x_pos, value + 0.5, f'{value:.1f}%', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
        
        x_pos_idx += 1

# Customize Figure 1
ax1.set_ylabel('Performance Improvement (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax1.set_title('Pruning+AIA Improvement over Pruning\nAcross Datasets and Models', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax1.set_xticks(labels_positions)
ax1.set_xticklabels(datasets, fontsize=12, fontweight='bold')

# Add legend
ax1.legend(loc='upper right', fontsize=12, framealpha=0.9, title='Model')

# Add grid
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis limits
max_val1 = df['PruningAIA_over_Pruning'].max()
ax1.set_ylim(0, max_val1 * 1.15)

# Add vertical lines to separate datasets
for i in range(1, len(datasets)):
    sep_x = (labels_positions[i-1] + labels_positions[i]) / 2
    ax1.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Increase tick label sizes
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)

plt.tight_layout()
plt.show()

# Save Figure 1
plt.savefig('pruning_aia_over_pruning.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('pruning_aia_over_pruning.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Figure 2: Pruning+AIA over Baseline
fig2, ax2 = plt.subplots(figsize=(14, 8))

# Plot bars for Pruning+AIA over Baseline
x_pos_idx = 0

for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        value = row['PruningAIA_over_Baseline'].iloc[0]
        
        x_pos = x_positions[x_pos_idx]
        bar = ax2.bar(x_pos, value, bar_width, color=colors_models[j], 
                     label=model if i == 0 else "", 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        ax2.text(x_pos, value + 1, f'{value:.1f}%', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
        
        x_pos_idx += 1

# Customize Figure 2
ax2.set_ylabel('Performance Improvement (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax2.set_title('Pruning+AIA Improvement over Baseline\nAcross Datasets and Models', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax2.set_xticks(labels_positions)
ax2.set_xticklabels(datasets, fontsize=12, fontweight='bold')

# Add legend
ax2.legend(loc='upper right', fontsize=12, framealpha=0.9, title='Model')

# Add grid
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis limits
max_val2 = df['PruningAIA_over_Baseline'].max()
ax2.set_ylim(0, max_val2 * 1.15)

# Add vertical lines to separate datasets
for i in range(1, len(datasets)):
    sep_x = (labels_positions[i-1] + labels_positions[i]) / 2
    ax2.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Increase tick label sizes
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)

plt.tight_layout()
plt.show()

# Save Figure 2
plt.savefig('pruning_aia_over_baseline.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('pruning_aia_over_baseline.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Print detailed comparison table
print("\nPruning+AIA Improvement Analysis - Two Separate Comparisons:")
print("="*80)
print(f"{'Dataset':<10} {'Model':<6} {'vs Pruning (%)':<15} {'vs Baseline (%)':<15}")
print("="*80)

for _, row in df.iterrows():
    print(f"{row['Dataset']:<10} {row['Model']:<6} {row['PruningAIA_over_Pruning']:<15.1f} "
          f"{row['PruningAIA_over_Baseline']:<15.1f}")

print("\n" + "="*80)
print("Summary Statistics:")
print(f"Average Pruning+AIA improvement over Pruning: {df['PruningAIA_over_Pruning'].mean():.1f}%")
print(f"Average Pruning+AIA improvement over Baseline: {df['PruningAIA_over_Baseline'].mean():.1f}%")
print(f"Best vs Pruning: {df.loc[df['PruningAIA_over_Pruning'].idxmax(), 'Dataset']} {df.loc[df['PruningAIA_over_Pruning'].idxmax(), 'Model']} ({df['PruningAIA_over_Pruning'].max():.1f}%)")
print(f"Best vs Baseline: {df.loc[df['PruningAIA_over_Baseline'].idxmax(), 'Dataset']} {df.loc[df['PruningAIA_over_Baseline'].idxmax(), 'Model']} ({df['PruningAIA_over_Baseline'].max():.1f}%)")
