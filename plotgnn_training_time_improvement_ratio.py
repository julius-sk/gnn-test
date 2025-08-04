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

# Set up the plot with larger figure size
fig, ax = plt.subplots(figsize=(16, 10))

# Define colors and new labels
colors = ['#3498db', '#2ecc71']  # Blue for Pruning+AIA over Pruning, Green for Pruning+AIA over Baseline
labels = ['Pruning+AIA over Pruning', 'Pruning+AIA over Baseline']

# Create x positions for bars
datasets = ['Reddit', 'Protein', 'Flickr', 'Yelp']
models = ['GCN', 'GIN', 'SAGE']
n_datasets = len(datasets)
n_models = len(models)
n_conditions = 2  # Only two bars now

# Width of bars and spacing - increased bar width
bar_width = 0.35  # Increased bar width
group_spacing = 0.2
dataset_spacing = 1.5

# Calculate x positions
x_positions = []
labels_positions = []
dataset_labels = []

for i, dataset in enumerate(datasets):
    dataset_start = i * (n_models * n_conditions * bar_width + group_spacing + dataset_spacing)
    
    for j, model in enumerate(models):
        model_start = dataset_start + j * (n_conditions * bar_width + group_spacing)
        
        # Store center position for model label
        model_center = model_start + (n_conditions * bar_width) / 2
        labels_positions.append(model_center)
        dataset_labels.append(model)  # Only model name, dataset will be added separately
        
        for k in range(n_conditions):
            x_pos = model_start + k * bar_width
            x_positions.append(x_pos)

# Prepare data for plotting
x_pos_idx = 0
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        
        values = [row['PruningAIA_over_Pruning'].iloc[0], row['PruningAIA_over_Baseline'].iloc[0]]
        
        for k, (value, color, label) in enumerate(zip(values, colors, labels)):
            x_pos = x_positions[x_pos_idx + k]
            bar = ax.bar(x_pos, value, bar_width, color=color, 
                        label=label if i == 0 and j == 0 else "", 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on top of bars with larger font
            ax.text(x_pos, value + 1, f'{value:.1f}%', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        x_pos_idx += n_conditions

# Customize the plot with larger fonts
ax.set_xlabel('Performance Improvement (%)', fontsize=16, fontweight='bold')
ax.set_ylabel('Dataset and Model', fontsize=16, fontweight='bold')
ax.set_title('GNN Training Time Improvement Ratio Comparison\nAcross Datasets and Models', 
             fontsize=18, fontweight='bold', pad=25)

# Set y-axis labels - only show model names
ax.set_yticks(labels_positions)
ax.set_yticklabels(dataset_labels, fontsize=12)

# Add legend in top right with larger font
ax.legend(loc='lower right', fontsize=14, framealpha=0.9)

# Add grid for better readability
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# Set x-axis limits with more space
max_improvement = max(df['PruningAIA_over_Baseline'].max(), df['PruningAIA_over_Pruning'].max())
ax.set_xlim(0, max_improvement * 1.2)

# Add vertical line at 0% for reference
ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

# Add horizontal lines to separate datasets
for i in range(1, len(datasets)):
    sep_y = (labels_positions[(i-1)*3 + 2] + labels_positions[i*3]) / 2
    ax.axhline(y=sep_y, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Add dataset labels on the left - only once per dataset
dataset_centers = [1, 4, 7, 10]  # Center of each group of 3 models
for i, (center_idx, dataset) in enumerate(zip(dataset_centers, datasets)):
    center_y = labels_positions[center_idx]
    ax.text(-max_improvement * 0.08, center_y, dataset, ha='right', va='center',
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

# Increase tick label sizes
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Save the plot
plt.savefig('gnn_training_time_improvement_ratio_modified.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('gnn_training_time_improvement_ratio_modified.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Print detailed comparison table
print("\nPruning+AIA Improvement Percentage Analysis:")
print("="*100)
print(f"{'Dataset':<10} {'Model':<6} {'Pruning+AIA over':<18} {'Pruning+AIA over':<18} {'Difference':<15}")
print(f"{'':^10} {'':^6} {'Pruning (%)':<18} {'Baseline (%)':<18} {'(Baseline-Pruning)':<15}")
print("="*100)

for _, row in df.iterrows():
    difference = row['PruningAIA_over_Baseline'] - row['PruningAIA_over_Pruning']
    print(f"{row['Dataset']:<10} {row['Model']:<6} {row['PruningAIA_over_Pruning']:<18.1f} "
          f"{row['PruningAIA_over_Baseline']:<18.1f} {difference:<15.1f}")

print("\n" + "="*100)
print("Summary Statistics:")
print(f"Average Pruning+AIA improvement over Pruning: {df['PruningAIA_over_Pruning'].mean():.1f}%")
print(f"Average Pruning+AIA improvement over Baseline: {df['PruningAIA_over_Baseline'].mean():.1f}%")
print(f"Best Pruning+AIA vs Pruning: {df.loc[df['PruningAIA_over_Pruning'].idxmax(), 'Dataset']} {df.loc[df['PruningAIA_over_Pruning'].idxmax(), 'Model']} with {df['PruningAIA_over_Pruning'].max():.1f}% improvement")
print(f"Best Pruning+AIA vs Baseline: {df.loc[df['PruningAIA_over_Baseline'].idxmax(), 'Dataset']} {df.loc[df['PruningAIA_over_Baseline'].idxmax(), 'Model']} with {df['PruningAIA_over_Baseline'].max():.1f}% improvement")
