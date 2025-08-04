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

# Set up the plot with larger figure
fig, ax = plt.subplots(figsize=(16, 10))

# Define colors and labels in new order: CuSparse -> Baseline, Without AIA -> Pruning, With AIA -> Pruning+AIA
colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
labels = ['Baseline', 'Pruning', 'Pruning+AIA']

# Create x positions for bars
datasets = ['Reddit', 'Protein', 'Flickr', 'Yelp']
models = ['GCN', 'GIN', 'SAGE']
n_datasets = len(datasets)
n_models = len(models)
n_conditions = 3

# Width of bars and spacing - increased bar width
bar_width = 0.25  # Increased from 0.08
group_spacing = 0.15
dataset_spacing = 1.2

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

# Prepare data for plotting in new order: CuSparse, Without_AIA, With_AIA
x_pos_idx = 0
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        
        # New order: CuSparse -> Baseline, Without AIA -> Pruning, With AIA -> Pruning+AIA
        values = [row['CuSparse'].iloc[0], row['Without_AIA'].iloc[0], row['With_AIA'].iloc[0]]
        
        for k, (value, color, label) in enumerate(zip(values, colors, labels)):
            x_pos = x_positions[x_pos_idx + k]
            bar = ax.bar(x_pos, value, bar_width, color=color, 
                        label=label if i == 0 and j == 0 else "", 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
        
        x_pos_idx += n_conditions

# Customize the plot with larger fonts
ax.set_ylabel('Training Time per Epoch (ms)', fontsize=16, fontweight='bold')  # Increased font size
ax.set_xlabel('Dataset and Model', fontsize=16, fontweight='bold')  # Increased font size
ax.set_title('GNN Training Time Performance Comparison\nAcross Datasets and Models', 
             fontsize=18, fontweight='bold', pad=25)  # Increased font size

# Set x-axis labels - only show model names
ax.set_xticks(labels_positions)
ax.set_xticklabels(dataset_labels, fontsize=12)

# Add legend in top right with larger font
ax.legend(loc='upper right', fontsize=14, framealpha=0.9)  # Changed to upper right and increased font size

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis to start from 0
ax.set_ylim(0, max(df[['Without_AIA', 'With_AIA', 'CuSparse']].max()) * 1.15)

# Add vertical lines to separate datasets
for i in range(1, len(datasets)):
    sep_x = (labels_positions[(i-1)*3 + 2] + labels_positions[i*3]) / 2
    ax.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Add dataset labels at the top - only once per dataset
dataset_centers = [1, 4, 7, 10]  # Center of each group of 3 models
max_y = max(df[['Without_AIA', 'With_AIA', 'CuSparse']].max()) * 1.1
for i, (center_idx, dataset) in enumerate(zip(dataset_centers, datasets)):
    center_x = labels_positions[center_idx]
    ax.text(center_x, max_y, dataset, ha='center', va='center',
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig('GNN_training_time_per_epoch_modified.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Print numerical comparison table
print("\nDetailed Performance Comparison:")
print("="*80)
print(f"{'Dataset':<10} {'Model':<6} {'Baseline':<12} {'Pruning':<12} {'Pruning+AIA':<12} {'Improvement':<12}")
print("="*80)

for _, row in df.iterrows():
    improvement = ((row['Without_AIA'] - row['With_AIA']) / row['Without_AIA']) * 100
    print(f"{row['Dataset']:<10} {row['Model']:<6} {row['CuSparse']:<12.1f} "
          f"{row['Without_AIA']:<12.1f} {row['With_AIA']:<12.1f} {improvement:<12.1f}%")
