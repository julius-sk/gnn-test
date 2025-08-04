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
colors = ['#3498db', '#2ecc71']  # Blue and Green
labels = ['Pruning+AIA over Pruning', 'Pruning+AIA over Baseline']

# Create y positions for bars (horizontal)
datasets = ['Reddit', 'Protein', 'Flickr', 'Yelp']
models = ['GCN', 'GIN', 'SAGE']
n_conditions = 2

bar_height = 0.35
group_spacing = 0.2
dataset_spacing = 1.5

# Calculate y positions
y_positions = []
label_positions = []
y_labels = []

for i, dataset in enumerate(datasets):
    dataset_start = i * (len(models) * n_conditions * bar_height + group_spacing + dataset_spacing)
    
    for j, model in enumerate(models):
        model_start = dataset_start + j * (n_conditions * bar_height + group_spacing)

        center = model_start + (n_conditions * bar_height) / 2
        label_positions.append(center)
        y_labels.append(f"{dataset} - {model}")
        
        for k in range(n_conditions):
            y_pos = model_start + k * bar_height
            y_positions.append(y_pos)

# Plot bars
y_pos_idx = 0
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        values = [row['PruningAIA_over_Pruning'].iloc[0], row['PruningAIA_over_Baseline'].iloc[0]]
        
        for k, (value, color, label) in enumerate(zip(values, colors, labels)):
            y_pos = y_positions[y_pos_idx + k]
            ax.barh(y_pos, value, height=bar_height, color=color,
                    label=label if i == 0 and j == 0 else "", 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels to the right of bars
            ax.text(value + 1, y_pos, f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
        
        y_pos_idx += n_conditions

# Set axis labels and title
ax.set_ylabel('Dataset and Model', fontsize=16, fontweight='bold')
ax.set_xlabel('Performance Improvement (%)', fontsize=16, fontweight='bold')
ax.set_title('GNN Training Time Improvement Ratio Comparison\nAcross Datasets and Models',
             fontsize=18, fontweight='bold', pad=25)

# Y-axis ticks and labels
ax.set_yticks(label_positions)
ax.set_yticklabels(y_labels, fontsize=12)

# Legend
ax.legend(loc='lower right', fontsize=14, framealpha=0.9)

# Gridlines
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# Set x-axis limits
max_improvement = max(df['PruningAIA_over_Baseline'].max(), df['PruningAIA_over_Pruning'].max())
ax.set_xlim(0, max_improvement * 1.2)

# Vertical line at 0%
ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

# Dataset separation lines
for i in range(1, len(datasets)):
    sep_y = (label_positions[(i-1)*3 + 2] + label_positions[i*3]) / 2
    ax.axhline(y=sep_y, color='gray', linestyle='-', alpha=0.5, linewidth=1)

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
