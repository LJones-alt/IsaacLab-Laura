import matplotlib.pyplot as plt
import pandas as pd

headers = [
    'joint pos 0', 'joint grad 0', 
    'joint pos 1', 'joint grad 1', 
    'joint pos 2', 'joint grad 2', 
    'joint pos 3', 'joint grad 3', 
    'joint pos 4', 'joint grad 4', 
    'joint pos 5', 'joint grad 5', 
    'joint pos 6', 'joint grad 6', 
    'safety_val'
]
df = pd.read_csv('docs/apf/joint_pos_vals.csv', header=None, names=headers)

# Single plot with twin axes (both curves on the same figure/axes sharing X-axis = joint pos 1)
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#1f77b4'  # blue
color2 = '#d62728'  # red
for i in range(7):
    ax1.plot(df[f'joint pos {i}'], df['safety_val'], color=color1, linewidth=2, label='Safety Value')

    ax1.set_xlabel(f'Joint {i} Position (rad)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Safety Value', color=color1, fontsize=11, fontweight='bold')
    line1 = ax1.plot(df[f'joint pos {i}'], df['safety_val'], color=color1, linewidth=2, label='Safety Value')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(f'Joint {i} Gradient', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(df[f'joint pos {i}'], df[f'joint grad {i}'], color=color2, linewidth=2, linestyle='--', label=f'Joint {i} Gradient')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)

    plt.title(f'Joint {i}: Position vs. Safety Value & Gradient', fontsize=14, fontweight='bold', pad=25)
    plt.savefig(f'docs/apf/joint_{i}_twin_axis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also let's create a 2-subplot figure (side-by-side without heatmaps) just in case or show twin axis as primary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Subplot 1: Pos vs Safety
    axes[0].plot(df[f'joint pos {i}'], df['safety_val'], color='#1f77b4', linewidth=2)
    axes[0].set_title(f'Joint {i} Position vs. Safety Value', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'Joint {i} Position (rad)', fontsize=10)
    axes[0].set_ylabel('Safety Value', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Pos vs Grad
    axes[1].plot(df[f'joint pos {i}'], df[f'joint grad {i}'], color='#d62728', linewidth=2)
    axes[1].set_title(f'Joint {i} Position vs. Joint {i} Gradient', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'Joint {i} Position (rad)', fontsize=10)
    axes[1].set_ylabel(f'Joint {i} Gradient', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'Joint {i} Position vs. Safety Value and Gradient', fontsize=14, fontweight='bold')
    plt.savefig(f'docs/apf/joint_{i}_two_subplots.png', dpi=300)
    plt.close()

    print("Both options created successfully.")