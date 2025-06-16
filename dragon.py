import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set styling for professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# Define the data for the bar chart
scenarios = ['Urban Dense', 'Suburban NLoS', 'Rural Wide', 'Disaster Zone', 'Mountain Terrain', 'Coastal Area']
models = ['P-PPO', 'C-PPO', 'C-DDPG', 'MPT', 'RSS', 'DRAGON']

# Values from the image (admission rates in %)
values = np.array([
    [75, 80, 70, 75, 75, 80],  # P-PPO
    [78, 80, 72, 80, 78, 82],  # C-PPO
    [80, 82, 70, 80, 80, 85],  # C-DDPG
    [70, 77, 65, 72, 70, 78],  # MPT
    [65, 72, 62, 68, 65, 72],  # RSS
    [85, 88, 80, 85, 82, 95]   # DRAGON
])

# Create a figure with GridSpec to control subplot layout
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

# Bar chart - top section
ax1 = plt.subplot(gs[0])

# Define professional colors that match the original image
colors = ['#3366CC', '#FF9933', '#33CC33', '#CC3333', '#9966CC', '#00CC99']

width = 0.12  # width of the bars
x = np.arange(len(scenarios))

# Create bars
for i in range(len(models)):
    ax1.bar(x + width*i - width*2.5, values[i], width, label=models[i], color=colors[i], edgecolor='black', linewidth=0.5)

# Add labels and title
ax1.set_ylabel('Admission Rate (%)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, rotation=0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure y-axis is visible and properly formatted
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, 100)
ax1.set_yticks(np.arange(0, 101, 20))  # Major ticks from 0 to 100 by 20
ax1.tick_params(axis='y', which='both', length=5)

# Add horizontal gridlines for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Position the legend
ax1.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.05))

# 3D trajectory plots - bottom section
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1], wspace=0.3)

# Sample trajectory data
def generate_trajectory(scenario_index):
    # Set random seed for reproducibility
    np.random.seed(42 + scenario_index)
    
    # Generate different trajectory patterns for each scenario
    t = np.linspace(0, 10, 100)
    
    if scenario_index == 0:  # Urban Dense
        x = 5 * np.cos(t) * np.exp(-0.1*t)
        y = 5 * np.sin(t) * np.exp(-0.1*t)
        z = 45 + 10 * np.sin(t/2)
        star_pos = [0, 0, 55]
    elif scenario_index == 1:  # Suburban NLoS
        x = 2.5 + np.sin(t)
        y = 5 * np.cos(t/2)
        z = 75 + 7.5 * np.sin(t)
        star_pos = [0, 0, 85]
    elif scenario_index == 2:  # Rural Wide
        x = 5 * np.sin(t/3)
        y = 10 * np.cos(t/5)
        z = 95 + 10 * np.sin(t/2)
        star_pos = [0, 0, 105]
    elif scenario_index == 3:  # Disaster Zone
        x = 7 * np.sin(t/2)
        y = 7 * np.cos(t/3)
        z = 65 + 10 * np.sin(t/4)
        star_pos = [5, 5, 75]
    elif scenario_index == 4:  # Mountain Terrain
        x = 10 * np.cos(t/4)
        y = 10 * np.sin(t/4)
        z = 110 + 20 * np.sin(t/6)
        star_pos = [0, 0, 130]
    else:  # Coastal Area
        x = 5 * np.sin(t/2)
        y = 5 * np.cos(t/3)
        z = 85 + 15 * np.sin(t/5)
        star_pos = [5, 0, 100]
    
    # Generate baseline trajectory (simplified)
    x_base = x * 0.8 + np.random.normal(0, 0.5, len(x))
    y_base = y * 0.8 + np.random.normal(0, 0.5, len(y))
    z_base = z * 0.9 + np.random.normal(0, 1, len(z))
    
    return x, y, z, x_base, y_base, z_base, star_pos

# Create 3D trajectory plots
for i in range(6):
    ax = fig.add_subplot(gs_bottom[i], projection='3d')
    x, y, z, x_base, y_base, z_base, star_pos = generate_trajectory(i)
    
    # Plot DRAGON trajectory (solid green line)
    dragon_line = ax.plot(x, y, z, color='#00CC99', linewidth=2, alpha=0.9)[0]
    
    # Plot baseline trajectory (dashed blue line)
    baseline_line = ax.plot(x_base, y_base, z_base, color='#3366CC', linewidth=1.5, linestyle='--', alpha=0.6)[0]
    
    # Plot RIS position (black star marker)
    ris_point = ax.scatter(star_pos[0], star_pos[1], star_pos[2], color='black', s=100, marker='*')
    
    # Set title and appearance
    ax.set_title(scenarios[i], fontsize=10, pad=5)
    
    # Ensure axes are visible with appropriate gridlines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Set ticks format with visible labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    
    # Ensure axis labels are readable
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)
    
    # Set view angle to match original figure
    ax.view_init(elev=20, azim=-35)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Only add a legend to the first 3D plot to avoid overcrowding
    if i == 0:
        ax.legend([dragon_line, baseline_line, ris_point], 
                 ['DRAGON', 'Baseline', 'RIS Position'],
                 loc='upper left', fontsize=8)

# Main title for the whole figure
fig.suptitle('DRAGON: IoT Device Admission Performance Analysis', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4)
plt.savefig('dragon_performance_analysis_improved.png', dpi=300, bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set styling for professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# Define the data for the bar chart
scenarios = ['Urban Dense', 'Suburban NLoS', 'Rural Wide', 'Disaster Zone', 'Mountain Terrain', 'Coastal Area']
models = ['P-PPO', 'C-PPO', 'C-DDPG', 'MPT', 'RSS', 'DRAGON']

# Values from the image (data rates in Mbps)
values = np.array([
    [10.5, 12.2, 9.8, 11.0, 10.8, 12.5],  # P-PPO
    [11.0, 12.5, 10.0, 11.5, 11.2, 13.0],  # C-PPO
    [11.2, 12.8, 10.2, 11.8, 11.5, 13.2],  # C-DDPG
    [9.8, 11.0, 9.5, 10.5, 10.2, 11.8],    # MPT
    [9.5, 10.8, 9.2, 10.2, 9.8, 11.2],     # RSS
    [12.9, 14.0, 11.8, 13.0, 12.8, 14.5]   # DRAGON
])

# Create a figure with GridSpec to control subplot layout
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

# Bar chart - top section
ax1 = plt.subplot(gs[0])

# Define professional colors that match the original image
colors = ['#3366CC', '#FF9933', '#33CC33', '#CC3333', '#9966CC', '#00CC99']

width = 0.12  # width of the bars
x = np.arange(len(scenarios))

# Create bars
for i in range(len(models)):
    bars = ax1.bar(x + width*i - width*2.5, values[i], width, label=models[i], color=colors[i], edgecolor='black', linewidth=0.5)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

# Add labels and title
ax1.set_ylabel('Data Rate (Mbps)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, rotation=0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure y-axis is visible and properly formatted
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, 16)
ax1.set_yticks(np.arange(0, 17, 2))  # Major ticks from 0 to 16 by 2
ax1.tick_params(axis='y', which='both', length=5)

# Add horizontal gridlines for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Position the legend
ax1.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.05))

# 3D trajectory plots - bottom section
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1], wspace=0.3)

# Sample trajectory data
def generate_trajectory(scenario_index):
    # Set random seed for reproducibility
    np.random.seed(42 + scenario_index)
    
    # Generate different trajectory patterns for each scenario
    t = np.linspace(0, 10, 100)
    
    if scenario_index == 0:  # Urban Dense
        x = 5 * np.cos(t) * np.exp(-0.1*t)
        y = 5 * np.sin(t) * np.exp(-0.1*t)
        z = 45 + 10 * np.sin(t/2)
        star_pos = [0, 0, 55]
        z_range = [45, 55]
    elif scenario_index == 1:  # Suburban NLoS
        x = 2.5 + np.sin(t)
        y = 5 * np.cos(t/2)
        z = 75 + 7.5 * np.sin(t)
        star_pos = [0, 0, 85]
        z_range = [75.0, 85.0]
    elif scenario_index == 2:  # Rural Wide
        x = 5 * np.sin(t/3)
        y = 10 * np.cos(t/5)
        z = 95 + 10 * np.sin(t/2)
        star_pos = [0, 0, 105]
        z_range = [95, 105]
    elif scenario_index == 3:  # Disaster Zone
        x = 7 * np.sin(t/2)
        y = 7 * np.cos(t/3)
        z = 65 + 10 * np.sin(t/4)
        star_pos = [5, 5, 75]
        z_range = [65, 75]
    elif scenario_index == 4:  # Mountain Terrain
        x = 10 * np.cos(t/4)
        y = 10 * np.sin(t/4)
        z = 110 + 20 * np.sin(t/6)
        star_pos = [0, 0, 130]
        z_range = [110, 130]
    else:  # Coastal Area
        x = 5 * np.sin(t/2)
        y = 5 * np.cos(t/3)
        z = 85 + 15 * np.sin(t/5)
        star_pos = [5, 0, 100]
        z_range = [85, 100]
    
    # Generate baseline trajectory (simplified)
    x_base = x * 0.8 + np.random.normal(0, 0.5, len(x))
    y_base = y * 0.8 + np.random.normal(0, 0.5, len(y))
    z_base = z * 0.9 + np.random.normal(0, 1, len(z))
    
    return x, y, z, x_base, y_base, z_base, star_pos, z_range

# Create 3D trajectory plots
for i in range(6):
    ax = fig.add_subplot(gs_bottom[i], projection='3d')
    x, y, z, x_base, y_base, z_base, star_pos, z_range = generate_trajectory(i)
    
    # Plot DRAGON trajectory (solid green line)
    dragon_line = ax.plot(x, y, z, color='#00CC99', linewidth=2, alpha=0.9)[0]
    
    # Plot baseline trajectory (dashed blue line)
    baseline_line = ax.plot(x_base, y_base, z_base, color='#3366CC', linewidth=1.5, linestyle='--', alpha=0.6)[0]
    
    # Plot RIS position (black star marker)
    ris_point = ax.scatter(star_pos[0], star_pos[1], star_pos[2], color='black', s=100, marker='*')
    
    # Set title and appearance
    ax.set_title(scenarios[i], fontsize=10, pad=5)
    
    # Ensure axes are visible with appropriate gridlines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Set ticks format with visible labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    
    # Ensure axis labels are readable
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)
    
    # Set view angle to match original figure
    ax.view_init(elev=20, azim=-35)
    
    # Set z-axis limits to match the original figure
    ax.set_zlim(z_range)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    
    # Only add a legend to the first 3D plot to avoid overcrowding
    if i == 0:
        ax.legend([dragon_line, baseline_line, ris_point], 
                 ['DRAGON', 'Baseline', 'RIS Position'],
                 loc='upper left', fontsize=8)

# Main title for the whole figure
fig.suptitle('DRAGON: Communication Performance Analysis', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4)
plt.savefig('dragon_communication_performance.png', dpi=300, bbox_inches='tight')
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Set styling for professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Define the data for the bar chart
scenarios = ['Urban\nDense', 'Suburban\nNLoS', 'Rural\nWide', 'Disaster\nZone', 'Mountain\nTerrain', 'Coastal\nArea']
models = ['P-PPO', 'C-PPO', 'C-DDPG', 'MPT', 'RSS', 'DRAGON']

# Energy Efficiency values (Mbps/J) from the image
values = np.array([
    [0.56, 0.62, 0.60, 0.54, 0.52, 0.68],  # P-PPO
    [0.64, 0.64, 0.68, 0.60, 0.58, 0.58],  # C-PPO
    [0.48, 0.52, 0.54, 0.46, 0.42, 0.58],  # C-DDPG
    [0.52, 0.58, 0.60, 0.50, 0.46, 0.64],  # MPT
    [0.50, 0.56, 0.58, 0.48, 0.48, 0.62],  # RSS
    [0.62, 0.64, 0.64, 0.58, 0.56, 0.70]   # DRAGON
])

# Create a figure with GridSpec to control subplot layout
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# Energy Efficiency Comparison - top section
ax1 = fig.add_subplot(gs[0])

# Define professional colors
colors = ['#FF6B6B', '#4ECDC4', '#1A535C', '#FFE66D', '#A8DADC', '#2EC4B6']

width = 0.12  # width of the bars
x = np.arange(len(scenarios))

# Create grouped bars
for i in range(len(models)):
    bars = ax1.bar(x + width*i - width*2.5, values[i], width, label=models[i], color=colors[i], edgecolor='black', linewidth=0.5)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

# Add labels and formatting
ax1.set_ylabel('Energy Efficiency (Mbps/J)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios)
ax1.set_ylim(0, 0.8)
ax1.set_yticks(np.arange(0, 0.9, 0.1))
ax1.set_title('Energy Efficiency Comparison Across Scenarios', fontsize=12, fontweight='bold', pad=10)

# Remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add legend below the chart
ax1.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.05))

# UAV Trajectory plots - bottom section
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1], wspace=0.1)

# Function to create elliptical trajectory paths
def generate_trajectory_data(scenario_index):
    # Set different parameters for each scenario
    if scenario_index == 0:  # Urban Dense
        center = (0, 0)
        width = 3
        height = 2
        angle = 30
        ris_pos = (0, -0.5)
    elif scenario_index == 1:  # Suburban NLoS
        center = (0, 0)
        width = 2.5
        height = 2
        angle = 45
        ris_pos = (-0.2, -0.5)
    elif scenario_index == 2:  # Rural Wide
        center = (0, 0)
        width = 3
        height = 1.5
        angle = 60
        ris_pos = (0.5, 0)
    elif scenario_index == 3:  # Disaster Zone
        center = (0, 0)
        width = 2
        height = 2
        angle = 15
        ris_pos = (-0.5, 0)
    elif scenario_index == 4:  # Mountain Terrain
        center = (0, 0)
        width = 2.5
        height = 1.8
        angle = 35
        ris_pos = (0, 0.5)
    else:  # Coastal Area
        center = (0, 0)
        width = 3
        height = 1.5
        angle = 20
        ris_pos = (1, 0)
        
    return center, width, height, angle, ris_pos

# Function to draw ellipse
def plot_ellipse(ax, center, width, height, angle, edgecolor, linestyle, alpha, label=None):
    ellipse = Ellipse(center, width, height, angle=angle, fill=False, 
                      edgecolor=edgecolor, linestyle=linestyle, linewidth=2, alpha=alpha, label=label)
    ax.add_patch(ellipse)
    
    # Create points along the ellipse for 3D effect
    t = np.linspace(0, 2*np.pi, 100)
    rotation = angle * np.pi/180
    x = center[0] + width/2 * np.cos(t) * np.cos(rotation) - height/2 * np.sin(t) * np.sin(rotation)
    y = center[1] + width/2 * np.cos(t) * np.sin(rotation) + height/2 * np.sin(t) * np.cos(rotation)
    
    return x, y

# Create UAV Trajectory plots
for i in range(6):
    ax = fig.add_subplot(gs_bottom[i], projection='3d')
    center, width, height, angle, ris_pos = generate_trajectory_data(i)
    
    # Create a figure-8 or elliptical pattern for DRAGON
    t = np.linspace(0, 2*np.pi, 100)
    
    # First ellipse for DRAGON (solid green)
    x1, y1 = plot_ellipse(ax, center, width, height, angle, '#2EC4B6', '-', 0.9)
    z1 = 0.2 + 0.1 * np.sin(3*t)
    dragon_line = ax.plot(x1, y1, z1, color='#2EC4B6', linewidth=2, alpha=0.9, label='DRAGON')[0]
    
    # Second ellipse for baseline (dashed red)
    x2, y2 = plot_ellipse(ax, center, width*0.8, height*0.8, angle-15, '#FF6B6B', '--', 0.7)
    z2 = 0.1 + 0.05 * np.sin(2*t)
    baseline_line = ax.plot(x2, y2, z2, color='#FF6B6B', linewidth=1.5, linestyle='--', alpha=0.7, label='Baseline')[0]
    
    # Plot RIS position (black star marker)
    ris_point = ax.scatter(ris_pos[0], ris_pos[1], 0, color='black', s=100, marker='*', label='RIS Position')
    
    # Set title and format the plot
    ax.set_title(scenarios[i].replace('\n', ' '), fontsize=10, pad=5)
    
    # Set axis labels
    ax.set_xlabel('X (m)', labelpad=-10, fontsize=8)
    ax.set_ylabel('Y (m)', labelpad=-10, fontsize=8)
    ax.set_zlabel('Z (m)', labelpad=-10, fontsize=8)
    
    # Remove grid lines and ticks for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Remove tick labels for cleaner appearance
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Set view angle
    ax.view_init(elev=25, azim=-45)
    
    # Set axis limits for consistent appearance
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 0.5)
    
    # Only add a legend to the first plot
    if i == 0:
        ax.legend(loc='upper left', fontsize=7)

# Main title for the whole figure
fig.suptitle('DRAGON: Comprehensive Performance Analysis', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.35)
plt.savefig('dragon_comprehensive_performance.png', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# Set the style for a professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create the figure with two subplots
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

# Set background to a very light gray for professional look
fig.patch.set_facecolor('#f8f9fa')

#==============================================
# TOP PLOT: TRAINING DYNAMICS
#==============================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#f8f9fa')

# Generate training data matching the figure
epochs = np.arange(0, 500)
training_loss_function = lambda x: 5 * np.exp(-0.01 * x) + 0.2 * np.exp(-0.005 * x) + 0.1
reward_function = lambda x: 2.0 * (1 - np.exp(-0.015 * x)) + 0.1 * (1 - np.exp(-0.01 * x)) 

# Calculate reward and loss values
reward_values = reward_function(epochs)
loss_values = training_loss_function(epochs)

# Add some noise for realism
np.random.seed(42)
reward_noise = np.random.normal(0, 0.05, len(epochs))
reward_values += reward_noise
loss_noise = np.random.normal(0, 0.05, len(epochs))
loss_values += loss_noise

# Calculate the confidence interval (for the shaded region)
reward_std = 0.15 * np.ones_like(reward_values)
reward_std[:100] = 0.05 + 0.1 * np.linspace(0, 1, 100)  # Increasing variance at the beginning
reward_upper = reward_values + reward_std
reward_lower = reward_values - reward_std
reward_lower = np.maximum(reward_lower, 0)  # Ensure non-negative values

# Twin axes for reward and loss
ax2 = ax1.twinx()

# Plot average reward with shaded confidence interval
reward_line = ax1.plot(epochs, reward_values, '-', color='#2ec4b6', linewidth=2.5, label='Average Reward')
ax1.fill_between(epochs, reward_lower, reward_upper, color='#2ec4b6', alpha=0.2)

# Plot training loss
loss_line = ax2.plot(epochs, loss_values, '--', color='#e71d36', linewidth=2, label='Training Loss')

# Add vertical line at convergence point (around epoch 250)
convergence_epoch = 250
ax1.axvline(x=convergence_epoch, color='#463f3a', linestyle=':', alpha=0.7, linewidth=1.5)

# Add annotation for convergence point
convergence_text = ax1.text(convergence_epoch + 10, 1.5, 'Policy Stabilization\n(Convergence Point)', 
               fontsize=10, color='#463f3a', alpha=0.8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))
convergence_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Set labels and limits
ax1.set_xlabel('Training Epochs', fontweight='bold')
ax1.set_ylabel('Average Reward', fontweight='bold', color='#2ec4b6')
ax2.set_ylabel('Training Loss', fontweight='bold', color='#e71d36')

ax1.set_xlim(0, 500)
ax1.set_ylim(0.0, 2.1)
ax2.set_ylim(0, 5.5)

# Customize ticks for cleaner appearance
ax1.tick_params(axis='y', colors='#2ec4b6')
ax2.tick_params(axis='y', colors='#e71d36')

# Add grid with lower opacity
ax1.grid(True, linestyle='--', alpha=0.3)

# Add title
ax1.set_title('DRAGON Training Dynamics', fontweight='bold', pad=10)

# Create combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
          framealpha=0.9, edgecolor='white')

#==============================================
# BOTTOM PLOT: OPTIMIZED UAV TRAJECTORY
#==============================================
ax3 = fig.add_subplot(gs[1])
ax3.set_facecolor('#f8f9fa')

# Create spiral trajectory data
theta = np.linspace(0, 4*np.pi, 1000)
r = np.linspace(0, 5, len(theta))
x = r * np.cos(theta)
y = r * np.sin(theta)

# Create mission time data (0-30 minutes)
mission_time = np.linspace(0, 30, len(theta))

# Create a custom colormap for the trajectory
colors = plt.cm.viridis(np.linspace(0, 1, len(x)))

# Define RIS positions
ris_positions = [(-5, -2), (0, 6), (6, 0)]

# Create a scatter plot for the trajectory with color mapped to mission time
scatter = ax3.scatter(x, y, c=mission_time, cmap='viridis', s=5, zorder=3)

# Create a line plot for the trajectory with the same color mapping
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='viridis', linewidth=2.5)
lc.set_array(mission_time[:-1])
ax3.add_collection(lc)

# Plot RIS positions with diamond markers
for i, (ris_x, ris_y) in enumerate(ris_positions):
    ax3.scatter(ris_x, ris_y, s=120, marker='D', color='#ff9f1c', 
               edgecolor='black', linewidth=1, zorder=4, 
               label='RIS Position' if i == 0 else "")

# Add coverage radius circles around RIS positions
for ris_x, ris_y in ris_positions:
    coverage_circle = plt.Circle((ris_x, ris_y), 4, fill=False, 
                              edgecolor='#b5179e', linewidth=2, 
                              linestyle='-', alpha=0.7, zorder=2)
    ax3.add_patch(coverage_circle)

# Set limits and labels
ax3.set_xlim(-8, 8)
ax3.set_ylim(-8, 8)
ax3.set_xlabel('X Position (km)', fontweight='bold')
ax3.set_ylabel('Y Position (km)', fontweight='bold')
ax3.set_aspect('equal')

# Add grid with lower opacity
ax3.grid(True, linestyle='--', alpha=0.3)

# Add colorbar for mission time
cbar = fig.colorbar(scatter, ax=ax3, pad=0.02)
cbar.set_label('Mission Time (min)', fontweight='bold')

# Add title
ax3.set_title('Optimized UAV Trajectory with RIS Coverage', fontweight='bold', pad=10)

# Add legend
ax3.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='white')

# Add a subtle background coverage heatmap (light blue shading for coverage)
x_grid, y_grid = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
coverage = np.zeros_like(x_grid)

for ris_x, ris_y in ris_positions:
    distance = np.sqrt((x_grid - ris_x)**2 + (y_grid - ris_y)**2)
    # Create coverage with falloff from each RIS
    coverage += 4 * np.exp(-0.15 * distance)

# Plot the heatmap with very light opacity
coverage_plot = ax3.contourf(x_grid, y_grid, coverage, 15, cmap='Blues', alpha=0.15, zorder=1)

# Add spiral center marker
ax3.scatter(0, 0, s=150, marker='*', color='navy', edgecolor='white', linewidth=1, zorder=5)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Save figure
plt.savefig('dragon_training_dynamics_enhanced.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

# Set the style for a professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create the figure with fixed size matching the original
fig = plt.figure(figsize=(9, 12), dpi =300)
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

# Set background to a very light gray for professional look
fig.patch.set_facecolor('#f8f9fa')

#==============================================
# TOP PLOT: TRAINING DYNAMICS
#==============================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#f8f9fa')

# Generate training data matching the figure
epochs = np.arange(0, 500)
training_loss_function = lambda x: 5 * np.exp(-0.01 * x) + 0.2 * np.exp(-0.005 * x) + 0.1
reward_function = lambda x: 2.0 * (1 - np.exp(-0.015 * x)) + 0.1 * (1 - np.exp(-0.01 * x)) 

# Calculate reward and loss values
reward_values = reward_function(epochs)
loss_values = training_loss_function(epochs)

# Add some noise for realism
np.random.seed(42)
reward_noise = np.random.normal(0, 0.05, len(epochs))
reward_values += reward_noise
loss_noise = np.random.normal(0, 0.05, len(epochs))
loss_values += loss_noise

# Calculate the confidence interval (for the shaded region)
reward_std = 0.15 * np.ones_like(reward_values)
reward_std[:100] = 0.05 + 0.1 * np.linspace(0, 1, 100)  # Increasing variance at the beginning
reward_upper = reward_values + reward_std
reward_lower = reward_values - reward_std
reward_lower = np.maximum(reward_lower, 0)  # Ensure non-negative values

# Twin axes for reward and loss
ax2 = ax1.twinx()

# Plot average reward with shaded confidence interval
reward_line = ax1.plot(epochs, reward_values, '-', color='#2ec4b6', linewidth=2.5, label='Average Reward')
ax1.fill_between(epochs, reward_lower, reward_upper, color='#2ec4b6', alpha=0.2)

# Plot training loss
loss_line = ax2.plot(epochs, loss_values, '--', color='#e71d36', linewidth=2, label='Training Loss')

# Add vertical line at convergence point (around epoch 250)
convergence_epoch = 250
ax1.axvline(x=convergence_epoch, color='#463f3a', linestyle=':', alpha=0.7, linewidth=1.5)

# Add annotation for convergence point
convergence_text = ax1.text(convergence_epoch + 10, 1.5, 'Policy Stabilization\n(Convergence Point)', 
               fontsize=10, color='#463f3a', alpha=0.8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))
convergence_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Set labels and limits
ax1.set_xlabel('Training Epochs', fontweight='bold')
ax1.set_ylabel('Average Reward', fontweight='bold', color='#2ec4b6')
ax2.set_ylabel('Training Loss', fontweight='bold', color='#e71d36')

ax1.set_xlim(0, 500)
ax1.set_ylim(0.0, 2.1)
ax2.set_ylim(0, 5.5)

# Customize ticks for cleaner appearance
ax1.tick_params(axis='y', colors='#2ec4b6')
ax2.tick_params(axis='y', colors='#e71d36')

# Add grid with lower opacity
ax1.grid(True, linestyle='--', alpha=0.3)

# Add title
ax1.set_title('DRAGON Training Dynamics', fontweight='bold', pad=10)

# Create combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
          framealpha=0.9, edgecolor='white')

#==============================================
# BOTTOM PLOT: OPTIMIZED UAV TRAJECTORY (2D)
#==============================================
ax3 = fig.add_subplot(gs[1])
ax3.set_facecolor('#f8f9fa')

# Create spiral trajectory data
theta = np.linspace(0, 4*np.pi, 1000)
r = np.linspace(0, 5, len(theta))
x = r * np.cos(theta)
y = r * np.sin(theta)

# Create mission time data (0-30 minutes)
mission_time = np.linspace(0, 30, len(theta))

# Define RIS positions
ris_positions = [(-5, -2), (0, 6), (6, 0)]

# Create a scatter plot for the trajectory with color mapped to mission time
scatter = ax3.scatter(x, y, c=mission_time, cmap='viridis', s=5, zorder=3)

# Create a line plot for the trajectory with the same color mapping
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='viridis', linewidth=2.5)
lc.set_array(mission_time[:-1])
ax3.add_collection(lc)

# Plot RIS positions with diamond markers
for i, (ris_x, ris_y) in enumerate(ris_positions):
    ax3.scatter(ris_x, ris_y, s=120, marker='D', color='#ff9f1c', 
               edgecolor='black', linewidth=1, zorder=4, 
               label='RIS Position' if i == 0 else "")

# Add coverage radius circles around RIS positions
for ris_x, ris_y in ris_positions:
    coverage_circle = plt.Circle((ris_x, ris_y), 4, fill=False, 
                              edgecolor='#b5179e', linewidth=2, 
                              linestyle='-', alpha=0.7, zorder=2)
    ax3.add_patch(coverage_circle)

# Set limits and labels
ax3.set_xlim(-8, 8)
ax3.set_ylim(-8, 8)
ax3.set_xlabel('X Position (km)', fontweight='bold')
ax3.set_ylabel('Y Position (km)', fontweight='bold')
ax3.set_aspect('equal')

# Add grid with lower opacity
ax3.grid(True, linestyle='--', alpha=0.3)

# Add colorbar for mission time
cbar = fig.colorbar(scatter, ax=ax3, pad=0.02)
cbar.set_label('Mission Time (min)', fontweight='bold')

# Add title
ax3.set_title('Optimized UAV Trajectory with RIS Coverage', fontweight='bold', pad=10)

# Add legend
ax3.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='white')

# Add a subtle background coverage heatmap (light blue shading for coverage)
x_grid, y_grid = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
coverage = np.zeros_like(x_grid)

for ris_x, ris_y in ris_positions:
    distance = np.sqrt((x_grid - ris_x)**2 + (y_grid - ris_y)**2)
    # Create coverage with falloff from each RIS
    coverage += 4 * np.exp(-0.15 * distance)

# Plot the heatmap with very light opacity
coverage_plot = ax3.contourf(x_grid, y_grid, coverage, 15, cmap='Blues', alpha=0.15, zorder=1)

# Add spiral center marker
ax3.scatter(0, 0, s=150, marker='*', color='navy', edgecolor='white', linewidth=1, zorder=5)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Save figure
plt.savefig('dragon_training_dynamics_fixed.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle

# Set the style for a professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create the figure with two subplots
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# Set background to a very light gray for professional look
fig.patch.set_facecolor('#f8f9fa')

#==================================================
# TOP PLOT: IoT Device Admission Rate vs Network Density
#==================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#f8f9fa')

# Generate x-axis data: Number of IoT Devices
num_devices = np.linspace(50, 500, 10)

# Generate admission rate data based on the figure
# DRAGON data
dragon_rate = 82 - 0.045 * (num_devices - 50)

# P-PPO data
pppo_rate = 75 - 0.09 * (num_devices - 50)

# C-DDPG data
cddpg_rate = 60 - 0.075 * (num_devices - 50)

# Add slight curve to make the lines more realistic
dragon_curve = 2 * np.sin(np.pi * (num_devices - 50) / 450) 
pppo_curve = 1.5 * np.sin(np.pi * (num_devices - 50) / 450)
cddpg_curve = 1 * np.sin(np.pi * (num_devices - 50) / 450)

dragon_rate += dragon_curve
pppo_rate += pppo_curve
cddpg_rate += cddpg_curve

# Plot the admission rates
dragon_line = ax1.plot(num_devices, dragon_rate, '-', color='#2ec4b6', linewidth=3, 
                      marker='o', markersize=8, markerfacecolor='white', markeredgewidth=1.5,
                      label='DRAGON')

pppo_line = ax1.plot(num_devices, pppo_rate, '-', color='#ff9f1c', linewidth=2.5, 
                    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                    label='P-PPO')

cddpg_line = ax1.plot(num_devices, cddpg_rate, '-', color='#e71d36', linewidth=2.5, 
                     marker='^', markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                     label='C-DDPG')

# Add the annotation box highlighting DRAGON's advantage
x_pos = 275
y_pos = 65
ax1.annotate('DRAGON Maintains Superior Admission Rates\nEven at High Device Densities', 
            xy=(x_pos, y_pos), xytext=(x_pos, y_pos),
            fontsize=10, color='#2f4858', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#2ec4b6'))

# Add a subtle arrow pointing to the DRAGON line
ax1.annotate('', xy=(350, 60), xytext=(x_pos + 100, y_pos),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#2ec4b6', connectionstyle='arc3,rad=0.2'))

# Set labels and limits
ax1.set_xlabel('Number of IoT Devices', fontweight='bold')
ax1.set_ylabel('Admission Success Rate (%)', fontweight='bold')
ax1.set_xlim(50, 500)
ax1.set_ylim(20, 85)

# Add grid with lower opacity
ax1.grid(True, linestyle='--', alpha=0.3)

# Add legend with better placement
ax1.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='white')

# Add title
ax1.set_title('IoT Device Admission Rate vs Network Density', fontweight='bold', pad=10)

#==================================================
# BOTTOM PLOT: DRAGON UAV Trajectory with RIS Coverage Zones
#==================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#f8f9fa')

# Create trajectory data - a curved path
t = np.linspace(0, 2*np.pi, 500)
r = 4
x = r * np.sin(t)
y = r * np.cos(t) - 2 * np.sin(t/2)

# Time along the trajectory (0-20 minutes)
mission_time = np.linspace(0, 20, len(t))

# Define RIS positions
ris_positions = [(-3, 2), (0, -3), (4, -1)]

# Create a line plot for the trajectory with color mapping
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='viridis', linewidth=3.5)
lc.set_array(mission_time[:-1])
line = ax2.add_collection(lc)

# Add RIS positions with X markers
for i, (ris_x, ris_y) in enumerate(ris_positions):
    # Add a light red circular coverage zone
    coverage_circle = Circle((ris_x, ris_y), 2, color='#ffccd5', alpha=0.3, zorder=1)
    ax2.add_patch(coverage_circle)
    
    # Add X marker
    ax2.scatter(ris_x, ris_y, s=120, marker='x', color='red', linewidth=2, zorder=3)

# Set limits and labels
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('X Position (km)', fontweight='bold')
ax2.set_ylabel('Y Position (km)', fontweight='bold')
ax2.set_aspect('equal')

# Add grid with lower opacity
ax2.grid(True, linestyle='--', alpha=0.3)

# Add colorbar for mission time
cbar = fig.colorbar(line, ax=ax2, pad=0.02)
cbar.set_label('Mission Time (min)', fontweight='bold')

# Add arrows along the path to show direction
arrow_positions = [100, 200, 300, 400]
for pos in arrow_positions:
    if pos < len(x) - 1:
        dx = x[pos+1] - x[pos]
        dy = y[pos+1] - y[pos]
        ax2.arrow(x[pos], y[pos], dx*0.8, dy*0.8, head_width=0.2, head_length=0.3, 
                 fc='white', ec='black', zorder=4, width=0.05)

# Add title
ax2.set_title('DRAGON UAV Trajectory with RIS Coverage Zones', fontweight='bold', pad=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Save figure
plt.savefig('dragon_admission_trajectory.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects

# Set the style for a professional appearance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create the figure with two subplots
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# Set background to a very light gray for professional look
fig.patch.set_facecolor('#f9f9f9')

#==============================================
# TOP PLOT: UAV ENERGY EFFICIENCY VS RIS SIZE
#==============================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#f9f9f9')

# RIS element sizes
ris_sizes = np.array([64, 128, 256])

# Energy efficiency data (Mbps/J)
dragon_ee = np.array([75, 83, 92])
pppo_ee = np.array([58, 63, 67])
cddpg_ee = np.array([52, 57, 61])
mpt_ee = np.array([48, 52, 55])

# Confidence interval for DRAGON (to create the shaded region)
dragon_ee_lower = dragon_ee - 3
dragon_ee_upper = dragon_ee + 3

# Plot the energy efficiency curves
dragon_line = ax1.plot(ris_sizes, dragon_ee, '-', color='#5a189a', linewidth=3, 
                      marker='o', markersize=8, markerfacecolor='white', markeredgewidth=1.5,
                      label='DRAGON')

# Add confidence interval shading for DRAGON
ax1.fill_between(ris_sizes, dragon_ee_lower, dragon_ee_upper, color='#5a189a', alpha=0.2)

pppo_line = ax1.plot(ris_sizes, pppo_ee, '--', color='#3a86ff', linewidth=2, 
                    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                    label='P-PPO')

cddpg_line = ax1.plot(ris_sizes, cddpg_ee, '--', color='#118ab2', linewidth=2, 
                     marker='^', markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                     label='C-DDPG')

mpt_line = ax1.plot(ris_sizes, mpt_ee, '--', color='#06d6a0', linewidth=2, 
                   marker='d', markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                   label='MPT')

# Add annotation for DRAGON peak performance
ax1.annotate('DRAGON Peak Performance', 
            xy=(256, 92), xytext=(220, 85),
            fontsize=10, color='#4c9627', fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#4c9627'))

# Set labels and limits
ax1.set_xlabel('RIS Elements (Number)', fontweight='bold')
ax1.set_ylabel('Energy Efficiency (Mbps/J)', fontweight='bold')
ax1.set_xlim(50, 270)
ax1.set_ylim(45, 95)

# Set custom x-ticks to match the data points
ax1.set_xticks(ris_sizes)
ax1.set_xticklabels(ris_sizes)

# Add grid with lower opacity
ax1.grid(True, linestyle='--', alpha=0.3)

# Add legend with better placement
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='white')

# Add title
ax1.set_title('UAV Energy Efficiency vs RIS Size', fontweight='bold', pad=10)

#==============================================
# BOTTOM PLOT: DRAGON UAV TRAJECTORY OPTIMIZATION
#==============================================
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#f9f9f9')

# Create a figure-8 trajectory
t = np.linspace(0, 2*np.pi, 1000)
a, b = 1.5, 1.0  # Width, height parameters
x = a * np.sin(t)
y = b * np.sin(t) * np.cos(t)

# Mission time data (0-100 units)
mission_time = np.linspace(0, 100, len(t))

# Compute optimization points (hovering locations)
hover_points = [
    (-1.0, -0.5),  # Left hover point
    (0.0, 0.0),    # Center
    (1.0, 0.5)     # Right hover point
]

# Create a line plot for the trajectory with color mapping
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='viridis', linewidth=4)
lc.set_array(mission_time[:-1])
line = ax2.add_collection(lc)

# Add hover points with red ellipses
for i, (hover_x, hover_y) in enumerate(hover_points):
    # Add elliptical hover zones
    hover_ellipse = Ellipse((hover_x, hover_y), width=0.4, height=0.2, 
                           angle=30 if i == 0 else (0 if i == 1 else -30),
                           facecolor='#ffccd5', edgecolor='#e63946', 
                           linewidth=1.5, alpha=0.7, zorder=2)
    ax2.add_patch(hover_ellipse)
    
    # Add a smaller inner ellipse for emphasis
    inner_ellipse = Ellipse((hover_x, hover_y), width=0.2, height=0.1, 
                           angle=30 if i == 0 else (0 if i == 1 else -30),
                           facecolor='#e63946', alpha=0.4, zorder=3)
    ax2.add_patch(inner_ellipse)

# Add special markers at specific trajectory points
special_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4]
for idx in special_indices:
    ax2.scatter(x[idx], y[idx], s=80, marker='o', color='yellow', 
                edgecolor='black', linewidth=1.5, zorder=4)

# Set limits and labels
ax2.set_xlim(-2, 2)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('X Position (km)', fontweight='bold')
ax2.set_ylabel('Y Position (km)', fontweight='bold')
ax2.set_aspect('equal')

# Add grid with lower opacity
ax2.grid(True, linestyle='--', alpha=0.3)

# Add colorbar for mission time
cbar = fig.colorbar(line, ax=ax2, pad=0.02)
cbar.set_label('Mission Time (s)', fontweight='bold')

# Add direction arrows along the trajectory
arrow_positions = [100, 300, 500, 700, 900]
for i in arrow_positions:
    if i < len(x) - 1:
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        length = np.sqrt(dx**2 + dy**2)
        ax2.arrow(x[i], y[i], dx*0.6, dy*0.6, 
                 head_width=0.06, head_length=0.1, 
                 fc='white', ec='black', zorder=5)

# Add title
ax2.set_title('DRAGON UAV Trajectory Optimization', fontweight='bold', pad=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Save figure
plt.savefig('dragon_energy_trajectory.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

# Set professional style parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Create figure with custom layout
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Set overall figure title
fig.suptitle('DRAGON: Performance Evaluation', fontsize=16, fontweight='bold', y=0.98)

# Background color for a professional look
fig.patch.set_facecolor('#f8f9fa')

#==================================================
# LEFT PLOT: Energy Efficiency Comparison
#==================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#f0f5fa')  # Light blue background

# Data for energy efficiency vs RIS elements
ris_sizes = np.array([64, 128, 192, 256])  # RIS element sizes
models = ['DRAGON', 'P-PPO', 'C-DDPG', 'MPT', 'RSS']
    
# Energy efficiency values (Mbps/J)
ee_values = {
    'DRAGON': [12, 16, 19, 24],
    'P-PPO': [6, 8, 9.5, 11],
    'C-DDPG': [5, 6.5, 8, 9.5],
    'MPT': [4, 5, 6, 7],
    'RSS': [3, 3.5, 4, 4.5]
}

# Colors and line styles for each model
colors = {
    'DRAGON': '#0077b6',
    'P-PPO': '#ef476f',
    'C-DDPG': '#06d6a0',
    'MPT': '#ffd166',
    'RSS': '#8338ec'
}

markers = {
    'DRAGON': 'o',
    'P-PPO': 's',
    'C-DDPG': '^',
    'MPT': 'd',
    'RSS': 'X'
}

# Plot energy efficiency for each model
for model in models:
    ax1.plot(ris_sizes, ee_values[model], 
            color=colors[model], 
            marker=markers[model], 
            linewidth=2.5 if model == 'DRAGON' else 2,
            markersize=8 if model == 'DRAGON' else 6,
            label=model)
    
    # Add markers with white center for better visibility
    if model == 'DRAGON':
        ax1.plot(ris_sizes, ee_values[model], 
                color=colors[model], 
                marker=markers[model], 
                linewidth=0,
                markerfacecolor='white',
                markeredgewidth=2,
                markersize=5)

# Add annotation for DRAGON's best performance
best_x = ris_sizes[-1]
best_y = ee_values['DRAGON'][-1]
ax1.annotate(f"Peak EE: {best_y} Mbps/J", 
            xy=(best_x, best_y), 
            xytext=(best_x-40, best_y+2),
            fontsize=9,
            fontweight='bold',
            color='#0077b6',
            arrowprops=dict(arrowstyle='->',
                          connectionstyle="arc3,rad=.2",
                          color='#0077b6'))

# Add light blue shaded region for DRAGON improvement area
ax1.fill_between(ris_sizes, 
                ee_values['P-PPO'], 
                ee_values['DRAGON'], 
                color='#0077b6', 
                alpha=0.1,
                label='DRAGON Improvement')

# Set labels and title
ax1.set_xlabel('RIS Elements (Number)', fontweight='bold')
ax1.set_ylabel('Energy Efficiency (Mbps/J)', fontweight='bold')
ax1.set_title('Energy Efficiency Comparison', fontweight='bold', pad=10)

# Set limits
ax1.set_xlim(50, 270)
ax1.set_ylim(0, 30)

# Add grid with dashed lines
ax1.grid(True, linestyle='--', alpha=0.7, color='#dcdcdc')

# Create custom legend with color squares
legend = ax1.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='gray')
frame = legend.get_frame()
frame.set_boxstyle('round,pad=0.5')
frame.set_facecolor('white')

#==================================================
# RIGHT PLOT: UAV Trajectory Comparison
#==================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#f8f8f8')  # Light gray background

# Generate trajectory data
# DRAGON trajectory (spiral pattern)
theta_dragon = np.linspace(0, 4*np.pi, 500)
r_dragon = np.linspace(0.5, 10, len(theta_dragon))
x_dragon = r_dragon * np.cos(theta_dragon)
y_dragon = r_dragon * np.sin(theta_dragon)

# Baseline trajectory (more erratic pattern)
theta_baseline = np.linspace(0, 4*np.pi, 250)
r_baseline = 3 + 2 * np.random.rand(len(theta_baseline))
x_baseline = r_baseline * np.cos(theta_baseline)
y_baseline = r_baseline * np.sin(theta_baseline)

# Plot the trajectories
ax2.plot(x_dragon, y_dragon, color='#0077b6', linewidth=2.5, label='DRAGON')
ax2.plot(x_baseline, y_baseline, color='#ef476f', linewidth=2, linestyle='--', label='With RIS')

# Add RIS positions
ris_positions = [(0, 0), (5, 5), (-5, -5)]
for i, (x, y) in enumerate(ris_positions):
    ax2.scatter(x, y, s=100, color='#ffd166', edgecolor='black', linewidth=1.5, zorder=5,
               marker='*', label='RIS Position' if i == 0 else "")

# Set labels and title
ax2.set_xlabel('X Position (km)', fontweight='bold')
ax2.set_ylabel('Y Position (km)', fontweight='bold')
ax2.set_title('UAV Trajectory Comparison', fontweight='bold', pad=10)

# Add grid
ax2.grid(True, linestyle='--', alpha=0.7, color='#dcdcdc')

# Create legend with custom styling
legend2 = ax2.legend(loc='lower right', frameon=True)
frame2 = legend2.get_frame()
frame2.set_facecolor('#f8f8f8')
frame2.set_edgecolor('gray')
frame2.set_boxstyle('round,pad=0.5')

# Set equal aspect ratio for trajectory plot
ax2.set_aspect('equal')
ax2.set_xlim(-15, 15)
ax2.set_ylim(-15, 15)

# Add arrows to indicate direction of movement
arrow_indices_dragon = [100, 200, 300, 400]
for idx in arrow_indices_dragon:
    if idx < len(x_dragon) - 10:
        dx = x_dragon[idx+10] - x_dragon[idx]
        dy = y_dragon[idx+10] - y_dragon[idx]
        ax2.annotate('', 
                   xy=(x_dragon[idx]+dx, y_dragon[idx]+dy),
                   xytext=(x_dragon[idx], y_dragon[idx]),
                   arrowprops=dict(arrowstyle='->', color='#0077b6', lw=2))

arrow_indices_baseline = [50, 100, 150, 200]
for idx in arrow_indices_baseline:
    if idx < len(x_baseline) - 5:
        dx = x_baseline[idx+5] - x_baseline[idx]
        dy = y_baseline[idx+5] - y_baseline[idx]
        ax2.annotate('', 
                   xy=(x_baseline[idx]+dx, y_baseline[idx]+dy),
                   xytext=(x_baseline[idx], y_baseline[idx]),
                   arrowprops=dict(arrowstyle='->', color='#ef476f', lw=1.5))

# Add UAV start positions
ax2.scatter(x_dragon[0], y_dragon[0], s=120, marker='o', color='#0077b6', 
           edgecolor='white', linewidth=1.5, zorder=6, label='DRAGON Start')
           
ax2.scatter(x_baseline[0], y_baseline[0], s=120, marker='o', color='#ef476f',
           edgecolor='white', linewidth=1.5, zorder=6, label='Baseline Start')

# Add subtle background gradient effect
# Create a radial gradient from the center
x = np.linspace(-15, 15, 100)
y = np.linspace(-15, 15, 100)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Create coverage gradient
coverage = np.zeros_like(R)
for x_ris, y_ris in ris_positions:
    dist = np.sqrt((X - x_ris)**2 + (Y - y_ris)**2)
    coverage += 10 * np.exp(-0.1 * dist)

# Plot coverage with very light opacity
ax2.contourf(X, Y, coverage, 20, cmap='Blues', alpha=0.1, zorder=1)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig('dragon_performance_evaluation.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as path_effects

# Custom 3D arrow class
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Set professional style parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16

# Create figure with custom layout
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

# Set background to a very light gray for professional look
fig.patch.set_facecolor('#f9f9f9')

# Define common colors
dragon_color = '#00b894'  # Teal for DRAGON
pppo_color = '#e17055'    # Coral for P-PPO
cddpg_color = '#74b9ff'   # Light blue for C-DDPG
mpt_color = '#fdcb6e'     # Yellow for MPT

#=================================================
# TOP LEFT: 3D UAV Trajectories with RIS Coordination
#=================================================
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_facecolor('#f8f8f8')

# Generate trajectory data
theta = np.linspace(0, 4*np.pi, 500)
z_height = np.linspace(20, 30, len(theta))
r = 8 + np.sin(theta) * 2
x_dragon = r * np.cos(theta)
y_dragon = r * np.sin(theta)
z_dragon = z_height

# Baseline trajectory (more erratic path)
theta_base = np.linspace(0, 4*np.pi, 300)
r_base = 6 + np.sin(theta_base*1.5) * 3
x_base = r_base * np.cos(theta_base)
y_base = r_base * np.sin(theta_base)
z_base = np.linspace(18, 26, len(theta_base)) + np.sin(theta_base*2) * 2

# Plot trajectories
dragon_line = ax1.plot(x_dragon, y_dragon, z_dragon, color=dragon_color, linewidth=3, label='DRAGON Trajectory')[0]
base_line = ax1.plot(x_base, y_base, z_base, color=pppo_color, linewidth=2, linestyle='--', label='Baseline Trajectory')[0]

# Add RIS positions (projections on ground plane)
ris_positions = [(5, 5, 0), (-5, 5, 0), (0, -8, 0)]
for i, (x, y, z) in enumerate(ris_positions):
    ax1.scatter(x, y, z, s=100, color='red', marker='*', edgecolor='black', linewidth=1, label='RIS Position' if i == 0 else "")
    
    # Add vertical lines connecting RIS to optimal points on the DRAGON trajectory
    # Find closest point on the DRAGON trajectory to this RIS
    distances = np.sqrt((x_dragon - x)**2 + (y_dragon - y)**2)
    idx = np.argmin(distances)
    ax1.plot([x, x_dragon[idx]], [y, y_dragon[idx]], [z, z_dragon[idx]], 
            color='red', linestyle='--', alpha=0.6, linewidth=1.5)

# Add arrows to show direction
for i in range(0, len(x_dragon), 100):
    if i+10 < len(x_dragon):
        arrow = Arrow3D([x_dragon[i], x_dragon[i+10]], 
                       [y_dragon[i], y_dragon[i+10]], 
                       [z_dragon[i], z_dragon[i+10]], 
                       mutation_scale=15, lw=2, arrowstyle='-|>', color=dragon_color)
        ax1.add_artist(arrow)

# Set axis labels and limits
ax1.set_xlabel('X Position (m)', fontweight='bold', labelpad=10)
ax1.set_ylabel('Y Position (m)', fontweight='bold', labelpad=10)
ax1.set_zlabel('Altitude (m)', fontweight='bold', labelpad=10)
ax1.set_xlim(-15, 15)
ax1.set_ylim(-15, 15)
ax1.set_zlim(0, 35)

# Add light grid for reference
ax1.grid(True, linestyle='--', alpha=0.3)

# Set optimal viewing angle
ax1.view_init(elev=30, azim=45)

# Add title
ax1.set_title('3D UAV Trajectories with RIS Coordination', fontweight='bold', pad=20)

# Customize legend
legend1 = ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
frame1 = legend1.get_frame()
frame1.set_facecolor('white')
frame1.set_edgecolor('gray')

#=================================================
# TOP RIGHT: Energy Efficiency Comparison
#=================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#f8f8f8')

# Generate time steps
time_steps = np.linspace(0, 100, 100)

# Efficiency values
dragon_ee = np.zeros_like(time_steps)
pppo_ee = np.zeros_like(time_steps)
cddpg_ee = np.zeros_like(time_steps)

# Set values based on time periods
# Initial phase (0-20)
dragon_ee[:20] = 30
pppo_ee[:20] = 25
cddpg_ee[:20] = 20

# Transition phase (20-40)
dragon_ee[20:40] = np.linspace(30, 45, 20)
pppo_ee[20:40] = np.linspace(25, 35, 20)
cddpg_ee[20:40] = np.linspace(20, 30, 20)

# Stable phase (40-100)
dragon_ee[40:] = 45
pppo_ee[40:] = 35
cddpg_ee[40:] = 30

# Plot efficiency curves
dragon_line = ax2.plot(time_steps, dragon_ee, '-', color=dragon_color, linewidth=3, label='DRAGON')[0]
pppo_line = ax2.plot(time_steps, pppo_ee, '--', color=pppo_color, linewidth=2, label='P-PPO')[0]
cddpg_line = ax2.plot(time_steps, cddpg_ee, '-.', color=cddpg_color, linewidth=2, label='C-DDPG')[0]

# Set labels and limits
ax2.set_xlabel('Time (s)', fontweight='bold')
ax2.set_ylabel('EE (Mbps/J)', fontweight='bold')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 50)

# Add grid
ax2.grid(True, linestyle='--', alpha=0.3)

# Add title
ax2.set_title('Energy Efficiency Comparison', fontweight='bold', pad=10)

# Add legend
legend2 = ax2.legend(loc='upper right', frameon=True)
frame2 = legend2.get_frame()
frame2.set_facecolor('white')

#=================================================
# BOTTOM LEFT: Multi-Metric Performance Radar
#=================================================
ax3 = fig.add_subplot(gs[1, 0], polar=True)
ax3.set_facecolor('#f8f8f8')

# Define the metrics and their scales
metrics = ['Energy\nEfficiency', 'Data Rate', 'Admission\nRate', 'Latency', 'Coverage']
num_metrics = len(metrics)

# Data for different algorithms (scaled 0-1 for the radar chart)
# Higher is better for all metrics (we'll invert latency when plotting)
dragon_scores = np.array([0.9, 0.85, 0.92, 0.88, 0.95])
pppo_scores = np.array([0.7, 0.65, 0.75, 0.72, 0.8])
cddpg_scores = np.array([0.6, 0.55, 0.65, 0.63, 0.7])
mpt_scores = np.array([0.5, 0.45, 0.55, 0.52, 0.6])

# Invert latency (lower is better)
dragon_scores[3] = 1 - (1 - dragon_scores[3]) * 0.5  # Less penalty for DRAGON
pppo_scores[3] = 1 - (1 - pppo_scores[3]) * 0.7
cddpg_scores[3] = 1 - (1 - cddpg_scores[3]) * 0.8
mpt_scores[3] = 1 - (1 - mpt_scores[3]) * 0.9

# Set angles for each metric (evenly spaced)
angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

# Complete the loops for plotting
dragon_scores = np.append(dragon_scores, dragon_scores[0])
pppo_scores = np.append(pppo_scores, pppo_scores[0])
cddpg_scores = np.append(cddpg_scores, cddpg_scores[0])
mpt_scores = np.append(mpt_scores, mpt_scores[0])

# Plot radar charts
ax3.plot(angles, dragon_scores, '-', linewidth=3, color=dragon_color, label='DRAGON')
ax3.plot(angles, pppo_scores, '--', linewidth=2, color=pppo_color, label='P-PPO')
ax3.plot(angles, cddpg_scores, '-.', linewidth=2, color=cddpg_color, label='C-DDPG')
ax3.plot(angles, mpt_scores, ':', linewidth=2, color=mpt_color, label='MPT')

# Fill the radar charts with semi-transparent colors
ax3.fill(angles, dragon_scores, color=dragon_color, alpha=0.1)
ax3.fill(angles, pppo_scores, color=pppo_color, alpha=0.05)

# Set labels at appropriate angles
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics)

# Set y-ticks (circles) to be invisible
ax3.set_yticks([])
# But add gridlines
ax3.grid(True, linestyle='--', alpha=0.3)

# Draw axis lines from center to each metric
for angle, label in zip(angles[:-1], metrics):
    ax3.plot([angle, angle], [0, 1.1], color='grey', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add percentage labels near the DRAGON points (excluding the duplicated last point)
    if angle != angles[-1]:  # Skip the duplicated last point
        idx = list(angles).index(angle)
        radial_pos = dragon_scores[idx] + 0.05  # Position slightly beyond the data point
        percent = int(dragon_scores[idx] * 100)
        ax3.text(angle, radial_pos, f"{percent}%", 
                color=dragon_color, fontsize=9, fontweight='bold',
                ha='center', va='center')

# Add title
ax3.set_title('Multi-Metric Performance Radar', fontweight='bold', pad=15)

# Add legend
legend3 = ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)
frame3 = legend3.get_frame()
frame3.set_facecolor('white')

#=================================================
# BOTTOM RIGHT: IoT Device Admission Rate
#=================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#f8f8f8')

# Generate time steps
time_steps = np.linspace(0, 100, 100)

# Admission rate values
dragon_ar = np.zeros_like(time_steps)
pppo_ar = np.zeros_like(time_steps)
cddpg_ar = np.zeros_like(time_steps)

# Set values based on time periods
# Initial phase (0-20)
dragon_ar[:20] = 45
pppo_ar[:20] = 35
cddpg_ar[:20] = 25

# Transition phase (20-40)
dragon_ar[20:40] = np.linspace(45, 85, 20)
pppo_ar[20:40] = np.linspace(35, 65, 20)
cddpg_ar[20:40] = np.linspace(25, 50, 20)

# Stable phase (40-100)
dragon_ar[40:] = 85
pppo_ar[40:] = 65
cddpg_ar[40:] = 50

# Plot admission rate curves
dragon_line = ax4.plot(time_steps, dragon_ar, '-', color=dragon_color, linewidth=3, label='DRAGON')[0]
pppo_line = ax4.plot(time_steps, pppo_ar, '--', color=pppo_color, linewidth=2, label='P-PPO')[0]
cddpg_line = ax4.plot(time_steps, cddpg_ar, '-.', color=cddpg_color, linewidth=2, label='C-DDPG')[0]

# Set labels and limits
ax4.set_xlabel('Training Epochs', fontweight='bold')
ax4.set_ylabel('Admission Rate (%)', fontweight='bold')
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)

# Add grid
ax4.grid(True, linestyle='--', alpha=0.3)

# Add title
ax4.set_title('IoT Device Admission Rate', fontweight='bold', pad=10)

# Adjust spacing between subplots
plt.tight_layout(pad=3.0)

# Save the figure
plt.savefig('dragon_comprehensive_metrics.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()