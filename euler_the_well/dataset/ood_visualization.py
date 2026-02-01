"""
NOTE: in the final report plot some test points have been removed for clarity but are here kept for completeness.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# build dataframe (values from means computations)
data = [
    ("Train", 1.22, "C2H6_15", 0.92404, 0.85714, 4.09723, 0.51284),
    ("Valid", 1.22, "C2H6_15", 0.93030, 1.01129, 4.83320, 0.56921),
    ("Test", 1.22, "C2H6_15", 0.92280, 0.98231, 4.66132, 0.49693),
    ("Train", 1.33, "H2O_20", 0.90625, 0.93171, 3.02846, 0.51138),
    ("Valid", 1.33, "H2O_20", 0.87018, 0.81486, 2.65788, 0.47462),
    ("Test", 1.33, "H2O_20", 0.91170, 0.91914, 2.98148, 0.50409),
    ("Train", 1.4, "Dry_air_20", 0.92992, 0.90027, 2.45969, 0.52674),
    ("Valid", 1.4, "Dry_air_20", 0.91331, 0.87083, 2.37108, 0.49017),
    ("Test", 1.4, "Dry_air_20", 0.84708, 0.75516, 2.08584, 0.48677),
    ("Test", 1.365, "Dry_air_1000", 0.90582, 0.93289, 2.76202, 0.51478),
    ("Test", 1.3, "CO2_20", 0.95501, 0.87964, 3.15166, 0.55149),
    ("Test", 1.404, "H2_100_Dry_air_-15", 0.94490, 0.97320, 2.62923, 0.55289),
    ("Test", 1.453, "H2_-76", 0.91525, 0.90739, 2.18149, 0.46837),
    ("Test", 1.13, "C3H8_16", 0.90222, 0.87668, 6.92274, 0.47338),
    ("Test", 1.597, "H2_-181", 0.96720, 1.08933, 2.04747, 0.55668),
    ("Test", 1.76, "Ar_-180", 0.90955, 1.00426, 1.52157, 0.50580),
]

df = pd.DataFrame(data, columns=["Split", "Gamma", "Dataset", "MeanDensity", "MeanPressure", "MeanEnergy", "MeanMomentum"])

# multivariate OOD with graded levels
train_gammas = [1.22, 1.33, 1.4]
train_df = df[df['Gamma'].isin(train_gammas)]
X_train = train_df[['Gamma','MeanEnergy','MeanMomentum']].values

mins = X_train.min(axis=0)
maxs = X_train.max(axis=0)
ranges = maxs - mins

def graded_ood(row):
    x = np.array([row['Gamma'], row['MeanEnergy'], row['MeanMomentum']])
    outside = np.maximum(mins - x, 0) + np.maximum(x - maxs, 0)
    max_out = np.max(outside / ranges)
    if row['Dataset'] in ["C2H6_15","H2O_20","Dry_air_20"]:
        return 'Train+Assoc'  # main training group + corresponding valid/test
    elif np.all(x >= mins) and np.all(x <= maxs):
        return 'In-Distribution'
    elif max_out <= 0.1:
        return 'OOD Mild'
    elif max_out <= 0.3:
        return 'OOD Moderate'
    else:
        return 'OOD Strong'

df['OODLevel_Graded'] = df.apply(graded_ood, axis=1)

# 3D scatter plot
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection='3d')

colors = {
    'Train+Assoc':'lightgrey',
    'In-Distribution':'green',
    'OOD Mild':'yellow',
    'OOD Moderate':'orange',
    'OOD Strong':'red'
}

marker_scale = 400
df = df.reset_index(drop=True)
df['PointID'] = df.index + 1

description_lines = []

for lvl in colors:
    subset = df[df['OODLevel_Graded']==lvl]
    for idx, row in subset.iterrows():
        edge_w = 3 if row['Split']=='Train' else 0.5
        ax.scatter(row['Gamma'], row['MeanEnergy'], row['MeanMomentum'],
                   s=row['MeanDensity']*marker_scale,
                   c=colors[lvl], edgecolor='k', linewidth=edge_w, alpha=0.6)
        ax.text(row['Gamma'], row['MeanEnergy'], row['MeanMomentum'], str(row['PointID']),
                color='black', fontsize=12, horizontalalignment='center', verticalalignment='center')
        description_lines.append(f"{row['PointID']}: {row['Dataset']} - {row['Split']}")

# legend
legend_elements = [
    Line2D([0],[0], marker='o', color='w', label='Train + Valid/Test', markerfacecolor='lightgrey', markersize=10),
    Line2D([0],[0], marker='o', color='w', label='In-Distribution', markerfacecolor='green', markersize=10),
    Line2D([0],[0], marker='o', color='w', label='OOD Mild', markerfacecolor='yellow', markersize=10),
    Line2D([0],[0], marker='o', color='w', label='OOD Moderate', markerfacecolor='orange', markersize=10),
    Line2D([0],[0], marker='o', color='w', label='OOD Strong', markerfacecolor='red', markersize=10),
    Line2D([0],[0], marker='o', color='k', label='Train Points', markerfacecolor='w', markersize=10, linewidth=2)
]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_xlabel('Gamma')
ax.set_ylabel('Mean Energy')
ax.set_zlabel('Mean Momentum Magnitude')
ax.set_title('3D Dataset Overview: Gamma vs Energy vs Momentum\n(Marker size ~ Density)')

# description box
description_text = "\n".join(description_lines)
fig.text(0.75, 0.1, description_text, fontsize=9, va='bottom', ha='left', wrap=True)

plt.show()
