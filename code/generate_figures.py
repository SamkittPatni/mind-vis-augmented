import matplotlib.pyplot as plt
import numpy as np

# === Metric Data (as extracted/approximated) ===
methods = ['Ours', 'Ozcelik et al.', 'Gaziv et al.', 'Beliy et al.']
colors = ['#e7a77e', '#c8503e', '#7d2c6a', '#2f1c4d']

top1_50 = [0.222, 0.20, 0.055, 0.034]
top1_100 = [0.152, 0.128, 0.035, 0.019]

top5_50 = [0.489, 0.525, 0.236, 0.106]
top5_100 = [0.381, 0.377, 0.097, 0.066]

fid = [1.67, 2.36, 10.1, 24.3]

# === Setup for plotting ===
bar_width = 0.18
num_methods = len(methods)
offsets = np.linspace(-1.5, 1.5, num_methods) * bar_width

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# === Top-1 Accuracy ===
x1 = np.arange(2)
top1_data = [top1_50, top1_100]
for i in range(num_methods):
    axes[0].bar(x1 + offsets[i], [d[i] for d in top1_data], width=bar_width, color=colors[i], label=methods[i])
axes[0].axhline(1/50, linestyle='--', color='black', linewidth=1)
axes[0].set_xticks(x1)
axes[0].set_xticklabels(['50-way', '100-way'])
axes[0].set_ylabel("Top-1 Identification Accuracy")
axes[0].set_ylim(0, 0.3)
axes[0].legend(title='', frameon=False)

# === Top-5 Accuracy ===
x2 = np.arange(2)
top5_data = [top5_50, top5_100]
for i in range(num_methods):
    axes[1].bar(x2 + offsets[i], [d[i] for d in top5_data], width=bar_width, color=colors[i])
axes[1].axhline(5/50, linestyle='--', color='black', linewidth=1)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(['50-way', '100-way'])
axes[1].set_ylabel("Top-5 Identification Accuracy")
axes[1].set_ylim(0, 0.6)

# === FID Scores ===
x3 = np.arange(1)
for i in range(num_methods):
    axes[2].bar(x3 + offsets[i], [fid[i]], width=bar_width, color=colors[i])
axes[2].set_xticks([0])
axes[2].set_xticklabels(['Methods'])
axes[2].set_ylabel("Fréchet Inception Distance")
axes[2].set_ylim(0, 25)

# Final layout
plt.tight_layout()
plt.show()
