#!/usr/bin/env python3
"""
Plot the Pareto trade-off curve for MODPO verification step.
Uses results from outputs/eval_paper/results.csv.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from eval_paper/results.csv
# Higher reward = more helpful, Higher cost = more harmful (less safe)
data = {
    'sft_baseline': {'w': None, 'reward': -0.5415, 'cost': 6.0702},
    'modpo_w0.0':   {'w': 0.0,  'reward': 1.3471, 'cost': 0.9443},
    'modpo_w0.2':   {'w': 0.2,  'reward': 2.0191, 'cost': 2.0566},
    'modpo_w0.4':   {'w': 0.4,  'reward': 2.7858, 'cost': 5.1886},
    'modpo_w0.5':   {'w': 0.5,  'reward': 3.0075, 'cost': 5.5029},
    'modpo_w0.6':   {'w': 0.6,  'reward': 3.0112, 'cost': 6.3586},
    'modpo_w0.8':   {'w': 0.8,  'reward': 4.2088, 'cost': 7.8173},
    'modpo_w1.0':   {'w': 1.0,  'reward': 5.1785, 'cost': 8.8103},
}

# Extract MODPO points (sorted by weight)
modpo_keys = [k for k in data.keys() if k.startswith('modpo')]
modpo_keys.sort(key=lambda k: data[k]['w'])

weights = [data[k]['w'] for k in modpo_keys]
rewards = [data[k]['reward'] for k in modpo_keys]
costs = [data[k]['cost'] for k in modpo_keys]

# For Pareto plot: x = negative cost (harmlessness), y = reward (helpfulness)
# Higher harmlessness = lower cost = better safety
harmlessness = [-c for c in costs]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left plot: Pareto trade-off curve (Helpfulness vs Harmlessness)
ax1 = axes[0]
ax1.plot(harmlessness, rewards, 'o-', color='#2563eb', linewidth=2, markersize=10, label='MODPO')

# Add weight annotations
for i, (h, r, w) in enumerate(zip(harmlessness, rewards, weights)):
    offset = (8, 8) if i % 2 == 0 else (-8, -12)
    ax1.annotate(f'w={w}', (h, r), textcoords='offset points', xytext=offset,
                fontsize=9, color='#1e40af')

# Add SFT baseline
sft_h = -data['sft_baseline']['cost']
sft_r = data['sft_baseline']['reward']
ax1.scatter([sft_h], [sft_r], color='#dc2626', s=120, marker='s', zorder=5, label='SFT Baseline')
ax1.annotate('SFT', (sft_h, sft_r), textcoords='offset points', xytext=(8, -12),
            fontsize=9, color='#991b1b', fontweight='bold')

ax1.set_xlabel('Harmlessness (negative cost)', fontsize=11)
ax1.set_ylabel('Helpfulness (reward)', fontsize=11)
ax1.set_title('MODPO Pareto Trade-off: Helpfulness vs Safety', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Add arrow showing trade-off direction
ax1.annotate('', xy=(-2, 4.5), xytext=(-7, 1.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax1.text(-5.5, 3.5, 'Trade-off\ndirection', fontsize=9, color='gray', ha='center')

# Right plot: Reward and Cost vs Weight
ax2 = axes[1]
ax2.plot(weights, rewards, 'o-', color='#16a34a', linewidth=2, markersize=8, label='Helpfulness (Reward)')
ax2.plot(weights, costs, 's-', color='#dc2626', linewidth=2, markersize=8, label='Harmfulness (Cost)')

# Add horizontal lines for SFT baseline
ax2.axhline(y=data['sft_baseline']['reward'], color='#16a34a', linestyle='--', alpha=0.5, label='SFT Reward')
ax2.axhline(y=data['sft_baseline']['cost'], color='#dc2626', linestyle='--', alpha=0.5, label='SFT Cost')

ax2.set_xlabel('Weight (w)', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Reward/Cost vs MODPO Weight', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 1.05)

# Add interpretation
ax2.text(0.5, -1.5, 'w=0: prioritize safety | w=1: prioritize helpfulness',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('outputs/eval_paper/pareto_tradeoff.png', dpi=150, bbox_inches='tight')
plt.savefig('../../pareto_tradeoff.png', dpi=150, bbox_inches='tight')  # Also save to thesis root
print("Saved: outputs/eval_paper/pareto_tradeoff.png")
print("Saved: ../../pareto_tradeoff.png (thesis root)")
plt.show()
