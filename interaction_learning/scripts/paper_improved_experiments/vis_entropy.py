import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle


fontsize = 11
fontname = "Arial"

plt.style.use("seaborn-whitegrid")
labels = [f"a{i}" for i in range(1,6)]
dist_1 = [0.05, 0.1, 0.025, 0.8, 0.025]
width = 0.35

x = np.arange(len(dist_1))

fig, ax = plt.subplots()
rects1 = ax.bar(x, dist_1, width, label='Team')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability', fontsize=30, fontname=fontname)
# ax.set_title('Accumulated Negative Rewards Misaligned Scenarios')
ax.set_xticks(x, labels, fontsize=30, fontname=fontname)
ax.set_ylim([0, 1])

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

fig.tight_layout()
plt.savefig("plots/dist2.svg")
plt.show()

