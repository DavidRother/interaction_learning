import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from time import sleep
import numpy as np
import pickle


plt.style.use("seaborn-whitegrid")

SMALL_SIZE = 30
MEDIUM_SIZE = 35
BIGGER_SIZE = 35

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
impact_tasks = ["i" + x + y for x in x_tasks for y in y_tasks] + ["i" + x for x in x_tasks] + ["i" + y for y in y_tasks]

with open("stats/training_impact_learners2_longer_train.stats", 'rb') as outp:  # Overwrites any existing file.
    obj = pickle.load(outp)

num_steps = 500

mean_train = []
std_train = []

for idx in range(num_steps):
    nums = []
    for task in tasks:
        nums.append(obj["ep_rews"][task][idx])
    nums = np.asarray(nums)
    mean_train.append(nums.mean())
    std_train.append(nums.std())

mean_train = savgol_filter(np.asarray(mean_train), 20, 3)
std_train = savgol_filter(np.asarray(std_train), 500, 3)

length_train = np.arange(0, len(mean_train))

plt.subplots(1, figsize=(10, 10))
plt.plot(length_train, mean_train, label="Episode Mean Score")

plt.fill_between(length_train, mean_train - std_train, mean_train + std_train, alpha=0.5, linewidth=1.5)

plt.title("Interaction Training")
plt.xlabel("Episode #")
plt.ylabel("Reward")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("plots/training_interaction_tasks.svg")

plt.show()

print("done")
