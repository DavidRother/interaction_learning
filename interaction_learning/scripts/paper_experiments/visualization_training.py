import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle


plt.style.use("seaborn-whitegrid")


x_tasks = ["a", "b", "c", "d", "e"]
y_tasks = ["0", "1", "2", "3", "4"]
tasks = ["t" + x + y for x in x_tasks for y in y_tasks] + ["t" + x for x in x_tasks] + ["t" + y for y in y_tasks]
impact_tasks = ["i" + x + y for x in x_tasks for y in y_tasks] + ["i" + x for x in x_tasks] + ["i" + y for y in y_tasks]

with open("stats/training_impact_learners.stats", 'rb') as outp:  # Overwrites any existing file.
    obj = pickle.load(outp)

num_steps = 200

mean_train = []
std_train = []

for idx in range(num_steps):
    nums = []
    for task in tasks:
        nums.append(obj["ep_rews"][task][idx])
    nums = np.asarray(nums)
    mean_train.append(nums.mean())
    std_train.append(nums.std())

mean_train = np.asarray(mean_train)
std_train = np.asarray(std_train)

length_train = np.arange(0, len(mean_train))

plt.subplots(1, figsize=(10, 10))
plt.plot(length_train, mean_train, label="Episode Mean Score")

plt.fill_between(length_train, mean_train - std_train, mean_train + std_train, alpha=0.5, linewidth=1.5)

plt.title("Training Curve Single Tasks")
plt.xlabel("Episode #"), plt.ylabel("Reward"), plt.legend(loc="best")
plt.tight_layout()

plt.show()

print("done")
