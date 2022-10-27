import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle

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

with open("stats/test_aligned_scenarios.stats", 'rb') as outp:  # Overwrites any existing file.
    obj = pickle.load(outp)

mean_aligned_interaction_learner_team = []
mean_aligned_interaction_learner_1 = []
mean_aligned_interaction_learner_2 = []
mean_non_aligned_interaction_learner_team = []
mean_non_aligned_interaction_learner_1 = []
mean_non_aligned_interaction_learner_2 = []
mean_selfish_task_solver_team = []
mean_selfish_task_solver_1 = []
mean_selfish_task_solver_2 = []
mean_joint_learner_team = []
mean_joint_learner_1 = []
mean_joint_learner_2 = []

std_aligned_interaction_learner = []
std_non_aligned_interaction_learner = []
std_selfish_task_solver = []
std_joint_learner = []

aligned_task_1 = ["ta", "te", "tb", "tc", "td"]
impact_task_1 = ["ie0", "ia1", "id2", "ic4", "ib3"]
aligned_task_2 = ["te0", "ta1", "td2", "tc4", "tb3"]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

num_evals = 100

eval_scores = obj["eval_scores"]

for t1, i1, t2 in zip(aligned_task_1, impact_task_1, aligned_task_2):
    id_string = t1 + i1 + t2
    for idx in range(num_evals):
        mean_aligned_interaction_learner_1.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_0"])
        mean_aligned_interaction_learner_2.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_1"])
        mean_aligned_interaction_learner_team.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_0"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_1"])

        mean_non_aligned_interaction_learner_1.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_0"])
        mean_non_aligned_interaction_learner_2.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_1"])
        mean_non_aligned_interaction_learner_team.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_0"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_1"])

        mean_selfish_task_solver_1.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_0"])
        mean_selfish_task_solver_2.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_1"])
        mean_selfish_task_solver_team.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_0"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_1"])

        mean_joint_learner_1.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_0"])
        mean_joint_learner_2.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_1"])
        mean_joint_learner_team.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_0"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_1"])

mean_aail_team = np.asarray(mean_aligned_interaction_learner_team).mean()
mean_aail_1 = np.asarray(mean_aligned_interaction_learner_1).mean()
mean_aail_2 = np.asarray(mean_aligned_interaction_learner_2).mean()

std_aail_team = np.asarray(mean_aligned_interaction_learner_team).std()
std_aail_1 = np.asarray(mean_aligned_interaction_learner_1).std()
std_aail_2 = np.asarray(mean_aligned_interaction_learner_2).std()

mean_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).mean()
mean_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).mean()
mean_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).mean()

std_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).std()
std_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).std()
std_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).std()

mean_jl_team = np.asarray(mean_joint_learner_team).mean()
mean_jl_1 = np.asarray(mean_joint_learner_1).mean()
mean_jl_2 = np.asarray(mean_joint_learner_2).mean()

std_jl_team = np.asarray(mean_joint_learner_team).std()
std_jl_1 = np.asarray(mean_joint_learner_1).std()
std_jl_2 = np.asarray(mean_joint_learner_2).std()

mean_sl_team = np.asarray(mean_selfish_task_solver_team).mean()
mean_sl_1 = np.asarray(mean_selfish_task_solver_1).mean()
mean_sl_2 = np.asarray(mean_selfish_task_solver_2).mean()

std_sl_team = np.asarray(mean_selfish_task_solver_team).std()
std_sl_1 = np.asarray(mean_selfish_task_solver_1).std()
std_sl_2 = np.asarray(mean_selfish_task_solver_2).std()

labels = ["AAIL", "NAIL", "SL", "JL"]

means_team = [-mean_aail_team, -mean_nail_team, -mean_jl_team, -mean_sl_team]
means_1 = [-mean_aail_1, -mean_nail_1, -mean_jl_1, -mean_sl_1]
means_2 = [-mean_aail_2, -mean_nail_2, -mean_jl_2, -mean_sl_2]

std_team = [std_aail_team, std_nail_team, std_jl_team, std_sl_team]
std_1 = [std_aail_1, std_nail_1, std_jl_1, std_sl_1]
std_2 = [std_aail_2, std_nail_2, std_jl_2, std_sl_2]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fontsize = 11
fontname = "Arial"

plt.style.use("seaborn-whitegrid")

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, means_team, width, label='Team')
rects2 = ax.bar(x, means_1, width, label='Agent 1')
rects3 = ax.bar(x + width, means_2, width, label='Agent 2')

plt.errorbar(x - width, means_team, yerr=std_team, fmt="o", color="k")
plt.errorbar(x, means_1, yerr=std_1, fmt="o", color="k")
plt.errorbar(x + width, means_2, yerr=std_2, fmt="o", color="k")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Negative Rewards', fontsize=30, fontname=fontname)
# ax.set_title('Accumulated Negative Rewards Misaligned Scenarios')
ax.set_xticks(x, labels, fontsize=30, fontname=fontname)
ax.legend(fontsize=23, loc="upper left")

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

fig.tight_layout()
plt.savefig("plots/test_aligned_scenarios.svg")
plt.show()

print("done")
