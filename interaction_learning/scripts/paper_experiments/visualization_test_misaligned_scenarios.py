import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
import numpy as np
import pickle

with open("stats/test_misaligned_scenarios.stats", 'rb') as outp:  # Overwrites any existing file.
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

aligned_task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
impact_task_1 = ["ib3", "ic4", "ia0", "id2", "ie1"]
aligned_task_2 = ["tb3", "tc4", "ta0", "td2", "te1"]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

num_evals = 10

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
# ax.set_ylabel('Negative Rewards', fontsize=30, fontname=fontname)
# ax.set_title('Accumulated Negative Rewards Misaligned Scenarios')
ax.set_xticks(x, labels, fontsize=30, fontname=fontname)
plt.yticks(fontsize=12)
ax.legend(fontsize=18)

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()

print("done")
