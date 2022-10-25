import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle

with open("stats/test_aligned_scenarios_many_agents.stats", 'rb') as outp:  # Overwrites any existing file.
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

task_1 = ["ta2", "te1", "tb2", "tc4", "td3"]
task_2 = ["ta0", "ta1", "ta2", "ta3", "ta4"]
task_3 = ["tb2", "tb3", "tb4", "tb0", "tb1"]
task_4 = ["tc4", "tc0", "tc1", "tc2", "tc3"]
task_5 = ["te1", "td2", "te3", "td4", "te0"]

impact_tasks_1 = [[t1.replace("t", "i"), t2.replace("t", "i"),
                   t3.replace("t", "i"), t4.replace("t", "i")]
                  for t1, t2, t3, t4 in zip(task_2, task_3, task_4, task_5)]

algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
              "selfish_task_solver", "joint_learner"]

num_evals = 100

eval_scores = obj["eval_scores"]

for t1, t2, t3, t4, t5, i1 in zip(task_1, task_2, task_3, task_4, task_5, impact_tasks_1):
    id_string = t1 + ''.join(i1) + t2 + t3 + t4 + t5
    for idx in range(num_evals):
        mean_aligned_interaction_learner_1.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_0"])
        mean_aligned_interaction_learner_2.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_1"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_2"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_3"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_4"])
        mean_aligned_interaction_learner_team.append(
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_0"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_1"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_2"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_3"] +
            eval_scores["action_aligned_interaction_learner"][id_string][0][idx]["player_4"])

        mean_non_aligned_interaction_learner_1.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_0"])
        mean_non_aligned_interaction_learner_2.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_1"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_2"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_3"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_4"])
        mean_non_aligned_interaction_learner_team.append(
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_0"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_1"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_2"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_3"] +
            eval_scores["non_aligned_interaction_learner"][id_string][0][idx]["player_4"])

        mean_selfish_task_solver_1.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_0"])
        mean_selfish_task_solver_2.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_1"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_2"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_3"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_4"])
        mean_selfish_task_solver_team.append(
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_0"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_1"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_2"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_3"] +
            eval_scores["selfish_task_solver"][id_string][0][idx]["player_4"])

        mean_joint_learner_1.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_0"])
        mean_joint_learner_2.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_1"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_2"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_3"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_4"])
        mean_joint_learner_team.append(
            eval_scores["joint_learner"][id_string][0][idx]["player_0"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_1"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_2"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_3"] +
            eval_scores["joint_learner"][id_string][0][idx]["player_4"])

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
plt.yticks(fontsize=12)
ax.legend(fontsize=18)

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()

print("done")
