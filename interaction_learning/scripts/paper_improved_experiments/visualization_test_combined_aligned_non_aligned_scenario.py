import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def get_aligned_stats():

    with open("stats/test_aligned_scenarios_after_submit9.stats", 'rb') as outp:  # Overwrites any existing file.
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

    aligned_task_1 = ["tc"]
    impact_task_1 = ["ic4"]
    aligned_task_2 = ["tc4"]

    algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
                  "selfish_task_solver", "joint_learner"]

    num_evals = 200
    root_eval = np.sqrt(num_evals)

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

    std_aail_team = np.asarray(mean_aligned_interaction_learner_team).std(ddof=1) / root_eval
    std_aail_1 = np.asarray(mean_aligned_interaction_learner_1).std(ddof=1) / root_eval
    std_aail_2 = np.asarray(mean_aligned_interaction_learner_2).std(ddof=1) / root_eval

    mean_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).mean()
    mean_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).mean()
    mean_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).mean()

    std_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).std(ddof=1) / root_eval
    std_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).std(ddof=1) / root_eval
    std_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).std(ddof=1) / root_eval

    mean_jl_team = np.asarray(mean_joint_learner_team).mean()
    mean_jl_1 = np.asarray(mean_joint_learner_1).mean()
    mean_jl_2 = np.asarray(mean_joint_learner_2).mean()

    std_jl_team = np.asarray(mean_joint_learner_team).std(ddof=1) / root_eval
    std_jl_1 = np.asarray(mean_joint_learner_1).std(ddof=1) / root_eval
    std_jl_2 = np.asarray(mean_joint_learner_2).std(ddof=1) / root_eval

    mean_sl_team = np.asarray(mean_selfish_task_solver_team).mean()
    mean_sl_1 = np.asarray(mean_selfish_task_solver_1).mean()
    mean_sl_2 = np.asarray(mean_selfish_task_solver_2).mean()

    std_sl_team = np.asarray(mean_selfish_task_solver_team).std(ddof=1) / root_eval
    std_sl_1 = np.asarray(mean_selfish_task_solver_1).std(ddof=1) / root_eval
    std_sl_2 = np.asarray(mean_selfish_task_solver_2).std(ddof=1) / root_eval

    labels = ["AAIL", "NAIL", "SL", "JL"]

    means_team = [mean_aail_team, mean_nail_team, mean_jl_team, mean_sl_team]
    means_1 = [mean_aail_1, mean_nail_1, mean_jl_1, mean_sl_1]
    means_2 = [mean_aail_2, mean_nail_2, mean_jl_2, mean_sl_2]

    base = min(means_team + means_1 + means_2) - 50

    means_team = np.asarray([abs(mean_aail_team - base), abs(mean_nail_team - base), abs(mean_jl_team - base), abs(mean_sl_team - base)])
    means_1 = np.asarray([abs(mean_aail_1 - base), abs(mean_nail_1 - base), abs(mean_jl_1 - base), abs(mean_sl_1 - base)])
    means_2 = np.asarray([abs(mean_aail_2 - base), abs(mean_nail_2 - base), abs(mean_jl_2 - base), abs(mean_sl_2 - base)])

    std_team = [std_aail_team, std_nail_team, std_jl_team, std_sl_team]
    std_1 = [std_aail_1, std_nail_1, std_jl_1, std_sl_1]
    std_2 = [std_aail_2, std_nail_2, std_jl_2, std_sl_2]

    return means_team, means_1, means_2, std_team, std_1, std_2


def get_non_aligned_stats():
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

    std_aligned_interaction_learner = []
    std_non_aligned_interaction_learner = []
    std_selfish_task_solver = []
    std_joint_learner = []

    aligned_task_1 = ["ta2"]
    impact_task_1 = ["ib3"]
    aligned_task_2 = ["tb3"]

    algorithms = ["action_aligned_interaction_learner", "non_aligned_interaction_learner",
                  "selfish_task_solver", "joint_learner"]

    num_evals = 200
    root_eval = np.sqrt(num_evals)

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

    std_aail_team = np.asarray(mean_aligned_interaction_learner_team).std(ddof=1) / root_eval
    std_aail_1 = np.asarray(mean_aligned_interaction_learner_1).std(ddof=1) / root_eval
    std_aail_2 = np.asarray(mean_aligned_interaction_learner_2).std(ddof=1) / root_eval

    mean_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).mean()
    mean_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).mean()
    mean_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).mean()

    std_nail_team = np.asarray(mean_non_aligned_interaction_learner_team).std(ddof=1) / root_eval
    std_nail_1 = np.asarray(mean_non_aligned_interaction_learner_1).std(ddof=1) / root_eval
    std_nail_2 = np.asarray(mean_non_aligned_interaction_learner_2).std(ddof=1) / root_eval

    mean_jl_team = np.asarray(mean_joint_learner_team).mean()
    mean_jl_1 = np.asarray(mean_joint_learner_1).mean()
    mean_jl_2 = np.asarray(mean_joint_learner_2).mean()

    std_jl_team = np.asarray(mean_joint_learner_team).std(ddof=1) / root_eval
    std_jl_1 = np.asarray(mean_joint_learner_1).std(ddof=1) / root_eval
    std_jl_2 = np.asarray(mean_joint_learner_2).std(ddof=1) / root_eval

    mean_sl_team = np.asarray(mean_selfish_task_solver_team).mean()
    mean_sl_1 = np.asarray(mean_selfish_task_solver_1).mean()
    mean_sl_2 = np.asarray(mean_selfish_task_solver_2).mean()

    std_sl_team = np.asarray(mean_selfish_task_solver_team).std(ddof=1) / root_eval
    std_sl_1 = np.asarray(mean_selfish_task_solver_1).std(ddof=1) / root_eval
    std_sl_2 = np.asarray(mean_selfish_task_solver_2).std(ddof=1) / root_eval

    labels = ["AAIL", "NAIL", "SL", "JL"]

    means_team = [mean_aail_team, mean_nail_team, mean_jl_team, mean_sl_team]
    means_1 = [mean_aail_1, mean_nail_1, mean_jl_1, mean_sl_1]
    means_2 = [mean_aail_2, mean_nail_2, mean_jl_2, mean_sl_2]

    base = min(means_team + means_1 + means_2) - 50

    means_team = np.asarray(
        [abs(mean_aail_team - base), abs(mean_nail_team - base), abs(mean_jl_team - base), abs(mean_sl_team - base)])
    means_1 = np.asarray(
        [abs(mean_aail_1 - base), abs(mean_nail_1 - base), abs(mean_jl_1 - base), abs(mean_sl_1 - base)])
    means_2 = np.asarray(
        [abs(mean_aail_2 - base), abs(mean_nail_2 - base), abs(mean_jl_2 - base), abs(mean_sl_2 - base)])

    std_team = [std_aail_team, std_nail_team, std_jl_team, std_sl_team]
    std_1 = [std_aail_1, std_nail_1, std_jl_1, std_sl_1]
    std_2 = [std_aail_2, std_nail_2, std_jl_2, std_sl_2]

    return means_team, means_1, means_2, std_team, std_1, std_2


aligned_means_team, aligned_means_1, aligned_means_2, aligned_std_team, aligned_std_1, aligned_std_2 = get_aligned_stats()
non_aligned_means_team, non_aligned_means_1, non_aligned_means_2, non_aligned_std_team, non_aligned_std_1, non_aligned_std_2 = get_non_aligned_stats()

labels = ["AAIL", "NAIL", "SL", "JL"]

fontsize = 11
fontname = "Arial"

aligned_base = min(aligned_means_team + aligned_means_1 + aligned_means_2) - 50
non_aligned_base = min(non_aligned_means_team + non_aligned_means_1 + non_aligned_means_2) - 50

plt.style.use("seaborn-whitegrid")

ax = plt.subplot(1, 2, 1)
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
rects1 = ax.bar(x - width, aligned_means_team, width, label='Team', bottom=aligned_base)
rects2 = ax.bar(x, aligned_means_1, width, label='Agent 1', bottom=aligned_base)
rects3 = ax.bar(x + width, aligned_means_2, width, label='Agent 2', bottom=aligned_base)

plt.errorbar(x - width, aligned_means_team + aligned_base, yerr=aligned_std_team, fmt="none", color="k")
plt.errorbar(x, aligned_means_1 + aligned_base, yerr=aligned_std_1, fmt="none", color="k")
plt.errorbar(x + width, aligned_means_2 + aligned_base, yerr=aligned_std_2, fmt="none", color="k")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Rewards', fontsize=30, fontname=fontname)
# ax.set_title('Accumulated Negative Rewards Misaligned Scenarios')
ax.set_xticks(x, labels, fontsize=30, fontname=fontname)
ax.legend(fontsize=23, loc='upper center', ncol=3, fancybox=True, shadow=True, markerscale=0.7,
          bbox_to_anchor=(0.5, 1.15), frameon=True, labelspacing=0.2, columnspacing=0.8, handletextpad=0.2)

ax = plt.subplot(1, 2, 2)

non_aligned_rects1 = ax.bar(x - width, non_aligned_means_team, width, label='Team', bottom=non_aligned_base)
non_aligned_rects2 = ax.bar(x, non_aligned_means_1, width, label='Agent 1', bottom=non_aligned_base)
non_aligned_rects3 = ax.bar(x + width, non_aligned_means_2, width, label='Agent 2', bottom=non_aligned_base)

plt.errorbar(x - width, non_aligned_means_team + non_aligned_base, yerr=non_aligned_std_team, fmt="none", color="k")
plt.errorbar(x, non_aligned_means_1 + non_aligned_base, yerr=non_aligned_std_1, fmt="none", color="k")
plt.errorbar(x + width, non_aligned_means_2 + non_aligned_base, yerr=non_aligned_std_2, fmt="none", color="k")

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Rewards', fontsize=30, fontname=fontname)
# ax.set_title('Accumulated Negative Rewards Misaligned Scenarios')
ax.set_xticks(x, labels, fontsize=30, fontname=fontname)
ax.legend(fontsize=23, loc='upper center', ncol=3, fancybox=True, shadow=True, markerscale=0.7,
          bbox_to_anchor=(0.5, 1.15), frameon=True, labelspacing=0.2, columnspacing=0.8, handletextpad=0.2)


plt.show()
