from scipy.stats import ttest_rel, ttest_ind
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle
from scipy.stats import shapiro
from scipy.stats import lognorm
from scipy.stats import wilcoxon
import math


np.random.seed(1)

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

aail_data = np.asarray(mean_aligned_interaction_learner_team)
nail_data = np.asarray(mean_non_aligned_interaction_learner_team)
sl_data = np.asarray(mean_selfish_task_solver_team)
jl_data = np.asarray(mean_joint_learner_team)

stat, p = ttest_rel(aail_data, nail_data)
stat2, p2 = ttest_rel(aail_data, sl_data)
stat3, p3 = ttest_rel(aail_data, jl_data)

print(f"AAIL is mean different than NAIL with {p} confidence")
print(f"AAIL is mean different than SL with {p2} confidence")
print(f"AAIL is mean different than JL with {p3} confidence")

res = shapiro(aail_data)
res2 = shapiro(nail_data)
res3 = shapiro(sl_data)
res4 = shapiro(jl_data)

print(f"AAIL is normal distributed: {res}")
print(f"NAIL is normal distributed: {res2}")
print(f"sL is normal distributed: {res3}")
print(f"JL is normal distributed: {res4}")

stat4, p4 = wilcoxon(aail_data, nail_data)
stat5, p5 = wilcoxon(aail_data, sl_data)
stat6, p6 = wilcoxon(aail_data, jl_data)

print(f"AAIL is mean different than NAIL with {p4} confidence")
print(f"AAIL is mean different than SL with {p5} confidence")
print(f"AAIL is mean different than JL with {p6} confidence")

print("done")
