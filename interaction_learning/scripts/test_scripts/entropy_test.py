import numpy as np

act_dist = [0.3, 0.15, 0.05, 0.35, 0.2]
act_dist2 = [0.5, 0.1, 0.25, 0.05, 0.1]
act_dist3 = [0.7, 0.1, 0.05, 0.05, 0.1]
act_dist4 = [0.7, 0.05, 0.05, 0.15, 0.05]
act_dist5 = [0.2, 0.2, 0.2, 0.2, 0.2]
act_dist6 = [1, 0, 0, 0, 0]


def calc_entropy(dist):
    my_sum = 0
    for p in dist:
        if p > 0:
            my_sum += p * np.log(p) / np.log(len(dist))
    return - my_sum


def calc_relative_entropy(dist1, dist2):
    my_sum = 0
    for p, q in zip(dist1, dist2):
        if p > 0 and q > 0:
            my_sum += p * np.log(p/q) / np.log(len(dist1))
    return my_sum


def calc_extropy(dist):
    my_sum = 0
    for p in dist:
        p = 1 - p
        if p > 0:
            my_sum += p * np.log(p) / np.log(len(dist))
    return - my_sum


a = calc_entropy(act_dist)
# print(calc_entropy(act_dist))
print(calc_relative_entropy(act_dist, act_dist))
# print(calc_relative_entropy(act_dist2, act_dist3))

task_entropy = calc_entropy(act_dist4)

relative_entropies = [calc_relative_entropy(act_dist, act_dist4), calc_relative_entropy(act_dist2, act_dist4),
                      calc_relative_entropy(act_dist3, act_dist4),
                      calc_relative_entropy(act_dist5, act_dist4), calc_relative_entropy(act_dist6, act_dist4)]

impact_entropies = [calc_entropy(act_dist), calc_entropy(act_dist2), calc_entropy(act_dist3), calc_entropy(act_dist5),
                    calc_entropy(act_dist6)]

task_extropy = calc_extropy(act_dist4)
impact_extropies = [calc_extropy(act_dist), calc_extropy(act_dist2), calc_extropy(act_dist3), calc_extropy(act_dist5),
                    calc_extropy(act_dist6)]

unnormalized_task_weight = (1 - task_entropy)
relative_entropy_task_weight = sum(relative_entropies) / len(relative_entropies)

task_weight = unnormalized_task_weight * (1 - relative_entropy_task_weight)

impact_1_weight = task_entropy * relative_entropies[0]
impact_weights = [(1 - task_entropy) * ent / sum(relative_entropies) for ent in relative_entropies]

all_weights = [task_weight] + impact_weights

all_weights_normalized = [w / sum(all_weights) for w in all_weights]

linear_space_task_entropy = np.power(len(act_dist4), task_entropy) / len(act_dist4)
linear_space_impact_entropies = [np.power(len(act_dist), ent) / len(act_dist4) for ent in impact_entropies]

linear_relative_entropies = [np.power(len(act_dist), ent) / len(act_dist4) for ent in relative_entropies]

linear_normalize_sum = ((1 - linear_space_task_entropy) + sum([(1-ent) * (1 - r_ent) for ent, r_ent in zip(linear_space_impact_entropies, linear_relative_entropies)]))
linear_task_self_weight = (1 - linear_space_task_entropy) / linear_normalize_sum
linear_impact_self_weights = [(1 - ent) * (1 - r_ent) / linear_normalize_sum for ent, r_ent in zip(linear_space_impact_entropies, linear_relative_entropies)]

print(f"task_entropy: {task_entropy}")
print(f"relative_entropies: {relative_entropies}")
print(f"unnormalized_task_weight: {unnormalized_task_weight}")
print(f"relative entropy task weight: {relative_entropy_task_weight}")
print(f"impact 1 weight: {impact_1_weight}")
print(f"Impact weights: {impact_weights}")
print(f"sum_impact_weights: {sum(impact_weights)}")
print(f"task_weight: {task_weight}")
print(f"sum weights: {task_weight + sum(impact_weights)}")
print(f"all_weights_normalized: {all_weights_normalized}")
print(f"Sum all weights: {sum(all_weights_normalized)}")
print(f"Linear_space_task_entropy: {linear_space_task_entropy}")
print(f"Linear_space_impact_entropies: {linear_space_impact_entropies}")
print(f"Linear Relative Entropies: {linear_relative_entropies}")
print(f"Linear Task Self Weight: {linear_task_self_weight}")
print(f"Linear Impact Self Weights: {linear_impact_self_weights}")
print(f"Sum linear_weights: {linear_task_self_weight + sum(linear_impact_self_weights)}")




