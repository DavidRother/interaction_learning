from collections import namedtuple

ParticleConfig = namedtuple("ParticleConfig", ["n_agents", "agents_x", "agents_y", "landmarks_x", "landmarks_y",
                                               "initial_std", "asymmetric_reward_locations", "collision_multiplier",
                                               "scenario_name", "num_goals", "shared_goals"])


antipodal = ParticleConfig(4, [-0.9, 0.9, -0.9, 0.9], [-0.9, 0.9, 0.9, -0.9], [0.9, -0.9, 0.9, -0.9],
                           [0.9, -0.9, -0.9, 0.9], 0, [], 1, "multi-goal_spread", 4, False)

cross = ParticleConfig(4, [-0.9, 0.9, 0.15, -0.15], [-0.15, 0.15, -0.9, 0.9], [0.9, -0.9, 0.15, -0.15],
                       [-0.15, 0.15, 0.9, -0.9], 0, [], 10, "multi-goal_spread", 4, False)

merge = ParticleConfig(2, [-0.9, -0.9], [0.2, -0.2], [0.9, 0.9], [-0.2, 0.2], 0.05, [], 1, "multi-goal_spread", 2,
                       False)

single = ParticleConfig(1, [-1.0], [-1.0], [1.0], [1.0], 0, [], 1, "multi-goal_spread", 1, False)

merge_asymmetric = ParticleConfig(2, [-0.9, -0.9], [0.2, -0.2], [0.9, 0.9], [-0.2, 0.2], 0.05, [[0.9, -0.2]], 10,
                                  "multi-goal_spread", 2, False)

competing_goal = ParticleConfig(2, [0.4, -0.9], [0.3, -0.9], [-0.1, 0.9], [0.85, 0.85], 0, [], 1,
                                "multi_common_goal_spread", 2, True)

single_agent_multi_goal = ParticleConfig(1, [-1.0], [-1.0], [0.8, 0.5], [0.5, 0.8], 0, [], 1,
                                         "multi_common_goal_spread", 2, True)

common_goal_circle = ParticleConfig(2, [0.0, -2.0], [-0.5, -2.0], [-0.5, 0.5], [0.0, 0.0], 0, [], 1,
                                    "common_goal_circle", 2, True)

common_goal_half_circle = ParticleConfig(2, [0.0, -2.0], [-0.5, -2.0], [-0.5, 0.5], [0.0, 0.0], 0, [], 1,
                                         "common_goal_half_circle", 2, True)

common_goal_two_sides = ParticleConfig(2, [0.0, -2.0], [-0.5, -2.0], [-0.5, 0.5], [0.0, 0.0], 0, [], 1,
                                       "common_goal_two_sides", 2, True)

