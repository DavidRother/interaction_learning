import numpy as np
import pandas as pd
from interaction_learning.environments.multiagent_particle_envs.multiagent.core import World, Agent, Landmark
from interaction_learning.environments.multiagent_particle_envs.multiagent.scenario import BaseScenario
import random

colors = np.array([[221,127,106],
                   [204,169,120],
                   [191,196,139],
                   [176,209,152],
                   [152,209,202],
                   [152,183,209],
                   [152,152,209],
                   [185,152,209],
                   [209,152,203],
                   [209,152,161]])


class Scenario(BaseScenario):

    def __init__(self):
        self.n_agents = 0
        self.n_landmarks = 0
        self.agents_x = []
        self.agents_y = []
        self.landmarks_x = []
        self.landmarks_y = []
        self.initial_std = 0
        self.prob_random = 0
        self.colors = None
        self.collisions = 0
        self.time_penalty = False
        self.random_scenario = False
        self.asymmetric_reward_location = []
        self.collision_multiplier = 1
        self.agent_landmark_bindings = {}
        self.correct_goal = None
        self.common_goals = []
        self.landmark_locations = ["left", "right"]
        self.agent_claims = ["None", "None"]

    def make_world(self, n_agents, config, prob_random):
        """
        n_agents - number of agents (also equal to number of target landmarks)
        config - dictionary
        prob_random - probability of random agent and landmark initial locations
        """
        world = World()
        # set any world properties first
        world.dim_c = 0
        self.n_agents = n_agents
        self.n_landmarks = len(config['landmarks_x'])
        self.agents_x = config['agents_x']
        self.agents_y = config['agents_y']
        self.landmarks_x = config['landmarks_x']
        self.landmarks_y = config['landmarks_y']
        self.collision_multiplier = config["collision_multiplier"]
        # standard deviation of Gaussian noise on initial agent location
        self.initial_std = config['initial_std']
        self.prob_random = prob_random
        self.common_goals = config["shared_goals"]
        self.landmark_locations = ["left", "right"]
        self.agent_claims = ["None", "None"]
        num_agents = n_agents
        num_landmarks = len(config['landmarks_x'])
        # deliberate False, to prevent environment.py from sharing reward
        world.collaborative = False 
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.idx = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # agent.size = 0.1
            agent.reached = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.idx = i
            landmark.collide = False
            landmark.movable = False
        # read colors
        # self.colors = np.loadtxt('colors.csv', delimiter=',')
        self.colors = colors
        # make initial conditions
        self.random_scenario = False
        self.asymmetric_reward_location = config["asymmetric_reward_locations"]
        self.reset_world(world)
        self.time_penalty = True
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = self.colors[i]/256
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = self.colors[i]/256
        # set initial states
        rand_num = random.random()
        rand_landmarks_claimed = self.generate_random_landmark_claimed(self.n_landmarks, self.n_agents)
        if rand_num < self.prob_random:
            self.random_scenario = True
            self.asymmetric_reward_location = []
        for i, landmark in enumerate(world.landmarks):
            if rand_num < self.prob_random:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.claimed = rand_landmarks_claimed[i]
            else:
                landmark.state.p_pos = np.array([self.landmarks_x[i], self.landmarks_y[i]])
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            if rand_num < self.prob_random:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                if i > 0:
                    # all but the first agent will be placed on a circle around the middle point of the two goals
                    distance = np.random.uniform(1.5, 3)
                    sampled_angle = np.random.uniform(0.0, 2 * np.pi)
                    x = distance * np.cos(sampled_angle) + np.mean(self.landmarks_x)
                    y = distance * np.sin(sampled_angle) + np.mean(self.landmarks_y)
                    agent.state.p_pos = np.array([x, y])
                    distances = []
                    for idx, target in enumerate(world.landmarks):
                        if target.claimed:
                            distances.append(-1)
                        else:
                            distances.append(np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos))))

                    argmax_distance = np.argmax(distances).item()
                    self.correct_goal = self.landmark_locations[argmax_distance]
                else:
                    x = self.agents_x[i] + np.random.normal(0, self.initial_std)
                    y = self.agents_y[i] + np.random.normal(0, self.initial_std)
                    # agent.state.p_pos = np.array([self.agents_x[i], self.agents_y[i]])
                    agent.state.p_pos = np.array([x, y])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.reached = False
        self.collisions = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world):
        rew = 0

        if agent.reached:
            target = self.agent_landmark_bindings[agent]
            distance = np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
            # rew -= distance
        else:
            distances = []
            for idx, target in enumerate(world.landmarks):
                if target.claimed:
                    distances.append(np.inf)
                else:
                    distances.append(np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos))))

            min_distance = -1 * np.min(distances)
            argmin_distance = np.argmin(distances).item()

            if min_distance >= -0.05:
                agent.reached = True
                agent_idx = world.agents.index(agent)
                self.agent_claims[agent_idx] = self.landmark_locations[argmin_distance]
                world.landmarks[argmin_distance].claimed = True
                self.agent_landmark_bindings[agent] = world.landmarks[argmin_distance]

            rew += min_distance

            rew -= 0.3

        collision_ignore = False
        for landmark_location in self.asymmetric_reward_location:
            collision_ignore = np.allclose(np.asarray(landmark_location), target.state.p_pos)


        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    if not collision_ignore:
                        rew -= 1 * self.collision_multiplier
                    # Note that this will double count, since reward()
                    # will be called by environment again for the other agent
                    self.collisions += 1

        return rew

    def done(self, agent, world):
        if agent.reached:
            return True
        return False

    def observation(self, agent, world):
        others = []
        for other in world.agents:
            if other is agent and self.n_agents > 1:
                # allow storing of self in other_pos when n=1, so that
                # np.concat works. It won't be used in alg.py anyway when n=1
                continue
            others.append(other.state.p_vel - agent.state.p_vel)  # relative velocity
            others.append(other.state.p_pos - agent.state.p_pos)  # relative position
            others.append(np.asarray([other.reached]))  # done flag
        return np.concatenate([agent.state.p_vel, agent.state.p_pos, np.asarray([agent.reached])]), \
               np.concatenate(others)

    def generate_random_landmark_claimed(self, num_landmarks, num_agents):
        safe = [False] * num_agents
        rand = random.choices([True, False], k=num_landmarks - num_agents)
        rand_landmarks = safe + rand
        random.shuffle(rand_landmarks)
        return rand_landmarks
