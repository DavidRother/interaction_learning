import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.optim as optim
from clean_rl.rl_core.agents.model import GatheringModel
from gathering_zoo.environment import gathering_zoo
from clean_rl.rl_core import collect_experience
from interaction_learning.algorithms.ppo.buffer import MultiAgentBuffer
from interaction_learning.algorithms.ppo.postprocessing import postprocess
from interaction_learning.algorithms.ppo.ppo_loss import ppo_surrogate_loss

num_agents = 1
training_iterations = 5000
training_steps = 400
num_batches = 1
num_epochs = 4
lr = 1e-4

agent_1_config = {"prosocial_level": 0.0, "update_prosocial_level": False, "use_prosocial": False}

agent_configs = {"player_0": agent_1_config}

# initialize agents
agents = {f"player_{idx}": GatheringModel() for idx in range(num_agents)}
optimizer = {f"player_{idx}": optim.Adam(agents[f"player_{idx}"].parameters(), lr=lr) for idx in range(num_agents)}

# initialize environment
env_config = {
    "level": 'deterministic_room',
    "num_agents": num_agents,
    "record": False,
    "max_steps": 100,
    "reward_scheme": "scheme_1"
}
env = gathering_zoo.parallel_env(**env_config)

# initialize training loop
buffer = MultiAgentBuffer(training_steps, num_agents)

for training_iteration in range(training_iterations):
    episode_stats = collect_experience(env, buffer, agents, training_steps)
    postprocess(agents, buffer, agent_configs)

    rewards = {player: sum(buffer.buffer[player][REWARDS]) for player in agents}

    for epoch in range(num_epochs):
        for player in agents:
            for batch in buffer.buffer[player].build_batches(num_batches):
                loss, stats = ppo_surrogate_loss(agents[player], buffer.buffer[player], agent_configs[player])

                optimizer[player].zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                optimizer[player].step()

        # print(player, stats)
    print(episode_stats)
    buffer.reset()