from gym_cooking.environment import cooking_zoo


cooking_config = {
    "n_agents": 1,
    "level": 'open_room_salad',
    "record": False,
    "max_steps": 100,
    "recipes": ["TomatoSalad"],
    "action_scheme": "scheme3"
}

cooking_env_creator = lambda: cooking_zoo.parallel_env(**cooking_config)

env_creator_map = {"cooking_env": cooking_env_creator}
