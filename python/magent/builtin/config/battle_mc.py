""" battle of two armies """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    multi_channel = cfg.register_agent_type(
        "multi_channel",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         #for multi-channel communication
         'comm_channel': 2, 'hear_all_group': False,

         'step_reward': -0.005,  'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    single_channel = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         #for multi-channel communication
         'comm_channel': 1, 'hear_all_group': False,

         'step_reward': -0.005,  'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    g = []
    g.append(cfg.add_group(multi_channel))
    g.append(cfg.add_group(multi_channel))
    g.append(cfg.add_group(single_channel))
    g.append(cfg.add_group(single_channel))

    a = []
    for i in g:
        a.append(gw.AgentSymbol(i, index='any'))

    # reward shaping to encourage attack
    for i in a:
        for j in a:
            if(i!=j):
                cfg.add_reward_rule(gw.Event(i, 'attack', j), receiver=i, value=0.2)

    return cfg
