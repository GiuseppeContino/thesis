import gym
import tqdm

import Policy

from matplotlib import pyplot as plt
import numpy as np
from gym.envs.registration import register


size = 10

epsilon = 0.35

learning_rate = 0.7  # 0.8  # 0.7
gamma = 0.95  # 0.9  # 0.95

epochs = 250  # 1000
max_episode_steps = 300

# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=max_episode_steps,
)

agents = ['agent_1', 'agent_2', 'agent_3']
actions = ['up', 'right', 'down', 'left', 'push_button']
events = ['press_button_1', 'press_button_2', 'press_button_3_1', 'press_button_3_2', 'press_button_3', 'press_target']

agent_1_events = ['press_button_1', 'press_button_3', 'press_target']
agent_2_events = ['press_button_1', 'press_button_2', 'press_button_3_1', 'not_press_button_3_1', 'press_button_3']
agent_3_events = ['press_button_2', 'press_button_3_2', 'not_press_button_3_2', 'press_button_3']

q_tables_1 = np.zeros((len(agent_1_events), size * size, len(actions)))
q_tables_2 = np.zeros((len(agent_2_events), size * size, len(actions)))
q_tables_3 = np.zeros((len(agent_3_events), size * size, len(actions)))

q_tables = [q_tables_1, q_tables_2, q_tables_3]

agent_1_trust = np.zeros((len(agent_1_events)))
agent_2_trust = np.zeros((len(agent_2_events)))
agent_3_trust = np.zeros((len(agent_3_events)))

agents_trust = [agent_1_trust, agent_2_trust, agent_3_trust]

agent_1_n_value = np.zeros((len(agent_1_events)))
agent_2_n_value = np.zeros((len(agent_2_events)))
agent_3_n_value = np.zeros((len(agent_3_events)))

n_value = [agent_1_n_value, agent_2_n_value, agent_3_n_value]

# train_env = gym.make('GridWorld-v0', render_mode='human', events=events, training=True)
train_env = gym.make('GridWorld-v0', events=events, training=True)
# test_env = gym.make('GridWorld-v0', render_mode='human', events=events)
test_env = gym.make('GridWorld-v0', events=events)

steps_list = []
trust_list = []

# train loop for the agents
for epoch in tqdm.tqdm(range(epochs)):

    agent_states = [0, 0, 0]

    # single agent train
    for agent_idx, agent in enumerate(agents):

        obs, _ = train_env.reset()

        for agent_state in range(len(agent_states)):
            if not agent_state == agent_idx:
                agent_states[agent_idx] = 0

        for step in range(max_episode_steps):

            state = obs['agents'][agent][1] * size + obs['agents'][agent][0]

            actions = []

            # set the other agents to do nothing
            for elem in range(agent_idx):
                actions.append(5)

            actions.append(
                Policy.epsilon_greedy_policy(
                    train_env,
                    q_tables[agent_idx][0],
                    state,
                    epsilon,
                )
            )

            # Perform the environment step
            obs, _, term, _, _ = train_env.step(actions)
            rew = train_env.get_agent_transitioned()

            new_state = obs['agents'][agent][1] * size + obs['agents'][agent][0]

            actual_q_value = q_tables[agent_idx][agent_states[agent_idx]][state][actions[agent_idx]]
            max_near_q_value = np.max(q_tables[agent_idx][agent_states[agent_idx]][new_state])

            q_tables[agent_idx][agent_states[agent_idx]][state][actions[agent_idx]] = (
                actual_q_value + learning_rate * (rew[agent_idx] + gamma * max_near_q_value - actual_q_value)
            )

            if train_env.get_next_flags()[agent_idx]:
                agent_states[agent_idx] += 1

            if term:
                break

    obs, _ = test_env.reset()
    epoch_step = 0
    agent_states = [0, 0, 0]

    # test policy with all agents
    for step in range(max_episode_steps):

        epoch_step += 1

        state_1 = obs['agents']['agent_1'][1] * size + obs['agents']['agent_1'][0]
        state_2 = obs['agents']['agent_2'][1] * size + obs['agents']['agent_2'][0]
        state_3 = obs['agents']['agent_3'][1] * size + obs['agents']['agent_3'][0]

        actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
                   Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
                   Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

        # Perform the environment step
        obs, rew, term, _, _ = test_env.step(actions)

        # update the trust if an event is occurred
        if np.any(test_env.get_next_flags()):
            for agent_idx in range(len(agents)):
                if rew[agent_idx] == 1.0:
                    n_value[agent_idx][agent_states[agent_idx]] += 1
                    agents_trust[agent_idx][agent_states[agent_idx]] = (
                        agents_trust[agent_idx][agent_states[agent_idx]] +
                        (test_env.get_agent_transitioned()[agent_idx] - agents_trust[agent_idx][agent_states[agent_idx]]) /
                        n_value[agent_idx][agent_states[agent_idx]]
                    )
                # ema = alpha * valore + (1 - alpha) * ema

        # update the using agents q_tables
        for flag_idx, flag in enumerate(test_env.get_next_flags()):
            if flag:
                agent_states[flag_idx] += 1

        # check for termination
        if term:
            # steps_list.append(epoch_step)
            break

    # update the trust for event that are not occurred
    for agent_idx in range(len(agents)):
        for trust_idx in range(len(agents_trust[agent_idx])):
            if n_value[agent_idx][trust_idx] < epoch + 1:
                n_value[agent_idx][trust_idx] += 1
                agents_trust[agent_idx][trust_idx] = (
                        agents_trust[agent_idx][trust_idx] +
                        (0 - agents_trust[agent_idx][trust_idx]) /
                        n_value[agent_idx][trust_idx]
                )

    steps_list.append(epoch_step)
    trust_list.append(agents_trust[0][0])

print('agents trust', agents_trust)

np.set_printoptions(suppress=True)

plt.plot(steps_list)
plt.show()

plt.plot(trust_list)
plt.show()

# show the result ( pass to a not trainer environment and to a full greedy policy )
show_env = gym.make('GridWorld-v0', render_mode='human', events=events)

# reset the environment
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = [0, 0, 0]

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    state_1 = obs['agents']['agent_1'][1] * size + obs['agents']['agent_1'][0]
    state_2 = obs['agents']['agent_2'][1] * size + obs['agents']['agent_2'][0]
    state_3 = obs['agents']['agent_3'][1] * size + obs['agents']['agent_3'][0]

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

    # Perform the environment step
    obs, _, term, _, _ = show_env.step(actions)
    rew = show_env.get_agent_transitioned()

    for flag_idx, flag in enumerate(show_env.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(rew)
    total_step += 1

    if term:
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)

# reset the environment
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = [0, 0, 0]

temp_plus = [0, 0, 0]

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    state_1 = obs['agents']['agent_1'][1] * size + obs['agents']['agent_1'][0]
    state_2 = obs['agents']['agent_2'][1] * size + obs['agents']['agent_2'][0]
    state_3 = obs['agents']['agent_3'][1] * size + obs['agents']['agent_3'][0]

    for agent_idx in range(len(agents)):

        # trust problem if not reach the threshold
        if agents_trust[agent_idx][agent_states[agent_idx]] < 0.2:
            temp_plus[agent_idx] = 1
        else:
            temp_plus[agent_idx] = 0

        if agent_states[agent_idx] + temp_plus[agent_idx] >= len(agents_trust[agent_idx]):
            temp_plus[agent_idx] = 0

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0] + temp_plus[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1] + temp_plus[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2] + temp_plus[2]], state_3)]

    # Perform the environment step
    obs, _, term, _, _ = show_env.step(actions)
    rew = show_env.get_agent_transitioned()

    for flag_idx, flag in enumerate(show_env.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(rew)
    total_step += 1

    if term:
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)
