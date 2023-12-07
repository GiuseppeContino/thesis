import gymnasium as gym
import tqdm

import Utilities
import Policy

from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register


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

n_agents = len(Utilities.agents_color)

q_tables = [
    np.zeros((len(Utilities.agent_1_events), size * size, len(Utilities.actions))),
    np.zeros((len(Utilities.agent_2_events), size * size, len(Utilities.actions))),
    np.zeros((len(Utilities.agent_3_events), size * size, len(Utilities.actions)))
]

agents_trust = [
    np.zeros((len(Utilities.agent_1_events))),
    np.zeros((len(Utilities.agent_2_events))),
    np.zeros((len(Utilities.agent_3_events)))
]

n_value = [
    np.zeros((len(Utilities.agent_1_events))),
    np.zeros((len(Utilities.agent_2_events))),
    np.zeros((len(Utilities.agent_3_events)))
]

# train_env = gym.make('GridWorld-v0', render_mode='human', events=events, training=True)
train_env = gym.make('GridWorld-v0', events=Utilities.events, training=True)
# test_env = gym.make('GridWorld-v0', render_mode='human', events=events)
test_env = gym.make('GridWorld-v0', events=Utilities.events)

steps_list = []
trust_list = []

# train epochs loop
for epoch in tqdm.tqdm(range(epochs)):

    # train loop for the agents
    for agent_idx, agent in enumerate(Utilities.agents):

        # reset the environment for the single agent training
        agent_state = 0
        obs, _ = train_env.reset()

        # single agent train
        for step in range(max_episode_steps):

            # get the old state and clean the actions array
            state = obs[agent][1] * size + obs[agent][0]
            actions = []

            # set the other agents to do nothing
            for elem in range(agent_idx):
                actions.append(5)

            # compute the agent action
            actions.append(
                Policy.epsilon_greedy_policy(
                    train_env,
                    q_tables[agent_idx][0],
                    state,
                    epsilon,
                )
            )

            # Perform the environment step
            # TODO: change the reward
            obs, _, term, _, _ = train_env.step(actions)
            rew = train_env.unwrapped.get_agent_transitioned()

            # compute the new state
            new_state = obs[agent][1] * size + obs[agent][0]

            # save the actual q_value and max q_value in the state for simplify the writing
            actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]
            max_near_q_value = np.max(q_tables[agent_idx][agent_state][new_state])

            # update the agent q_table
            q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
                actual_q_value + learning_rate * (rew[agent_idx] + gamma * max_near_q_value - actual_q_value)
            )

            # move up the agent_state to the next RM state
            if train_env.unwrapped.get_next_flags()[agent_idx]:
                agent_state += 1

            # if the episode is terminated, break the loop
            if term:
                break

    # set the value for the evaluation after the training step
    obs, _ = test_env.reset()
    epoch_step = 0
    agent_states = [0, 0, 0]

    # test policy with all agents
    for step in range(max_episode_steps):

        epoch_step += 1

        state_1 = obs['agent_1'][1] * size + obs['agent_1'][0]
        state_2 = obs['agent_2'][1] * size + obs['agent_2'][0]
        state_3 = obs['agent_3'][1] * size + obs['agent_3'][0]

        actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
                   Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
                   Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

        # Perform the environment step
        obs, rew, term, _, _ = test_env.step(actions)

        # update the trust if an event is occurred
        if np.any(test_env.unwrapped.get_next_flags()):
            for agent_idx in range(n_agents):
                if rew[agent_idx] == 1.0:
                    n_value[agent_idx][agent_states[agent_idx]] += 1
                    agents_trust[agent_idx][agent_states[agent_idx]] = (
                        agents_trust[agent_idx][agent_states[agent_idx]] +
                        (test_env.unwrapped.get_agent_transitioned()[agent_idx] - agents_trust[agent_idx][agent_states[agent_idx]]) /
                        n_value[agent_idx][agent_states[agent_idx]]
                    )
                # TODO: change mean with exponential moving average (ema)
                #  ema = alpha * valore + (1 - alpha) * ema

        # update the using agents q_tables
        for flag_idx, flag in enumerate(test_env.unwrapped.get_next_flags()):
            if flag:
                agent_states[flag_idx] += 1

        # if the episode is terminated, break the loop
        if term:
            break

    # update the trust for event that are not occurred
    for agent_idx in range(n_agents):
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

# plot the # of step during evaluation
plt.plot(steps_list)
plt.show()

# plot the trust of agent_1 respect the press_button_1 event over time
plt.plot(trust_list)
plt.show()

# show the result (pass to a not trainer environment and to a full greedy policy)
show_env = gym.make('GridWorld-v0', render_mode='human', events=Utilities.events)

# set the value for show after the training without the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = [0, 0, 0]

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    state_1 = obs['agent_1'][1] * size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * size + obs['agent_3'][0]

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

    # Perform the environment step
    obs, _, term, _, _ = show_env.step(actions)
    rew = show_env.unwrapped.get_agent_transitioned()

    for flag_idx, flag in enumerate(show_env.unwrapped.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(rew)
    total_step += 1

    if term:
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)

# set the value for show after the training with the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = [0, 0, 0]
temp_plus = [0, 0, 0]

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    state_1 = obs['agent_1'][1] * size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * size + obs['agent_3'][0]

    for agent_idx in range(n_agents):

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
    rew = show_env.unwrapped.get_agent_transitioned()

    for flag_idx, flag in enumerate(show_env.unwrapped.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(rew)
    total_step += 1

    if term:
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)
