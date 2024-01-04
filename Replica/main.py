import gymnasium as gym
import tqdm

import Utilities
import Policy

from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register


# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=Utilities.max_episode_steps,
)

n_agents = len(Utilities.agents_color)

max_actions = len(Utilities.actions)

q_tables = [
    np.zeros((7, Utilities.size * Utilities.size, len(Utilities.actions))),
    np.zeros((9, Utilities.size * Utilities.size, len(Utilities.actions))),
    np.zeros((12, Utilities.size * Utilities.size, len(Utilities.actions)))
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
for epoch in tqdm.tqdm(range(Utilities.epochs)):

    # train loop for the agents
    for agent_idx, agent in enumerate(Utilities.agents):

        # reset the environment for the single agent training
        agent_state = 0
        obs, _ = train_env.reset()

        # single agent train
        for step in range(Utilities.max_episode_steps):

            # get the old state and clean the actions array
            state = obs[agent][1] * Utilities.size + obs[agent][0]
            actions = []

            # set the other agents to do nothing
            for elem in range(agent_idx):
                actions.append(max_actions)

            # compute the agent action
            actions.append(
                Policy.epsilon_greedy_policy(
                    train_env,
                    q_tables[agent_idx][0],
                    state,
                    Utilities.epsilon,
                )
            )

            for elem in range(len(Utilities.agents) - agent_idx - 1):
                actions.append(max_actions)

            actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}
            # print(actions_dict)

            # Perform the environment step
            obs, rew, term, _, _ = train_env.step(actions_dict)

            # compute the new state
            new_state = obs[agent][1] * Utilities.size + obs[agent][0]

            # save the actual q_value and max q_value in the state for simplify the writing
            actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]
            max_near_q_value = np.max(q_tables[agent_idx][agent_state][new_state])

            # update the agent q_table
            q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
                actual_q_value + Utilities.learning_rate * (rew['agent_' + str(agent_idx + 1)] +
                Utilities.gamma * max_near_q_value - actual_q_value)
            )

            # move up the agent_state to the next RM state
            if train_env.unwrapped.get_next_flags()[agent_idx]:
                agent_state += 1

            # if the episode is terminated, break the loop
            if np.any(list(term.values())):
                break

    # set the value for the evaluation after the training step
    obs, _ = test_env.reset()
    epoch_step = 0
    agent_states = [0, 0, 0]

    # test policy with all agents
    for step in range(Utilities.max_episode_steps):

        epoch_step += 1

        state_1 = obs['agent_1'][1] * Utilities.size + obs['agent_1'][0]
        state_2 = obs['agent_2'][1] * Utilities.size + obs['agent_2'][0]
        state_3 = obs['agent_3'][1] * Utilities.size + obs['agent_3'][0]

        actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
                   Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
                   Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}
        # print(actions_dict)

        # Perform the environment step
        obs, rew, term, _, _ = test_env.step(actions_dict)

        # update the trust if an event is occurred
        if np.any(test_env.unwrapped.get_next_flags()):
            for agent_idx in range(n_agents):
                if test_env.unwrapped.get_next_flags()[agent_idx]:
                    n_value[agent_idx][agent_states[agent_idx]] += 1
                    agents_trust[agent_idx][agent_states[agent_idx]] = (
                        Utilities.alpha * agents_trust[agent_idx][agent_states[agent_idx]] +
                        (1 - Utilities.alpha) * rew['agent_' + str(agent_idx + 1)]
                    )

        # update the using agents q_tables
        for flag_idx, flag in enumerate(test_env.unwrapped.get_next_flags()):
            if flag:
                agent_states[flag_idx] += 1

        # if the episode is terminated, break the loop
        if np.all(list(term.values())):
            break

    # update the trust for event that are not occurred
    for agent_idx in range(n_agents):
        for trust_idx in range(len(agents_trust[agent_idx])):
            if n_value[agent_idx][trust_idx] < epoch + 1:
                n_value[agent_idx][trust_idx] += 1
                agents_trust[agent_idx][agent_states[agent_idx]] = (
                    Utilities.alpha * agents_trust[agent_idx][agent_states[agent_idx]] +
                    (1 - Utilities.alpha) * 0
                )

    steps_list.append(epoch_step)
    trust_list.append(agents_trust[0][2])

print('agents trust', agents_trust)

# plot the # of step during evaluation
plt.plot(steps_list)
plt.show()

# plot the trust of agent_1 respect the press_button_1 event over time
plt.plot(trust_list)
plt.show()

# show the result (pass to a not trainer environment and to a full greedy policy)
show_env = gym.make('GridWorld-v0', render_mode='human', events=Utilities.events)

print(q_tables[0].shape)
print(q_tables[1].shape)
print(q_tables[2].shape)
print(q_tables[0][0])

# set the value for show after the training without the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = [0, 0, 0]

# start the steps loop
for step in tqdm.tqdm(range(Utilities.max_episode_steps)):

    print(agent_states)

    state_1 = obs['agent_1'][1] * Utilities.size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * Utilities.size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * Utilities.size + obs['agent_3'][0]

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

    # actions = []
    # for i in range(0, 3):
    #     ele = int(input())
    #     actions.append(ele)

    actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}
    # print(actions_dict)

    # Perform the environment step
    obs, rew, term, _, _ = show_env.step(actions_dict)

    print(rew)

    for flag_idx, flag in enumerate(show_env.unwrapped.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(list(rew.values()))
    total_step += 1

    if np.all(list(term.values())):
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
for step in tqdm.tqdm(range(Utilities.max_episode_steps)):

    state_1 = obs['agent_1'][1] * Utilities.size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * Utilities.size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * Utilities.size + obs['agent_3'][0]

    for agent_idx in range(n_agents):

        # trust problem if not reach the threshold
        if agents_trust[agent_idx][agent_states[agent_idx]] < 0.2:
            # TODO: temp_plus can be greater than 1 if a sequence of state have no trust
            temp_plus[agent_idx] = 1
        else:
            temp_plus[agent_idx] = 0

        if agent_states[agent_idx] + temp_plus[agent_idx] >= len(agents_trust[agent_idx]):
            temp_plus[agent_idx] = 0

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0] + temp_plus[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1] + temp_plus[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2] + temp_plus[2]], state_3)]

    actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}
    # print(actions_dict)

    # Perform the environment step
    obs, rew, term, _, _ = show_env.step(actions_dict)

    for flag_idx, flag in enumerate(show_env.unwrapped.get_next_flags()):
        if flag:
            agent_states[flag_idx] += 1

    total_rew += sum(list(rew.values()))
    total_step += 1

    if np.all(list(term.values())):
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)
