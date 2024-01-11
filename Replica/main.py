import gymnasium as gym
import tqdm

import Utilities
import Policy

import copy

from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register


# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=Utilities.max_episode_steps,
)

agents_pythomata_rm = [
    Utilities.create_first_individual_rm(),
    Utilities.create_second_individual_rm(),
    Utilities.create_third_individual_rm()
]

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

# train_env = gym.make('GridWorld-v0', render_mode='human', events=Utilities.events, training=True,
#   rewarding_machines=agents_pythomata_rm)
train_env = gym.make('GridWorld-v0', events=Utilities.events, training=True, rewarding_machines=agents_pythomata_rm)
# test_env = gym.make('GridWorld-v0', render_mode='human', events=events)
test_env = gym.make('GridWorld-v0', events=Utilities.events, rewarding_machines=agents_pythomata_rm)

steps_list = []
trust_list = []

# train epochs loop
for epoch in tqdm.tqdm(range(Utilities.epochs)):

    # train loop for the agents
    for agent_idx, agent in enumerate(Utilities.agents):

        # reset the environment for the single agent training
        obs, _ = train_env.reset()
        agent_state = train_env.unwrapped.get_next_flags()[agent_idx]

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

            # Perform the environment step
            obs, rew, term, _, _ = train_env.step(actions_dict)

            # compute the new state
            new_state = obs[agent][1] * Utilities.size + obs[agent][0]

            # save the actual q_value and max q_value in the state for simplify the writing
            actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]
            max_near_q_value = np.max(q_tables[agent_idx][agent_state][new_state])

            # update the agent q_table
            q_tables[agent_idx][agent_state][state][actions[agent_idx]] = min((
                actual_q_value + Utilities.learning_rate * (rew['agent_' + str(agent_idx + 1)] +
                                                            Utilities.gamma * max_near_q_value - actual_q_value)
            ), 1)

            # move up the agent_state to the next RM state
            agent_state = train_env.unwrapped.get_next_flags()[agent_idx]

            # if the episode is terminated, break the loop
            if np.any(list(term.values())):
                break

    # test training every test_num value
    if epoch % Utilities.test_num == 0:

        # set the value for the evaluation after the training step
        obs, _ = test_env.reset()
        epoch_step = 0

        agent_states = test_env.get_next_flags()

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

            # Perform the environment step
            obs, rew, term, _, _ = test_env.step(actions_dict)

            agent_states = test_env.unwrapped.get_next_flags()

            # if the episode is terminated, break the loop
            if np.all(list(term.values())):
                break

        steps_list.append(epoch_step)
        trust_list.append(agents_trust[0][2])

# plot the # of step during evaluation
plt.plot(steps_list)
plt.show()

# print(q_tables[1][0])
# print(q_tables[2][0])
# print(q_tables[2][1])
# print(q_tables[2][2])
# print(q_tables[2][3])

# show the result (pass to a not trainer environment and to a full greedy policy)
show_env = gym.make('GridWorld-v0', render_mode='human', events=Utilities.events, rewarding_machines=agents_pythomata_rm)

# set the value for show after the training without the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = train_env.unwrapped.get_next_flags()

# start the steps loop
for step in tqdm.tqdm(range(Utilities.max_episode_steps)):

    state_1 = obs['agent_1'][1] * Utilities.size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * Utilities.size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * Utilities.size + obs['agent_3'][0]

    actions = [Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
               Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
               Policy.greedy_policy(q_tables[2][agent_states[2]], state_3)]

    # # select manually the agents action
    # actions = []
    # for i in range(0, 3):
    #     ele = int(input())
    #     actions.append(ele)

    actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

    # Perform the environment step
    obs, rew, term, _, _ = show_env.step(actions_dict)

    agent_states = show_env.unwrapped.get_next_flags()

    total_rew += sum(list(rew.values()))
    total_step += 1

    if np.all(list(term.values())):
        print(show_env.unwrapped.task_trust)
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)

# set the value for show after the training with the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

agent_states = show_env.unwrapped.get_next_flags()

# start the steps loop
for step in tqdm.tqdm(range(Utilities.max_episode_steps)):

    state_1 = obs['agent_1'][1] * Utilities.size + obs['agent_1'][0]
    state_2 = obs['agent_2'][1] * Utilities.size + obs['agent_2'][0]
    state_3 = obs['agent_3'][1] * Utilities.size + obs['agent_3'][0]

    temp_states = copy.copy(agent_states)

    # anticipate the tasks
    for agent_idx in range(n_agents):

        temp_goal = show_env.unwrapped.agents[agent_idx].temporal_goal
        temp_state = copy.copy(temp_goal.current_state)

        while (np.array_equal(q_tables[agent_idx][temp_state], np.zeros_like(q_tables[agent_idx][temp_state])) and
                len(list(temp_goal.automaton.get_transitions_from(temp_state))) == 1):

            if str(list(list(temp_goal.automaton.get_transitions_from(temp_state))[0])[1])[:-1] == 'press_target_':
                break

            temp_states[agent_idx] = list(list(
                    temp_goal.automaton.get_transitions_from(temp_state))[0])[2] - 1

            temp_state = copy.copy(temp_states[agent_idx] + 1)

    actions = [Policy.greedy_policy(q_tables[0][temp_states[0]], state_1),
               Policy.greedy_policy(q_tables[1][temp_states[1]], state_2),
               Policy.greedy_policy(q_tables[2][temp_states[2]], state_3)]

    actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

    # Perform the environment step
    obs, rew, term, _, _ = show_env.step(actions_dict)

    agent_states = show_env.unwrapped.get_next_flags()

    total_rew += sum(list(rew.values()))
    total_step += 1

    if np.all(list(term.values())):
        print(show_env.unwrapped.task_trust)
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)
