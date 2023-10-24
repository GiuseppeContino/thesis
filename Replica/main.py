import gym
import random
import tqdm

from matplotlib import pyplot as plt
import numpy as np
from gym.envs.registration import register


def epsilon_greedy_policy(environment, q_table, act_state, bound):
    random_float = random.uniform(0, 1)

    if random_float > bound:
        action = np.argmax(q_table[act_state])
    else:
        action = environment.action_space.sample()

    return action


def greedy_policy(q_table, act_state):
    action = np.argmax(q_table[act_state])

    return action


size = 10

epsilon = 0.35

learning_rate = 0.8  # 0.7
gamma = 0.9  # 0.95

epochs = 10000  # 100000
max_episode_steps = 300  # 1000  # 300

# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=max_episode_steps,
)

agents = ['agent_1', 'agent_2', 'agent_3']
actions = ['up', 'right', 'down', 'left', 'push_button']
events = ['button_1', 'button_2', 'button_3', 'target']

q_tables = np.zeros((len(agents), len(events), size * size, len(actions)))

test_env = gym.make('GridWorld-v0', events=events)
train_env = gym.make('GridWorld-v0', events=events, training=True)

steps_list = []

# train loop for the agents
for epoch in tqdm.tqdm(range(epochs)):

    for agent_idx, agent in enumerate(agents):

        obs, _ = train_env.reset()

        for step in range(max_episode_steps):

            state = obs['agents'][agent][1] * size + obs['agents'][agent][0]

            actions = []

            # set the other agents to do nothing
            for elem in range(agent_idx):
                actions.append(5)

            actions.append(epsilon_greedy_policy(train_env,
                                                 q_tables[agent_idx][train_env.get_reward_machine().get_idx()],
                                                 state,
                                                 epsilon,
                                                 ))

            # Perform the environment step
            obs, rew, term, _, info = train_env.step(actions)

            new_state = obs['agents'][agent][1] * size + obs['agents'][agent][0]

            machine_idx = train_env.get_reward_machine().get_idx()

            if train_env.get_next_flag():  # and rew == 1:
                # q_tables[agent_idx][machine_idx - 1][state][actions[agent_idx]] = rew

                actual_q_value = q_tables[agent_idx][machine_idx - 1][state][actions[agent_idx]]
                max_near_q_value = np.max(q_tables[agent_idx][machine_idx - 1][new_state])

                q_tables[agent_idx][machine_idx - 1][state][actions[agent_idx]] = ((1 - learning_rate) * actual_q_value
                                                                                   + learning_rate * (rew + gamma *
                                                                                   max_near_q_value))

            else:

                actual_q_value = q_tables[agent_idx][machine_idx][state][actions[agent_idx]]
                max_near_q_value = np.max(q_tables[agent_idx][machine_idx][new_state])

                q_tables[agent_idx][machine_idx][state][actions[agent_idx]] = ((1 - learning_rate) * actual_q_value +
                                                                               learning_rate * (rew + gamma *
                                                                               max_near_q_value))

            if term:
                break

    obs, _ = test_env.reset()
    epoch_step = 0

    for step in range(max_episode_steps):

        epoch_step += 1

        state_1 = obs['agents']['agent_1'][1] * size + obs['agents']['agent_1'][0]
        state_2 = obs['agents']['agent_2'][1] * size + obs['agents']['agent_2'][0]
        state_3 = obs['agents']['agent_3'][1] * size + obs['agents']['agent_3'][0]

        reward_machine_idx = test_env.get_reward_machine().get_idx()

        actions = [greedy_policy(q_tables[0][reward_machine_idx], state_1),
                   greedy_policy(q_tables[1][reward_machine_idx], state_2),
                   greedy_policy(q_tables[2][reward_machine_idx], state_3)]

        # Perform the environment step
        obs, rew, term, _, info = test_env.step(actions)

        if term:
            break

    steps_list.append(epoch_step)

np.set_printoptions(suppress=True)
print(q_tables[0][0])

print(len(steps_list))
print(steps_list[0])
print(steps_list[:-50])

plt.plot(steps_list)
plt.show()

# show the result ( pass to a not trainer environment and to a full greedy policy )
show_env = gym.make('GridWorld-v0', render_mode='human', events=events)

obs, _ = show_env.reset()

total_rew = 0
total_step = 0

for step in tqdm.tqdm(range(max_episode_steps)):

    state_1 = obs['agents']['agent_1'][1] * size + obs['agents']['agent_1'][0]
    state_2 = obs['agents']['agent_2'][1] * size + obs['agents']['agent_2'][0]
    state_3 = obs['agents']['agent_3'][1] * size + obs['agents']['agent_3'][0]

    actions = [greedy_policy(q_tables[0][show_env.get_reward_machine().get_idx()], state_1),
               greedy_policy(q_tables[1][show_env.get_reward_machine().get_idx()], state_2),
               greedy_policy(q_tables[2][show_env.get_reward_machine().get_idx()], state_3)]

    # Perform the environment step
    obs, rew, term, _, info = show_env.step(actions)
    total_rew += rew
    total_step += 1

    if term:
        break

print('accumulate reward function:', total_rew)
print('# of steps to complete the task:', total_step)
