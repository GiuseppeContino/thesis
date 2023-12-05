import random

import gym
from gym import spaces
import pygame

import Agent
import Utilities

from temprl.reward_machines.automata import RewardAutomaton
from temprl.wrapper import TemporalGoal

import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, events=None, training=False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.events = events
        self.training = training

        self.train_transition = 0.95

        self.pass_events = []

        # self.reward_machine = Reward_machine.RewardMachine(events=self.events)
        # self.next_flag = False
        self.next_flags = [False, False, False]

        # alphabet = Utilities.alphabet
        states = Utilities.states
        initial_state = 'init'
        goal_state = 'end_state'
        # accepting_states = {goal_state}

        transition_function = Utilities.transition_function

        # create the pythomata
        # self.pythomata_rm = pythomata.SimpleDFA(
        #     states,
        #     alphabet,
        #     initial_state,
        #     accepting_states,
        #     transition_function,
        # )

        pythomata_rm, _ = Utilities.transition_function_to_symbolic(transition_function, states)

        # save a file with the automata
        graph = pythomata_rm.to_graphviz()
        graph.render('./images/reward_machine')
        # print(self.pythomata_rm)

        # change to a reward automata
        automata = RewardAutomaton(pythomata_rm, 1)

        # wrap it in a TemporalGoal
        self.temp_goal = TemporalGoal(automata)

        # Agents initial position and colors
        agents_location = Utilities.agents_initial_location

        agents_color = Utilities.agents_color

        self.agents = []

        agent_1_events = ['press_button_1', 'press_button_3', 'press_target']
        agent_2_events = ['press_button_1', 'press_button_2', 'press_button_3_1', 'not_press_button_3_1',
                          'press_button_3']
        agent_3_events = ['press_button_2', 'press_button_3_2', 'not_press_button_3_2', 'press_button_3']

        agents_events = [agent_1_events, agent_2_events, agent_3_events]
        # print('environment: agent events', agents_events)

        for idx, agent_events in enumerate(agents_events):

            agent_transition_function = Utilities.individual_transition_function(
                agents_events[idx],
                initial_state,
                goal_state
            )

            agent_states = {state for state in agent_transition_function.keys()}
            agent_states.add('end_state')

            agent_pythomata_rm, _ = Utilities.transition_function_to_symbolic(agent_transition_function, agent_states)

            # agent_pythomata_rm = pythomata.SimpleDFA(
            #     agent_states,
            #     alphabet,
            #     initial_state,
            #     accepting_states,
            #     agent_transition_function,
            # )

            agent_graph = agent_pythomata_rm.to_graphviz()
            agent_graph.render('./images/agent_' + str(idx + 1) + '_reward_machine')

            # print(agent_pythomata_rm)
            agent_automata = RewardAutomaton(agent_pythomata_rm, 1)
            agent_temp_goal = TemporalGoal(agent_automata)
            # agent_temp_goal.reset()

            # print(agent_events[idx])

            self.agents.append(Agent.Agent(
                agents_location[idx],
                agents_color[idx],
                agent_temp_goal,
                agents_events[idx]
            ))

        # self.automata = RewardAutomaton(self.pythomata_rm, 1)
        # self.state = self.automata.initial_state
        #
        # # move a TemporalGoal library
        # self.temp_goal = TemporalGoal(self.automata)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                'agents': spaces.Dict({
                    'agent_1': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    'agent_2': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    'agent_3': spaces.Box(0, size - 1, shape=(2,), dtype=int)
                }),
                'doors': spaces.Dict({
                    'door_1': spaces.Dict({
                        'door_position': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_button': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_opener': spaces.Discrete(3),
                        'door_open_flag': spaces.Discrete(2)
                    }),
                    'door_2': spaces.Dict({
                        'door_position': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_button': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_opener': spaces.Discrete(3),
                        'door_open_flag': spaces.Discrete(2)
                    }),
                    'door_3': spaces.Dict({
                        'door_position': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_button': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        'door_opener': spaces.Discrete(3),
                        'door_open_flag': spaces.Discrete(2)
                    })
                }),
                'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to 'right', 'up', 'left', 'down', 'push_button'
        self.action_space = spaces.Discrete(5)

        '''
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        '''
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Target position
        self._target_location = np.array((9, 9))

        self._doors_location = [
            np.array((5, 4)),
            np.array((8, 4)),
            np.array((4, 8))
        ]
        self._doors_button = [
            np.array((3, 0)),
            np.array((4, 6)),
            np.array((9, 6))
        ]

        if self.training:
            self._doors_opener = [1, 1, 1]
        else:
            self._doors_opener = [1, 1, 2]

        self.events_idx = 0

        self._doors_flag = [1, 1, 1]

        self._doors_color = [(127, 255, 0), (255, 0, 127), (0, 127, 255)]

        self._walls = (
            # First vertical wall
            ((3, 0), (4, 0)),
            ((3, 1), (4, 1)),
            ((3, 2), (4, 2)),
            ((3, 3), (4, 3)),
            ((3, 4), (4, 4)),
            ((3, 5), (4, 5)),
            ((3, 6), (4, 6)),
            ((3, 7), (4, 7)),
            ((3, 9), (4, 9)),
            # First horizontal wall
            ((4, 6), (4, 7)),
            ((5, 6), (5, 7)),
            ((6, 6), (6, 7)),
            ((7, 6), (7, 7)),
            ((8, 6), (8, 7)),
            ((9, 6), (9, 7)),
            # Second vertical wall
            ((6, 0), (7, 0)),
            ((6, 1), (7, 1)),
            ((6, 2), (7, 2)),
            ((6, 3), (7, 3)),
            # Second horizontal wall
            ((4, 3), (4, 4)),
            ((6, 3), (6, 4)),
            ((7, 3), (7, 4)),
            ((9, 3), (9, 4)),

        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        '''
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame-rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        '''
        self.window = None
        self.clock = None

    # def get_reward_machine(self):
    #     return self.reward_machine

    # def get_agent_reward(self, agent_idx, signal):
    #     return self.agents[agent_idx].temporal_goal.step(signal)

    def get_next_flags(self):
        return self.next_flags

    def _get_obs(self):
        return {
            'agents': {
                'agent_1': self.agents[0].position,
                'agent_2': self.agents[1].position,
                'agent_3': self.agents[2].position
            },
            'doors': {
                'door_1': {
                    'door_position': self._doors_location[0],
                    'door_button': self._doors_button[0],
                    'door_opener': self._doors_opener[0],
                    'door_open_flag': self._doors_flag[0]
                },
                'door_2': {
                    'door_position': self._doors_location[1],
                    'door_button': self._doors_button[1],
                    'door_opener': self._doors_opener[1],
                    'door_open_flag': self._doors_flag[1]
                },
                'door_3': {
                    'door_position': self._doors_location[2],
                    'door_button': self._doors_button[2],
                    'door_opener': self._doors_opener[2],
                    'door_open_flag': self._doors_flag[2]
                }
            },
            'target': self._target_location
        }

    def _get_info(self):
        return {
            'distance': [np.linalg.norm(self.agents[0].position - self._target_location, ord=1),
                         np.linalg.norm(self.agents[1].position - self._target_location, ord=1),
                         np.linalg.norm(self.agents[2].position - self._target_location, ord=1)]
        }

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the reward machine
        # self.reward_machine = Reward_machine.RewardMachine(events=self.events)
        # self.temp_goal.reset()
        for agent in self.agents:
            agent.temporal_goal.reset()

        self.pass_events = []

        # Reset agents position
        self.agents[0].position = Utilities.agents_initial_location[0]
        self.agents[1].position = Utilities.agents_initial_location[1]
        self.agents[2].position = Utilities.agents_initial_location[2]

        # Reset the doors flag to close all them
        self._doors_flag = [1, 1, 1]
        self.events_idx = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, actions):

        # print('environment: step')

        # next_flag = False
        self.next_flags = [False, False, False]

        # Is suppose that only one event can occur at each time
        event = []

        reward = [0.0, 0.0, 0.0]

        opener = [0, 0, 0]

        for agent_idx, action in enumerate(actions):

            # 2% random slip
            # if random.uniform(0, 1) > 0.98:
            #     action = random.randint(0, 3)
            #     # if not self.training:
            #     #     print('agent', agent_idx, 'slip')

            # Map the action (element of {0,1,2,3}) to the direction we walk in
            if action < 4:
                direction = self._action_to_direction[action]

                collision = False

                for door, door_flag in zip(self._doors_location, self._doors_flag):
                    if np.all(self.agents[agent_idx].position + direction == door) and door_flag == 1:
                        # print('door collision')
                        collision = True

                for wall in self._walls:
                    if ((np.all(self.agents[agent_idx].position == wall[0]) and
                         np.all(self.agents[agent_idx].position + direction == wall[1])) or
                            (np.all(self.agents[agent_idx].position == wall[1]) and
                             np.all(self.agents[agent_idx].position + direction == wall[0]))):
                        # print('wall collision')
                        collision = True

                if (
                    self.agents[agent_idx].position[0] + direction[0] > self.size - 1 or
                    self.agents[agent_idx].position[1] + direction[1] > self.size - 1 or
                    self.agents[agent_idx].position[0] + direction[0] < 0 or
                    self.agents[agent_idx].position[1] + direction[1] < 0
                ):
                    # print('boarder collision')
                    collision = True

                # We use `np.clip` to make sure we don't leave the grid
                if not collision:

                    self.agents[agent_idx].position = np.clip(
                        self.agents[agent_idx].position + direction, 0, self.size - 1
                    )

                # else:
                #     reward = [0.0, 0.0, 0.0]
                #     reward[agent_idx] = -1
                #     return self._get_obs(), reward, False, False, self._get_info()

            # check for open door
            elif action == 4:

                if (np.all(self.agents[agent_idx].position == self._doors_button[0]) and
                        self._doors_flag[0] == 1):

                    # print('event press_button_1')

                    opener[0] += 1
                    if opener[0] >= self._doors_opener[0]:
                        event = ['press_button_1']
                        self._doors_flag[0] = 0

                elif (np.all(self.agents[agent_idx].position == self._doors_button[1]) and
                        self._doors_flag[1] == 1):

                    # print('event press_button_2')

                    opener[1] += 1
                    if opener[1] >= self._doors_opener[1]:
                        event = ['press_button_2']
                        self._doors_flag[1] = 0

                elif (np.all(self.agents[agent_idx].position == self._doors_button[2]) and
                      self._doors_flag[2] == 1):  # and self._doors_flag[1] == 0):

                    # print('agent', agent_idx, ' press_button_3')

                    opener[2] += 1
                    if opener[2] >= self._doors_opener[2]:
                        event = ['press_button_3_1', 'press_button_3_2']
                        self._doors_flag[2] = 0
                        # print('agent', agent_idx, event)

                # for door_idx in range(len(self._doors_location)):
                #     if (np.all(self.agents[agent_idx].position == self._doors_button[door_idx]) and
                #             self._doors_flag[door_idx] == 1):
                #
                #         opener[door_idx] += 1
                #
                #         if opener[door_idx] >= self._doors_opener[door_idx]:
                #             # event = self.events[door_idx]
                #             self._doors_flag[door_idx] = 0
                #
                #             if not self.training:
                #                 print('open door')
            # if event:
                # print('environment: event', event)
                # print('environment: agent idx', agent_idx)
                # print('environment: current state', self.agents[agent_idx].temporal_goal.current_state)
                # print(self.agents[agent_idx].temporal_goal._reward_machine)
                # print(dir(self.agents[agent_idx].temporal_goal._reward_machine._automaton))
                # print(self.agents[agent_idx].temporal_goal._reward_machine._automaton._transition_function)
                # self.agents[agent_idx].temporal_goal.step([event])
                # print('individual RM step')

        # target location reach
        if np.array_equal(self.agents[0].position, self._target_location):
            event = [self.events[-1]]
            # print('target press')

        environment_event = False

        # if event:
        #     print('environment: agent event', event)
        #     print('environment: next event', self.events[self.events_idx])

        # if event are None create a random event during training once for step
        if self.training and not event and self.events_idx != len(self.events) - 1:
            random_uniform = random.uniform(0, 1)
            if random_uniform > self.train_transition:
                event = [self.events[self.events_idx]]
                environment_event = True
                # print('environment: next event', event, self.events_idx, len(self.events))

        if (
            not event and
            'press_button_3_1' in self.pass_events and
            'press_button_3_2' in self.pass_events and
            'press_button_3' not in self.pass_events
        ):
            # print('hey')
            event = ['press_button_3']

        # if not self.training and self.render_mode == 'human':
        #     print(self.pass_events, event, self.events[self.events_idx])

        # print(self.events)
        # print(self.events_idx)
        # print(self.events[self.events_idx])
        # print(environment_event)
        if event and self.events[self.events_idx] in event:
            # print('environment: events', event)

            # move on the complete reward machine
            self.events_idx += len(event)
            # self.temp_goal.step([event])

            for agent_idx, agent in enumerate(self.agents):

                common_events = list(
                    set(event) & set(agent.get_events())
                )

                # print('environment: agent events', agent.get_events())
                # print('environment: common_events', common_events)

                if common_events and common_events[0] not in self.pass_events:
                    # print(self.pass_events)
                    # print(agent_idx, common_events)
                    # state, _ = agent.temporal_goal.step(common_events)
                    state, _ = agent.temporal_goal.step(common_events)
                    # print(state)
                    reward[agent_idx] = 1.0
                    self.next_flags[agent_idx] = True
                    # print('environment: next state', state, 'reward', reward)
                    if environment_event:
                        reward[agent_idx] = 0.0
                    # if reward[agent_idx] == 0.0:
                    #     reward[agent_idx] = -0.001
                    # else:
                    #     print('environment:'
                    #           '\nevent', common_events,
                    #           '\nagent', agent_idx,
                    #           '\nreward', reward[agent_idx])

                    # if reward[agent_idx] != 0.0:
                    #     print('environment: event', common_events, 'reward', reward[agent_idx])
                    # else:
                    #     reward[agent_idx] = -0.001

                    # if np.array_equal(event, ['press_button_3_1', 'press_button_3_2']):
                    # print('hey')
                    #     event = ['press_button_3']
                    #     common_events = list(
                    #         set(event) & set(agent.get_events())
                    #     )
                    #     state, _ = agent.temporal_goal.step(common_events)

            for element in event:
                if element not in self.pass_events:
                    self.pass_events.append(element)

        # reward, reward_machine_idx, self.next_flag = self.reward_machine.step(event, self.training)

        # open doors by random action
        if self.training:
            for door_idx in range(len(self._doors_location)):
                if self._doors_flag[door_idx] == 1 and 'press_button_' + str(door_idx + 1) in event:
                    self._doors_flag[door_idx] = 0
                    break

        # An episode is done iff the agent has reached the target
        terminated = (np.array_equal(self.agents[0].position, self._target_location) or
                      np.array_equal(self.agents[1].position, self._target_location) or
                      np.array_equal(self.agents[2].position, self._target_location))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the doors and the buttons
        for door_idx, door_color in zip(range(len(self._doors_location)), self._doors_color):

            # Change color doors when open
            if self._doors_flag[door_idx] == 0:
                door_color = (200, 200, 200)

            # Draw the doors
            pygame.draw.rect(
                canvas,
                door_color,
                pygame.Rect(
                    pix_square_size * self._doors_location[door_idx],
                    (pix_square_size, pix_square_size),
                ),
            )

            # Draw the buttons
            pygame.draw.rect(
                canvas,
                door_color,
                pygame.Rect(
                    pix_square_size * self._doors_button[door_idx] + [int(pix_square_size / 4),
                                                                      int(pix_square_size / 4)],
                    (int(pix_square_size / 2), int(pix_square_size / 2)),
                ),
            )

        # Now we draw the agents
        for agent in self.agents:
            pygame.draw.circle(
                canvas,
                agent.color,
                (agent.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Draw some gridlines for readability
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, pix_square_size * x - 2),
                (self.window_size, pix_square_size * x - 2),
                width=4,
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (pix_square_size * x - 2, 0),
                (pix_square_size * x - 2, self.window_size),
                width=4,
            )

        # Draw the walls
        for wall in self._walls:

            if wall[0][0] == wall[1][0]:
                pygame.draw.line(
                    canvas,
                    0,
                    (wall[0][0] * pix_square_size - 2,
                     wall[0][1] * pix_square_size + pix_square_size - 2),
                    (wall[1][0] * pix_square_size + pix_square_size - 2,
                     wall[1][1] * pix_square_size - 2),
                    width=4,
                )
            elif wall[0][1] == wall[1][1]:
                pygame.draw.line(
                    canvas,
                    0,
                    (wall[0][0] * pix_square_size + pix_square_size - 2,
                     wall[0][1] * pix_square_size - 2),
                    (wall[1][0] * pix_square_size - 2,
                     wall[1][1] * pix_square_size + pix_square_size - 2),
                    width=4,
                )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined frame-rate.
            # The following line will automatically add a delay to keep the frame-rate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
