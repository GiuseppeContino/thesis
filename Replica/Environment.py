import random
import copy

import gymnasium as gym
from gymnasium import spaces
import pygame

import Agent
import Utilities

from temprl.reward_machines.automata import RewardAutomaton
from temprl.wrapper import TemporalGoal

import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, events=None, training=False, rewarding_machines=[]):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.events = events
        self.training = training

        self.train_transition = 0.98

        self.next_event = copy.copy(Utilities.events)
        self.pass_events = []

        self.next_flags = [0, 0, 0]

        # agents initial position and colors
        agents_location = Utilities.agents_initial_location

        agents_color = Utilities.agents_color

        self.agents = []

        agents_events = [Utilities.agent_1_events, Utilities.agent_2_events, Utilities.agent_3_events]

        for idx, agent_events in enumerate(agents_events):

            agent_pythomata_rm = rewarding_machines[idx]
            agent_graph = agent_pythomata_rm.to_graphviz()
            agent_graph.render('./images/agent_' + str(idx + 1) + '_reward_machine')

            agent_automata = RewardAutomaton(agent_pythomata_rm, 1)
            agent_temp_goal = TemporalGoal(agent_automata)

            self.agents.append(Agent.Agent(
                agents_location[idx],
                agents_color[idx],
                agent_temp_goal,
                agents_events[idx]
            ))

        self.observation_space = spaces.Dict(
            {
                'agent_1': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                'agent_2': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                'agent_3': spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        # We have 6 actions, corresponding to 'right', 'up', 'left', 'down', 'push_button', 'open_pocket_door'
        self.action_space = spaces.Discrete(len(Utilities.actions))

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Target position
        self._target_location = np.array((3, 4))
        self._targets_location = [
            np.array((9, 9)),
            np.array((9, 4)),
            np.array((3, 4)),
        ]

        self._doors_location = [
            np.array((5, 4)),
            np.array((8, 4)),
            np.array((3, 8)),
        ]
        self._doors_button = [
            np.array((2, 0)),
            np.array((9, 6)),
            np.array((4, 6)),
        ]

        if self.training:
            self._doors_opener = [1, 1, 1]
        else:
            self._doors_opener = [1, 1, 1]

        self.events_idx = 0

        self._doors_flag = [1, 1, 1]

        self._doors_color = [(127, 255, 0), (255, 0, 127), (0, 127, 255)]

        self._pocket_doors_location = [
            np.array((1, 5)),
            np.array((4, 2)),
            np.array((9, 1)),
            np.array((4, 5)),
        ]

        self._pocket_doors_opener_position = [
            np.array((1, 4)),
            np.array((4, 1)),
            np.array((8, 1)),
            np.array((5, 5)),
        ]

        self._pocket_doors_flag = [1, 1, 1, 1]

        self._walls = (
            # First vertical wall
            ((2, 0), (3, 0)),
            ((2, 1), (3, 1)),
            ((2, 2), (3, 2)),
            ((2, 3), (3, 3)),
            ((2, 4), (3, 4)),
            ((2, 5), (3, 5)),
            ((2, 6), (3, 6)),
            ((2, 7), (3, 7)),
            ((2, 9), (3, 9)),
            # First horizontal wall
            ((3, 6), (3, 7)),
            ((4, 6), (4, 7)),
            ((5, 6), (5, 7)),
            ((6, 6), (6, 7)),
            ((7, 6), (7, 7)),
            ((8, 6), (8, 7)),
            ((9, 6), (9, 7)),
            # Second vertical wall
            ((5, 0), (6, 0)),
            ((5, 1), (6, 1)),
            ((5, 2), (6, 2)),
            ((5, 3), (6, 3)),
            # Second horizontal wall
            ((3, 3), (3, 4)),
            ((4, 3), (4, 4)),
            ((6, 3), (6, 4)),
            ((7, 3), (7, 4)),
            ((9, 3), (9, 4)),
            # Third horizontal wall
            ((3, 1), (3, 2)),
            ((5, 1), (5, 2)),
            # Fourth vertical wall
            ((8, 0), (9, 0)),
            ((8, 2), (9, 2)),
            ((8, 3), (9, 3)),
            # Fifth vertical wall
            ((4, 4), (5, 4)),
            ((4, 6), (5, 6)),
            # Second horizontal wall
            ((0, 4), (0, 5)),
            ((2, 4), (2, 5)),
        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def get_next_flags(self):
        return self.next_flags

    def _get_obs(self):
        return {
            'agent_1': self.agents[0].position,
            'agent_2': self.agents[1].position,
            'agent_3': self.agents[2].position
        }

    def _get_info(self):
        return {
            'agent_1': [np.linalg.norm(self.agents[0].position - self._target_location, ord=1)],
            'agent_2': [np.linalg.norm(self.agents[1].position - self._target_location, ord=1)],
            'agent_3': [np.linalg.norm(self.agents[2].position - self._target_location, ord=1)]
        }

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        for idx, agent in enumerate(self.agents):
            agent.temporal_goal.reset()
            self.next_flags[idx] = agent.temporal_goal.current_state - 1

        self.next_event = copy.copy(Utilities.events)
        self.pass_events = []

        # Reset agents position
        self.agents[0].position = Utilities.agents_initial_location[0]
        self.agents[1].position = Utilities.agents_initial_location[1]
        self.agents[2].position = Utilities.agents_initial_location[2]

        # Reset the doors flag to close all them
        self._doors_flag = [1, 1, 1]
        self._pocket_doors_flag = [1, 1, 1, 1]
        self.events_idx = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, actions):

        event = []

        reward = [0.0, 0.0, 0.0]

        openers = [0, 0, 0]

        for agent_idx, action in enumerate(list(actions.values())):

            # Map the action (element of {0,1,2,3}) to the direction we walk in
            if action < 4:
                direction = self._action_to_direction[action]

                collision = False

                for door, door_flag in zip(self._doors_location, self._doors_flag):
                    if np.all(self.agents[agent_idx].position + direction == door) and door_flag == 1:
                        collision = True

                for pocket_door, pocket_door_flag in zip(self._pocket_doors_location, self._pocket_doors_flag):
                    if np.all(self.agents[agent_idx].position + direction == pocket_door) and pocket_door_flag == 1:
                        collision = True

                for wall in self._walls:
                    if ((np.all(self.agents[agent_idx].position == wall[0]) and
                         np.all(self.agents[agent_idx].position + direction == wall[1])) or
                            (np.all(self.agents[agent_idx].position == wall[1]) and
                             np.all(self.agents[agent_idx].position + direction == wall[0]))):
                        collision = True

                if (
                    self.agents[agent_idx].position[0] + direction[0] > self.size - 1 or
                    self.agents[agent_idx].position[1] + direction[1] > self.size - 1 or
                    self.agents[agent_idx].position[0] + direction[0] < 0 or
                    self.agents[agent_idx].position[1] + direction[1] < 0
                ):
                    collision = True

                # We use `np.clip` to make sure we don't leave the grid
                if not collision:

                    self.agents[agent_idx].position = np.clip(
                        self.agents[agent_idx].position + direction, 0, self.size - 1
                    )

            # check for push button to open door
            elif action == 4:

                if (np.all(self.agents[agent_idx].position == self._doors_button[0]) and
                        self._doors_flag[0] == 1):

                    openers[0] += 1
                    if openers[0] >= self._doors_opener[0]:
                        event.append('open_green_door')
                        reward[agent_idx] = 1.0

                elif (np.all(self.agents[agent_idx].position == self._doors_button[1]) and
                        self._doors_flag[1] == 1):

                    openers[1] += 1
                    if openers[1] >= self._doors_opener[1]:
                        event.append('open_magenta_door')
                        reward[agent_idx] = 1.0

                elif (np.all(self.agents[agent_idx].position == self._doors_button[2]) and
                      self._doors_flag[2] == 1):  # and self._doors_flag[1] == 0):

                    openers[2] += 1
                    if openers[2] >= self._doors_opener[2]:
                        event.append('open_blue_door')
                        reward[agent_idx] = 1.0

            # check for open door
            elif action == 5:

                if (np.all(self.agents[agent_idx].position == self._pocket_doors_opener_position[0]) and
                        self._pocket_doors_flag[0] == 1):

                    event.append('open_pocket_door_1')
                    reward[agent_idx] = 1.0

                elif (np.all(self.agents[agent_idx].position == self._pocket_doors_opener_position[1]) and
                        self._pocket_doors_flag[1] == 1):

                    event.append('open_pocket_door_2')
                    reward[agent_idx] = 1.0

                elif (np.all(self.agents[agent_idx].position == self._pocket_doors_opener_position[2]) and
                        self._pocket_doors_flag[2] == 1 and self._doors_flag[1] == 1):

                    event.append('open_pocket_door_3')
                    reward[agent_idx] = 1.0

                elif (np.all(self.agents[agent_idx].position == self._pocket_doors_opener_position[3]) and
                        self._pocket_doors_flag[3] == 1 and self._doors_flag[1] == 0):

                    event.append('open_pocket_door_4')
                    reward[agent_idx] = 1.0

            agent_on_target = []

            # target location reach
            if not self.next_event:
                if self.training:
                    for target_location in self._targets_location:
                        if self.training and np.array_equal(self.agents[agent_idx].position, target_location):
                            event = ['press_target_' + str(agent_idx + 1)]
                            reward[agent_idx] = 1.0
                else:
                    for target_location in self._targets_location:
                        if self.training and np.array_equal(self.agents[agent_idx].position, target_location):
                            # event = ['press_target']
                            # reward[agent_idx] = 1.0
                            pass

                        elif np.array_equal(self.agents[agent_idx].position, target_location):
                            if target_location not in agent_on_target:
                                agent_on_target.append(agent_idx)

                    if len(agent_on_target) == len(self.agents):
                        event.append('press_target_' + str(agent_idx + 1))
                        reward[agent_idx] = 1.0

        # if event are None create a random event during training once for step
        if self.training and not event:
            random_uniform = random.uniform(0, 1)
            if random_uniform > self.train_transition and self.next_event:
                event = [self.next_event.pop(0)]

        # from the event to the correlated action
        if 'open_green_door' in event:
            self._doors_flag[0] = 0
        elif 'open_magenta_door' in event:
            self._doors_flag[1] = 0
        elif 'open_blue_door' in event:
            self._doors_flag[2] = 0
        elif 'open_pocket_door_1' in event:
            self._pocket_doors_flag[0] = 0
        elif 'open_pocket_door_2' in event:
            self._pocket_doors_flag[1] = 0
        elif 'open_pocket_door_3' in event:
            self._pocket_doors_flag[2] = 0
        elif 'open_pocket_door_4' in event:
            self._pocket_doors_flag[3] = 0

        if event and self.render_mode == 'human':
            print(event)

        if event:

            # remove from the next_event the pass event
            for item in event:
                if item in self.next_event:
                    self.next_event.remove(item)

            # step on the agent individual reward machine
            for agent_idx, agent in enumerate(self.agents):

                common_events = list(
                    set(event) & set(agent.get_events())
                )

                # print(agent_idx, agent.temporal_goal.current_state, event, common_events, self.next_event)

                if common_events and common_events[0] not in self.pass_events:

                    agent.temporal_goal.step(common_events)
                    self.next_flags[agent_idx] = agent.temporal_goal.current_state - 1

            # update the pass event
            for element in event:
                if element not in self.pass_events:
                    self.pass_events.append(element)

        # compute the reward dict
        reward_dict = {'agent_' + str(key + 1): value for key, value in enumerate(reward)}

        # compute the termination dict
        terminated = [
            (np.array_equal(self.agents[0].position, self._targets_location[0]) or
             np.array_equal(self.agents[0].position, self._targets_location[1]) or
             np.array_equal(self.agents[0].position, self._targets_location[2])),
            (np.array_equal(self.agents[1].position, self._targets_location[0]) or
             np.array_equal(self.agents[1].position, self._targets_location[1]) or
             np.array_equal(self.agents[1].position, self._targets_location[2])),
            (np.array_equal(self.agents[2].position, self._targets_location[0]) or
             np.array_equal(self.agents[2].position, self._targets_location[1]) or
             np.array_equal(self.agents[2].position, self._targets_location[2]))
        ]
        terminated_dict = {'agent_' + str(key + 1): value for key, value in enumerate(terminated)}

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward_dict, terminated_dict, False, info

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
        for target_location in self._targets_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
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

        for door_idx, door in enumerate(self._pocket_doors_location):

            door_color = (255, 127, 0)

            if self._pocket_doors_flag[door_idx] == 0:
                door_color = (200, 200, 200)

            # Draw the doors
            pygame.draw.rect(
                canvas,
                door_color,
                pygame.Rect(
                    pix_square_size * door,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agents
        for agent in self.agents:
            pygame.draw.circle(
                canvas,
                agent.color,
                (np.add(agent.position, 0.5)) * pix_square_size,
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
