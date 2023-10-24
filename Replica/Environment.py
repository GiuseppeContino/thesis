import random

import gym
from gym import spaces
import pygame
import Reward_machine

import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, events=None, training=False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.events = events
        self.training = training

        self.reward_machine = Reward_machine.RewardMachine(events=self.events)
        self.next_flag = False

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

        # Agents position
        self._agents_location = [
            np.array((0, 0)),
            np.array((4, 0)),
            np.array((7, 0))
        ]

        self._agents_color = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]

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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
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

    def get_reward_machine(self):
        return self.reward_machine

    def get_next_flag(self):
        return self.next_flag

    def _get_obs(self):
        return {
            'agents': {
                'agent_1': self._agents_location[0],
                'agent_2': self._agents_location[1],
                'agent_3': self._agents_location[2]
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
            'distance': [np.linalg.norm(self._agents_location[0] - self._target_location, ord=1),
                         np.linalg.norm(self._agents_location[1] - self._target_location, ord=1),
                         np.linalg.norm(self._agents_location[2] - self._target_location, ord=1)]
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the reward machine
        self.reward_machine = Reward_machine.RewardMachine(events=self.events)

        # Reset agents position
        self._agents_location = [
            np.array((0, 0)),
            np.array((4, 0)),
            np.array((7, 0))
        ]

        # Reset the doors flag to close all them
        self._doors_flag = [1, 1, 1]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, actions):

        # Is suppose that only one event can occur at each time
        event = None

        opener = [0, 0, 0]

        for agent_idx, action in enumerate(actions):

            # 2% random slip
            if random.uniform(0, 1) > 0.98:
                action = random.randint(0, 3)

                # if not self.training:
                #     print('agent', agent_idx, 'slip')

            # Map the action (element of {0,1,2,3}) to the direction we walk in
            if action < 4:
                direction = self._action_to_direction[action]

                collision = False

                for door, door_flag in zip(self._doors_location, self._doors_flag):
                    if np.all(self._agents_location[agent_idx] + direction == door) and door_flag == 1:
                        # print('door collision')
                        collision = True

                for wall in self._walls:
                    if ((np.all(self._agents_location[agent_idx] == wall[0]) and
                         np.all(self._agents_location[agent_idx] + direction == wall[1])) or
                        (np.all(self._agents_location[agent_idx] == wall[1]) and
                         np.all(self._agents_location[agent_idx] + direction == wall[0]))):

                        # print('wall collision')
                        collision = True

                if (self._agents_location[agent_idx][0] + direction[0] > self.size - 1 or
                   self._agents_location[agent_idx][1] + direction[1] > self.size - 1 or
                   self._agents_location[agent_idx][0] + direction[0] < 0 or
                   self._agents_location[agent_idx][1] + direction[1] < 0):

                    # print('boarder collision')
                    collision = True

                # We use `np.clip` to make sure we don't leave the grid
                if not collision:

                    self._agents_location[agent_idx] = np.clip(
                        self._agents_location[agent_idx] + direction, 0, self.size - 1
                    )

                else:
                    return self._get_obs(), -1, False, False, self._get_info()

            elif action == 4:

                for door_idx in range(len(self._doors_location)):
                    if (np.all(self._agents_location[agent_idx] == self._doors_button[door_idx]) and
                            self._doors_flag[door_idx] == 1):

                        opener[door_idx] += 1

                        if opener[door_idx] >= self._doors_opener[door_idx]:

                            event = self.events[door_idx]
                            self._doors_flag[door_idx] = 0

                            # if not self.training:
                            #     print('open door')

        if np.array_equal(self._agents_location[0], self._target_location):
            event = self.events[-1]

        reward, reward_machine_idx, self.next_flag = self.reward_machine.step(event, self.training)

        # open doors by random action
        if self.training and self.next_flag and reward != 1:
            for door_idx in range(len(self._doors_location)):
                if self._doors_flag[door_idx] == 1:
                    self._doors_flag[door_idx] = 0
                    break

        # An episode is done iff the agent has reached the target
        terminated = (np.array_equal(self._agents_location[0], self._target_location) or
                      np.array_equal(self._agents_location[1], self._target_location) or
                      np.array_equal(self._agents_location[2], self._target_location))

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
        for agent, color in zip(self._agents_location, self._agents_color):
            pygame.draw.circle(
                canvas,
                color,
                (agent + 0.5) * pix_square_size,
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
