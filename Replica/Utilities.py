import numpy as np
import pythomata


size = 10

epsilon = 0.35

learning_rate = 0.7  # 0.8  # 0.7
gamma = 0.95  # 0.9  # 0.95
alpha = 0.9

epochs = 1000  # 1000
max_episode_steps = 500
test_num = 1

agents = ['agent_1', 'agent_2', 'agent_3']
actions = ['up', 'right', 'down', 'left', 'push_button', 'open_pocket_door']
events = ['open_green_door', 'open_pocket_door_1', 'open_pocket_door_2', 'open_magenta_door', 'open_pocket_door_4',
          'open_blue_door']

agent_1_events = ['open_green_door', 'open_pocket_door_1', 'open_blue_door', 'press_target_1']
agent_2_events = ['open_green_door', 'open_pocket_door_2', 'open_magenta_door', 'open_pocket_door_4',
                  'open_blue_door', 'press_target_2']
agent_3_events = ['open_magenta_door', 'open_pocket_door_3', 'open_pocket_door_4', 'open_blue_door',
                  'press_target_3']

agents_initial_location = [
    np.array((0, 0)),
    np.array((4, 0)),
    np.array((7, 0)),
]

agents_color = [
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def create_first_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()
    state_6 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_6, True)

    automaton.add_transition((state_0, "open_green_door", state_1))
    automaton.add_transition((state_0, "open_pocket_door_1", state_2))

    automaton.add_transition((state_1, "open_blue_door & ~ open_pocket_door_1", state_4))
    automaton.add_transition((state_4, "open_pocket_door_1", state_5))

    automaton.add_transition((state_1, "open_pocket_door_1 & ~ open_blue_door", state_3))
    automaton.add_transition((state_2, "open_green_door", state_3))

    automaton.add_transition((state_3, "open_blue_door", state_5))

    automaton.add_transition((state_1, "open_pocket_door_1 & open_blue_door", state_5))

    automaton.add_transition((state_5, "press_target_1", state_6))

    return automaton


def create_second_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()
    state_6 = automaton.create_state()
    state_7 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_7, True)

    automaton.add_transition((state_0, "open_pocket_door_2 & ~ open_green_door", state_1))
    automaton.add_transition((state_0, "open_green_door & ~ open_pocket_door_2", state_2))

    automaton.add_transition((state_1, "open_green_door", state_3))
    automaton.add_transition((state_2, "open_pocket_door_2", state_3))

    automaton.add_transition((state_0, "open_green_door & open_pocket_door_2", state_3))

    automaton.add_transition((state_3, "open_magenta_door", state_4))

    automaton.add_transition((state_4, "open_pocket_door_4", state_5))
    automaton.add_transition((state_5, "open_blue_door", state_6))

    automaton.add_transition((state_6, "press_target_2", state_7))

    return automaton


def create_third_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()
    state_6 = automaton.create_state()
    state_7 = automaton.create_state()
    state_8 = automaton.create_state()
    state_9 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_8, True)
    automaton.set_accepting_state(state_9, True)

    automaton.add_transition((state_0, "open_pocket_door_3 & ~ open_magenta_door", state_1))
    automaton.add_transition((state_0, "open_magenta_door & ~ open_pocket_door_3", state_2))

    automaton.add_transition((state_1, "open_magenta_door", state_3))
    automaton.add_transition((state_2, "open_pocket_door_3", state_3))

    automaton.add_transition((state_0, "open_pocket_door_3 & open_magenta_door", state_3))

    automaton.add_transition((state_3, "open_pocket_door_4", state_6))
    automaton.add_transition((state_6, "open_blue_door", state_7))

    automaton.add_transition((state_2, "open_pocket_door_4", state_4))
    automaton.add_transition((state_4, "open_blue_door", state_5))

    automaton.add_transition((state_5, "press_target_3", state_8))
    automaton.add_transition((state_7, "press_target_3", state_9))

    return automaton
