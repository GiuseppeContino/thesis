import numpy as np
import pythomata


size = 10

epsilon = 0.35

learning_rate = 0.7  # 0.8  # 0.7
gamma = 0.95  # 0.9  # 0.95
alpha = 0.9

epochs = 300  # 1000
max_episode_steps = 500

agents = ['agent_1', 'agent_2', 'agent_3']
actions = ['up', 'right', 'down', 'left', 'push_button', 'open_pocket_door']
events = ['open_green_door', 'open_pocket_door_1', 'open_pocket_door_2', 'open_magenta_door', 'open_pocket_door_4',
          'open_blue_door']  # , 'press_target']

agent_1_events = ['open_green_door', 'open_pocket_door_1', 'open_blue_door', 'press_target_1']
agent_2_events = ['open_green_door', 'open_pocket_door_2', 'open_magenta_door', 'open_pocket_door_4',
                  'open_blue_door', 'press_target_2']
agent_3_events = ['open_magenta_door', 'open_pocket_door_3', 'open_pocket_door_4', 'open_blue_door',
                  'press_target_3']

transition_function = {
    'init': {
        'press_button_1': 'door_1',
    },
    'door_1': {
        'press_button_2': 'door_2',
    },
    'door_2': {
        'press_button_3_1': 'door_3_1',
        'press_button_3_2': 'door_3_2',
    },
    'door_3_1': {
        'press_button_3_2': 'door_3',
        'not_press_button_3_1': 'door_2',
        # '~press_button_3_1': 'door_2',
    },
    'door_3_2': {
        'press_button_3_1': 'door_3',
        'not_press_button_3_2': 'door_2',
        # '~ press_button_3_2': 'door_2',
    },
    'door_3': {
        'press_button_3': 'target',
        'not_press_button_3_1': 'door_3_2',
        'not_press_button_3_2': 'door_3_1',
        # '~ press_button_3_1': 'door_3_2',
        # '~ press_button_3_2': 'door_3_1',
    },
    'target': {
        'press_target': 'end_state',
    },
}

alphabet = {'press_button_1', 'press_button_2', 'press_button_3_1', 'not_press_button_3_1', 'press_button_3_2',
            'not_press_button_3_2', 'press_button_3', 'press_target'}
# alphabet = {'press_button_1', 'press_button_2', 'press_button_3_1', 'press_button_3_2', 'press_button_3',
#             'press_target'}

states = {'init', 'door_1', 'door_2', 'door_3', 'door_3_1', 'door_3_2', 'target', 'end_state'}

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

    automaton.add_transition((state_1, "open_pocket_door_1", state_3))
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


# def individual_transition_function(agent_events, initial_state, goal_state):
def individual_transition_function(temp_transition_function, agent_events, initial_state, goal_state):

    agent_transition_function = {k: temp_transition_function[k] for k in temp_transition_function.keys()}

    # delete the unwanted node
    signal = []
    delete_key = []

    # delete all unwanted actions and save state with no actions
    for elem in agent_transition_function.items():
        for item in {k: elem[1][k] for k in elem[1].keys()}:
            if item not in agent_events or item in signal:
                elem[1].pop(item)
            if item not in signal:
                signal.append(item)
        if elem[1] == {}:
            delete_key.append(elem[0])

    # delete the state with no actions
    for elem in delete_key:
        agent_transition_function.pop(elem)

    # compress the states
    change = []
    for elem, next_elem in zip(list(agent_transition_function.items())[:-1],
                               list(agent_transition_function.items())[1:]):
        if elem[1][list(elem[1].keys())[0]] != next_elem[0] and change == []:
            change = [elem[1][list(elem[1].keys())[0]], elem[0], next_elem[0]]

    # force the last node to be a goal state and the first to be the initial state
    list(agent_transition_function.items())[-1][1][
        list(list(agent_transition_function.items())[-1][1].keys())[0]
    ] = goal_state

    agent_transition_function[initial_state] = agent_transition_function.pop(
        list(agent_transition_function.items())[0][0]
    )

    for elem, next_elem in zip(list(agent_transition_function.items())[:-1],
                               list(agent_transition_function.items())[1:]):

        if change != [] and elem[1][list(elem[1].keys())[0]] == change[1]:
            elem[1][list(elem[1].keys())[0]] = change[2]

        if change != [] and elem[0] == change[1]:
            agent_transition_function[change[2]].update(agent_transition_function[change[1]])
            del agent_transition_function[change[1]]

    for elem in list(agent_transition_function.items()):
        if (change != [] and elem[1][list(elem[1].keys())[0]] == change[0] and
                change[0] not in list(agent_transition_function.keys())):
            elem[1][list(elem[1].keys())[0]] = change[2]

    return agent_transition_function


def transition_function_to_symbolic(_transition_function, _states):

    # print('trans_funct', _transition_function)

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_to_idx = {elem: idx for idx, elem in enumerate(list(_states))}

    automaton_state = []

    for _ in _states:

        automaton_state.append(automaton.create_state())

    for elem in _transition_function:
        for item in _transition_function[elem]:

            # print('elem:', elem)
            # print('item:', item)
            # print('aut_state', automaton_state)
            # print('sta_2_idx', state_to_idx)

            automaton.add_transition((
                automaton_state[state_to_idx[elem]],
                item,
                automaton_state[state_to_idx[_transition_function[elem][item]]]
            ))

    automaton.set_initial_state(automaton_state[state_to_idx['init']])
    automaton.set_accepting_state(automaton_state[state_to_idx['end_state']], True)

    return automaton, state_to_idx
