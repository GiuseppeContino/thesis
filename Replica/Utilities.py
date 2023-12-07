import numpy as np
import pythomata

agents = ['agent_1', 'agent_2', 'agent_3']
actions = ['up', 'right', 'down', 'left', 'push_button']
events = ['press_button_1', 'press_button_2', 'press_button_3_1', 'press_button_3_2', 'press_button_3', 'press_target']

agent_1_events = ['press_button_1', 'press_button_3', 'press_target']
agent_2_events = ['press_button_1', 'press_button_2', 'press_button_3_1', 'not_press_button_3_1', 'press_button_3']
agent_3_events = ['press_button_2', 'press_button_3_2', 'not_press_button_3_2', 'press_button_3']

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
    },
    'door_3_2': {
        'press_button_3_1': 'door_3',
        'not_press_button_3_2': 'door_2',
    },
    'door_3': {
        'press_button_3': 'target',
        'not_press_button_3_1': 'door_3_2',
        'not_press_button_3_2': 'door_3_1',
    },
    'target': {
        'press_target': 'end_state',
    },
}

alphabet = {'press_button_1', 'press_button_2', 'press_button_3_1', 'not_press_button_3_1', 'press_button_3_2',
            'not_press_button_3_2', 'press_button_3', 'press_target'}

states = {'init', 'door_1', 'door_2', 'door_3', 'door_3_1', 'door_3_2', 'target', 'end_state'}

agents_initial_location = [
    np.array((0, 0)),
    np.array((4, 0)),
    np.array((7, 0))
]

agents_color = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]


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

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_to_idx = {elem: idx for idx, elem in enumerate(list(_states))}

    automaton_state = []

    for _ in _states:

        automaton_state.append(automaton.create_state())

    for elem in _transition_function:
        for item in _transition_function[elem]:

            automaton.add_transition((
                automaton_state[state_to_idx[elem]],
                item,
                automaton_state[state_to_idx[_transition_function[elem][item]]]
            ))

    automaton.set_initial_state(automaton_state[state_to_idx['init']])
    automaton.set_accepting_state(automaton_state[state_to_idx['end_state']], True)

    return automaton, state_to_idx
