import random


class RewardMachineNode:

    def __init__(self, event=None, next_node=None):

        self.event = event
        self.next_node = next_node

    def get_event(self):
        return self.event

    def get_next(self):
        return self.next_node


class RewardMachine:

    def __init__(self, events=None):

        self.idx = 0

        nodes = []
        self.nodes = []

        next_node = None

        for event in reversed(events):

            node = RewardMachineNode(event, next_node)
            next_node = node

            nodes.append(node)

        for node in nodes[::-1]:
            self.nodes.append(node)

        # print(self.nodes)

    def get_nodes(self):
        return self.nodes

    def get_node(self, idx):

        assert idx < len(self.nodes)
        return self.nodes[idx]

    def get_next(self, idx):

        assert idx < len(self.nodes)
        return self.nodes[idx].next_node

    def get_idx(self):
        return self.idx

    def step(self, event, train):

        if train and self.nodes[self.idx].event != 'target':
            random_uniform = random.uniform(0, 1)
            if random_uniform > 0.98:  # 0.95:
                self.idx += 1
                # return -0.001, self.idx, True
                return 0., self.idx, True

        if event == self.nodes[self.idx].event:

            # print('reward machine change')
            # print('event:', event)
            self.idx += 1
            return 1., self.idx, True

        # return -0.001, self.idx, False
        return 0., self.idx, False
