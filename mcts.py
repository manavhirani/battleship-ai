import numpy as np
from flags import *

sizes = [3,4,4]

class Node:
    def __init__(self, state) -> None:
        self.state = state
        self.visit_count = 0
        self.score_total = 0
        self.score_estimate = 0
        self.child_list = None
        self.player = True
    
    def children(self):
        if self.child_list == None:
            self.child_list = list(map(Node, children_of(self.state)))
            # for child in self.child_list:
            #     child.player = not self.player
        return self.child_list
    
    def N_values(self):
        return [child.visit_count for child in self.children()]
    
    def Q_values(self):
        children = self.children()
        # TODO:
        
        sign = 1 if self.player else -1
        Q = [1 * child.score_total/(child.visit_count+1) for child in children]
        return Q

def children_of(state):
    # print('Finding children of\n', state)
    children = []
    hits = state
    size = len(hits)
    for row in range(size):
        for col in range(size):
            if hits[row][col] == 0:
                child = hits.copy()
                child[row][col] = 1
                children.append(child)
    return children

def is_leaf(state):
    children = children_of(state)
    value = np.count_nonzero(state == 0)
    # if value == 0:
    #     # print('Found leaf\n', state)
    return value == 0

def rollout(node, sizes):
    if is_leaf(node.state):
        result = score(node.state, sizes)
    else:
        result = rollout(explore(node), sizes)
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    return result

def exploit(node):
    return node.children()[np.argmax(node.Q_values())]

def explore(node):
    return node.children()[np.argmin(node.N_values())]

def uct(node):
    Q = np.array(node.Q_values())
    N = np.array(node.N_values())
    U = Q + np.sqrt(np.log(node.visit_count + 1) / (N + 1))
    return node.children()[np.argmax(U)]

def score(state, sizes):
    # TODO:
    # Vertical orientation
    # hits = state
    # positions = []
    # for size in sizes:
    #     positions += possible_ship_positions(hits, size)
    # return len(positions)/len(hits)**2
    # return 1 if len(positions) == 0 else -1
    s = 0
    for row in range(len(state)):
        for col in range(len(state)):
            if state[row][col] == 1:
                dirs = []
                if row - 1 >= 0:
                    dirs.append((row - 1,col))
                if col - 1 >= 0:
                    dirs.append((row, col-1))
                if row + 1 < len(state):
                    dirs.append((row + 1, col))
                if col + 1 < len(state):
                    dirs.append((row, col + 1))
                s += 1
                for d in dirs:
                    x, y = d
                    if state[x][y] == 1:
                        s += 1
    return s

def possible_ship_positions(hits, size):
    s = len(hits)
    positions = []
    for i in range(s):
        for j in range(s-size):
            locs = [x for x in hits[i][j:j+size]]
            if 2 not in locs:
                positions.append((i,j))
    for i in range(s-size):
        for j in range(s):
            locs = [x[j] for x in hits[i:i+size]]
            if 2 not in locs:
                positions.append((i,j))
    return positions