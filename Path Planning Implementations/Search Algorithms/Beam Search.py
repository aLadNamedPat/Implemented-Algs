import numpy as np
from copy import deepcopy

#Implemented for a specific use case, so not plug and play
class Node:
    def __init__(
        self,
        value,
        state : gridEnv,
        parent = None
    ):
        self.state = state        
        self.dps = value
        self.parent = parent    

    def apply_move(
        self,
        action : int,
    ):
        new_env = deepcopy(self.state)
        action = (action // new_env.gridSize, action % new_env.gridSize)
        new_env.false_take_action(action)
        return new_env

class BeamSearch:
    
    def __init__(
        self,
        width,
        depth,
    ):
        self.width = width
        self.depth = depth
        self.paths : list[list[Node]] = []

    def select_children(
        self,
        current_node : Node = None
    ):
        next_action_list : list[Node] = []
        possible_moves = current_node.state.get_possible_moves()

        for move in possible_moves:
            new_env = current_node.apply_move(move)
            next_action_list.append(Node(new_env.accumulated_reward / new_env.total_dist_traveled, new_env, current_node))

        next_action_list = sorted(next_action_list, key = lambda node: node.dps, reverse = True)
        return next_action_list[:self.width]

    def search(
        self,
        initial_state : gridEnv
    ):
        root = Node(0, initial_state, parent = None)
        beam = [root]

        for m in range(self.depth):
            all_children : list[Node] = []
            for node in beam:
                all_children.extend(self.select_children(node))            
            
            all_children = sorted(all_children, key = lambda node: node.dps, reverse = True)
            
            beam = all_children[:self.width]

        beam = sorted(beam, key = lambda node: node.dps, reverse=True)
        furthest_parent = self.get_furthest_parent(beam[0])
        return furthest_parent.state.location
    
    def get_furthest_parent(
        self,
        node : Node,
    ):
        if node.parent.parent == None:
            return node
        else:
            return self.get_furthest_parent(node.parent)