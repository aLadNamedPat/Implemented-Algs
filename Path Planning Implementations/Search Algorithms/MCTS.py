import numpy as np
import math
import random
from copy import deepcopy
import time

#Implemented for a specific use case, so not plug and play
class Node:
    def __init__(
        self, 
        state,
        parent = None,
    ) -> None:
        self.state : gridEnv = state
        self.parent = parent
        self.children : list[Node] = []
        self.times_visited = 0
        self.value = 0
        
    #Fully expanded refers to all of its possible nodes being visited
    def is_fully_expanded(
        self
    ):
        return len(self.children) == len(self.state.get_possible_moves())
    
    def apply_move(
        self,
        action : int,
    ):
        new_env = deepcopy(self.state)
        action = (action // new_env.gridSize, action % new_env.gridSize)
        new_env.false_take_action(action)
        return new_env
    
    # Will return the value of the best child that it has visited
    def best_child(
        self,
        exploration_bias = 1.41,
        final_return = False, # If this is the action that is actually going to be taken
    ) -> float: # 
        choices_weights = [
            child.value / child.times_visited + exploration_bias * math.sqrt(math.log(self.times_visited) / child.times_visited)
            for child in self.children
        ]
        # choices_weights = [
        #     child.value + exploration_bias * math.sqrt(math.log(self.times_visited) / child.times_visited)
        #     for child in self.children
        # ]

        new_env = self.apply_move(self.children[choices_weights.index(max(choices_weights))].state.current_position)
        reward = new_env.get_reward()

        if self.state.lower_bound()[1] > reward and final_return:
            to_return = Node(gridEnv(0, 5, self.state.lower_bound()[0], 0, 0, 0, 0))
        else:
            to_return = self.children[choices_weights.index(max(choices_weights))]

        return to_return, reward
    
class MCTS:
    #Define the MCTS search with the exploration bias that is given to it
    #Using the UCB1 formula which is value + exploration_bias * sqrt(ln(number of visits of parents) / number of times a node has been visited)

    def __init__(
        self, 
        exploration_weight=1.41,
        horizon_steps = 10,
    ) -> None:
        self.exploration_weight = exploration_weight
        self.horizon_steps = horizon_steps

    def search(
        self, 
        initial_state : gridEnv,
        num_iterations : int #Number of total steps downward that the algorithm is going to take
    ) -> Node:
        root = Node(initial_state)

        for _ in range(num_iterations):
            self.layers_deep = 0
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward, average_reward=True)

        # for child in root.children:
        #     print("Node: ", child.state.current_position, child.value, child.times_visited)
        # # # print(root.children.index(root.best_child(exploration_bias=0)))
        # print(root.best_child(exploration_bias=0)[0].state.current_position)
        # print(root.best_child(exploration_bias=0)[0])
        # time.sleep(2)
        return root.best_child(exploration_bias=0, final_return=True)[0]

    # Selects a node from the root that is currently unexplored, or chooses the best action otherwise
    def _select(
        self, 
        node : Node,
    ):
        self.saved_reward = 0
        while not node.state.is_terminal(self.horizon_steps, self.layers_deep):
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                self.layers_deep += 1
                node, reward = node.best_child(self.exploration_weight)
                self.saved_reward = reward
                # print("Current reward:", reward)
                # print("Reward saved:", self.saved_reward)
                # print("Layers deep:", self.layers_deep)
                # print("Node current position:", node.state.current_position)
                # time.sleep(0.5)
        return node

    def _expand(
        self, 
        node : Node
    ) -> Node:
        # An action is labeled as a move here instead of a possible state to travel to
        untried_actions = [action for action in node.state.get_possible_moves() if action not in [child.state.current_position for child in node.children]]
        action = random.choice(untried_actions)
        # Make a new grid environment here where the variant is applied
        next_state = node.apply_move(action)
        child_node = Node(next_state, node)
        node.children.append(child_node)
        return child_node

    def _simulate(
        self, 
        state : gridEnv,
    ) -> float:
        current_state = state
        num_steps_taken = self.layers_deep
        while not current_state.is_terminal(self.horizon_steps, num_steps_taken):
            num_steps_taken += 1
            possible_moves = current_state.get_possible_moves()
            action = random.choice(possible_moves)
            action = (action // current_state.gridSize, action % current_state.gridSize)
            current_state = deepcopy(current_state)
            current_state = current_state.false_take_action(action)
        return current_state.get_reward() + self.saved_reward

    def _backpropagate(
        self, 
        node : Node, 
        reward : float,
        average_reward : bool = True
    ):
        while node is not None:
            node.times_visited += 1
            if average_reward:
                node.value += reward # This needs to be weighted somehow to prevent one side from asymmetrically being biased
            else:
                if reward > node.value:
                # This algorithm could work because the reward is already streamed through the entirety of the rollout
                # Therefore, saving just the highest reward could help in determining which algorithm really is best
                    node.value = reward 
            node.value += reward # This needs to be weighted somehow to prevent one side from asymmetrically being biased
            node = node.parent