import copy
import numpy as np
import random
import time

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
EXIT = -1

ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}

OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)

def dict_argmax(d):   
    return  max(d, key=d.get)
        
class Grid:

    def __init__(self):
        self.x_size = 4
        self.y_size = 3
        self.p = 0.8
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.rewards = {(3, 1): -100, (3, 2): 1}
        self.discount = 0.9 # 'gamma' in lecture notes

        self.states = list((x, y) for x in range(self.x_size) for y in range(self.y_size))
        self.states.append(EXIT_STATE)
        for obstacle in OBSTACLES:
            self.states.remove(obstacle)
            
        # New variables added for Tutorial 11
        self.obstacles = OBSTACLES
        self.player_x = 0
        self.player_y = 0

    def attempt_move(self, s, a):
        # s: (x, y), x = s[0], y = s[1]
        # a: {UP, DOWN, LEFT, RIGHT}

        x, y = s[0], s[1]

        # Check absorbing state
        if s in self.rewards:
            return EXIT_STATE

        if s == EXIT_STATE:
            return s

        # Default: no movement
        result = s 

        # Check borders
        if a == LEFT and x > 0:
            result = (x - 1, y)
        if a == RIGHT and x < self.x_size - 1:
            result = (x + 1, y)
        if a == UP and y < self.y_size - 1:
            result = (x, y + 1)
        if a == DOWN and y > 0:
            result = (x, y - 1)

        # Check obstacle cells
        if result in OBSTACLES:
            return s

        return result

    def stoch_action(self, a):
        # Stochastic actions probability distributions
        if a == RIGHT: 
            stoch_a = {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == UP:
            stoch_a = {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        if a == LEFT:
            stoch_a = {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == DOWN:
            stoch_a = {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        return stoch_a

    def get_reward(self, s):
        if s == EXIT_STATE:
            return 0

        if s in self.rewards:
            return self.rewards[s]
        else:
            return 0
        
    """
    Write your code for apply_move() and random_restart(), as well as Q-learning below
    """      
    
    def apply_move(self, s, a):
        """
        Apply a player move to the map.
        a: {UP, DOWN, LEFT, RIGHT}
        """

    
    def random_restart(self):
        """
        Restart the agent in a random map location, avoidng obstacles
        """


class Q_Learning:
    def __init__(self, grid):
        self.grid = grid
        self.learning_rate = 0.05   # 'alpha' in lecture notes
        self.exploit_prob = 0.8     # 'epsilon' in epsilon-greedy
        self.q_values = {state: 0 for state in self.grid.states}            

    def choose_action(self):
        """
        Write a method to choose an action here
        Incorporate your agent's exploration strategy in this method
        """        
               
    def next_iteration(self):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """

    def print_q_values(self):
        """
        Write a method to print out your agent's q_values here
        """


def run_q_learning(max_iter = 20000,time_limit = 10):
    grid = Grid()
    ql = Q_Learning(grid)

    start = time.time()
    print("Q_learning")
    print()

    i = 0
    while i < max_iter and (time.time() - start) < time_limit:
        i = i + 1
        ql.next_iteration()
        if i%1000 == 0 :
            print("Q-values at iteration", i)
            print(ql.print_q_values())
            print()
        
    end = time.time()
    print("Time to complete", i+1, "Q-learning iterations")
    print(end - start)
    print()
    
    policy = {}
    for state in ql.grid.states:
        best_a = None
        best_q = float('-inf')
        for action in ACTIONS:
            this_q = ql.q_values.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action        
        policy[state] = best_a
    print("Final policy")
    print(policy)


class ValueIteration:
    """
    Value iteration, for benchmarking RL algorithm solutions
    """             
    def __init__(self, grid):
        self.grid = grid
        self.values = {state: 0 for state in self.grid.states}

    def next_iteration(self):
        new_values = dict()
        for s in self.grid.states:
            # Maximum value
            action_values = list()
            for a in self.grid.actions:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values.append(total)
            # Update state value with maximum
            new_values[s] = max(action_values)

        self.values = new_values

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)


def run_value_iteration(max_iter = 100):
    grid = Grid()
    vi = ValueIteration(grid)

    start = time.time()
    print("Value iteration")
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(max_iter):
        vi.next_iteration()
        if i%10 == 9:
            print("Values after iteration", i + 1)
            vi.print_values()
            print()

    end = time.time()
    print("Time to complete", max_iter, "VI iterations")
    print(end - start)
    print()
    


if __name__ == "__main__":
    run_value_iteration()
    run_q_learning()

