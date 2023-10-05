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

    def get_state(self):
        return self.player_x, self.player_y

    def apply_move(self, s, a):
        """
        Apply a player move to the map.
        a: {UP, DOWN, LEFT, RIGHT}
        """
        # Handle special cases
        if s in self.rewards.keys():
            # Go to the exit state and collect reward
            next_state = EXIT_STATE
            self.player_x, self.player_y = next_state
            return EXIT_STATE, self.rewards.get(s)
        elif s == EXIT_STATE:
            # Go to a random next state
            self.random_restart()
            next_state = (self.player_x, self.player_y)
            reward = 0
            return next_state, reward

        # Sample which action actually occurs and attempt it
        actions = self.stoch_action(a)
        action = random.choices(list(actions.keys()), list(actions.values()))[0]
        next_state = self.attempt_move(s, action)

        # Update player's location
        self.player_x, self.player_y = next_state

        # Return next state, reward
        return next_state, self.get_reward(next_state)

    
    def random_restart(self):
        """
        Restart the agent in a random map location, avoidng obstacles
        """
        while True:
            new_location = random.randint(0, self.x_size - 1), random.randint(0, self.y_size - 1)
            if new_location not in OBSTACLES:
                self.player_x, self.player_y = new_location
                break


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
        # Using epsilon-greedy
        current_state = self.grid.get_state()
        best_a = self._get_best_action(current_state)
        if best_a is None or random.random() < self.exploit_prob:
            return random.choice(ACTIONS)
        return best_a
    
    def _get_best_action(self, state):
        best_q = float('-inf')
        best_a = None
        for action in ACTIONS:
            this_q = self.q_values.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a

    def next_iteration(self):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """

        # Choose an action, simulate it, and receive a reward
        state = self.grid.get_state()
        action = self.choose_action()
        next_state, reward = self.grid.apply_move(state, action)

        # Update q-value for the (state, action) pair
        old_q = self.q_values.get((state, action), 0)
        best_next = self._get_best_action(next_state)

        best_next_q = self.q_values.get((next_state, best_next), 0)
        if next_state == EXIT_STATE:
            best_next_q = 0
        target = reward + self.grid.discount * best_next_q

        new_q = old_q + self.learning_rate * (target - old_q)
        self.q_values[(state, action)] = new_q

    def print_q_values(self):
        """
        Write a method to print out your agent's q_values here
        """
        for row in range(self.grid.x_size):   
            for col in range(self.grid.y_size):
                print(f'State ({row}, {col}) has Q-values:')
                for action in ACTIONS:
                    print(f'Action {action}: {self.q_values.get(((row, col), action), 0)}')

    def print_policy(self):
        # The states seem to be in (col, row), where (0, 0) is the bottom left corner
        # (0, 2) is the top left corner. Reconstruct to print nicely.
        for y in range(self.grid.y_size):
            row = self.grid.y_size - 1 - y
            print('|', end='')
            for col in range(self.grid.x_size):
                state = (col, row)
                action = ACTIONS_NAMES.get(self._get_best_action(state), ' ')
                if state in self.grid.rewards.keys():
                    action = 'E' # Exit
                print(action + '|', end='')
            print('\n--------')

def run_q_learning(max_iter = 200000,time_limit = 10):
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
    
    ql.print_policy()
    # policy = {}
    # for state in ql.grid.states:
    #     best_a = None
    #     best_q = float('-inf')
    #     for action in ACTIONS:
    #         this_q = ql.q_values.get((state, action))
    #         if this_q is not None and this_q > best_q:
    #             best_q = this_q
    #             best_a = action        
    #     policy[state] = best_a
    # print("Final policy")
    # print(policy)


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
