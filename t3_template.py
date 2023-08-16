from __future__ import annotations
import sys
import heapq
import time

"""
COMP3702 2022
Tutorial 3 Template

Last updated by njc 14/08/22
"""


class GridWorldEnv:

    # GridWorldState = (row, col) tuple

    ACTIONS = ['U', 'D', 'L', 'R']

    def __init__(self):
        self.n_rows = 9
        self.n_cols = 9

        # indexing is top to bottom, left to right (matrix indexing)
        init_r = 8
        init_c = 0
        self.initial = (init_r, init_c)
        goal_r = 0
        goal_c = 8
        self.goal = (goal_r, goal_c)

        self.obstacles = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.costs = [[1, 1,  1,  5,  5,  5,  5, 1, 1],
                      [1, 1,  1,  5,  5,  5,  5, 1, 1],
                      [1, 1, 10, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1, 10, 10, 1, 1],
                      [1, 1,  1,  1, 10, 10, 10, 1, 1],
                      [1, 1,  1, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1,  1,  1, 1, 1]]

    def perform_action(self, state, action):
        """
        :param state: (row, col) tuple
        :param action: 'U', 'D', 'L' or 'R'
        :return: (success [True/False], new state, action cost)
        """
        r, c = state

        if action == 'U':
            new_r = r - 1
            new_c = c
        elif action == 'D':
            new_r = r + 1
            new_c = c
        elif action == 'L':
            new_r = r
            new_c = c - 1
        elif action == 'R':
            new_r = r
            new_c = c + 1
        else:
            assert False, '!!! invalid action !!!'

        if (not (0 <= new_r < 9)) or (not (0 <= new_c < 9)) or self.obstacles[new_r][new_c] == 1:
            # collision occurs
            return False, self.costs[r][c], (r, c)
        else:
            return True, self.costs[new_r][new_c], (new_r, new_c)

    def is_solved(self, state):
        """
        :param state: (row, col) tuple
        :return: True/False
        """
        return state == self.goal

    def get_state_cost(self, state):
        r, c = state
        return self.costs[r][c]


class EightPuzzleState:

    def __init__(self, squares):
        """
        :param squares: a list where each element is in {'1', 2', ... '8', '_'}
        """
        self.squares = squares      # make sure squares is a deep copy

        # store index of blank space to improve performance
        idx = -1
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                idx = i
        self.idx = idx

    def __hash__(self):
        # required to allow this object to be placed inside a hashtable (i.e. set/dict)
        return hash(tuple(self.squares))

    def __eq__(self, other: EightPuzzleState):
        # required to test if this object is in a collection
        return tuple(self.squares) == tuple(other.squares)


class EightPuzzleEnv:

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ACTIONS = [LEFT, RIGHT, UP, DOWN]

    STATE_COST = 1

    def __init__(self, initial: EightPuzzleState, goal: EightPuzzleState):
        """
        :param initial: initial puzzle state
        :param goal: goal puzzle state
        """
        self.n_rows = 3
        self.n_cols = 3

        self.initial = initial
        self.goal = goal

    def is_solved(self, state: EightPuzzleState):
        """
        :param state: current state
        :return: True if solved
        """
        if state == self.goal:
            return True

    def perform_action(self, state: EightPuzzleState, action):
        """
        :param state: current state
        :param action: action to perform
        :return: (successful [True/False], cost [float], next_state [EightPuzzleState])
        """
        # screen invalid actions
        if action == self.LEFT and not state.idx % self.n_cols > 0:
            return False, None, None
        elif action == self.RIGHT and not state.idx % self.n_cols < 2:
            return False, None, None
        elif action == self.UP and not state.idx // self.n_rows > 0:
            return False, None, None
        elif action == self.DOWN and not state.idx // self.n_rows < 2:
            return False, None, None

        # make substitution
        new_squares = state.squares[:]  # deep copy
        swap_idx = state.idx + {self.LEFT: -1, self.RIGHT: 1, self.UP: -3, self.DOWN: 3}[action]
        new_squares[state.idx] = state.squares[swap_idx]
        new_squares[swap_idx] = state.squares[state.idx]

        return True, 1, EightPuzzleState(new_squares)

    def render(self, state):
        print(('+---' * self.n_cols) + '+')
        for r in range(self.n_rows):
            line = '|'
            for c in range(self.n_cols):
                line += ' ' + state.squares[c] + ' |'
            print(line)

    def get_state_cost(self, _):
        # same cost for all states in EightPuzzle
        return self.STATE_COST


class StateNode:

    def __init__(self, env, state, parent, action_from_parent, path_steps, path_cost):
        """
        :param env: environment
        :param state: state belonging to this node
        :param parent: parent of this node
        :param action_from_parent: LEFT, RIGHT, UP, or DOWN
        """
        self.env = env
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.path_steps = path_steps
        self.path_cost = path_cost

    def get_path(self):
        """
        :return: A list of actions
        """
        path = []
        cur = self
        while cur.action_from_parent is not None:
            path.append(cur.action_from_parent)
            cur = cur.parent
        path.reverse()
        return path

    def get_successors(self):
        """
        :return: A list of successor StateNodes
        """
        successors = []
        for a in self.env.ACTIONS:
            success, cost, next_state = self.env.perform_action(self.state, a)
            if success:
                successors.append(StateNode(self.env, next_state, self, a, self.path_steps + 1, self.path_cost + cost))
        return successors

    def __lt__(self, other):
        # we won't use this as a priority directly, so result doesn't matter
        return True


def bfs(env, verbose=True):
    container = [StateNode(env, env.initial, None, None, 0, 0)]
    visited = set()

    n_expanded = 0
    while len(container) > 0:
        # expand node
        node = container.pop(0)

        # test for goal
        if env.is_solved(node.state):
            if verbose:
                print(f'Visited Nodes: {len(visited)},\t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Container: {len(container)}')
                print(f'Cost of Path (with Costly Moves): {node.path_cost}')
            return node.get_path()

        # add successors
        successors = node.get_successors()
        for s in successors:
            if s.state not in visited:
                container.append(s)
                visited.add(s.state)
        n_expanded += 1

    return None


def depth_limited_dfs(env, max_depth, verbose=True):
    #
    #
    # TODO: implement your Depth Limited DFS (Ex 3.1c) here
    #
    #
    pass


def iddfs(env, verbose=True):
    #
    #
    # TODO: implement your Iterative Deepening DFS (Ex 3.1c) here
    #
    #
    pass


def ucs(env, verbose=True):
    #
    #
    # TODO: implement your Uniform Cost Search (Ex 3.2b) here
    #
    #
    pass


def heuristic_g1(env, state):
    #
    #
    # TODO: implement your heuristic for Grid World (Ex 3.2d) here
    #
    #
    pass


def heuristic_e1(env, state):
    #
    #
    # TODO: implement your 1st heuristic for 8-Puzzle (Ex 3.3c) here
    #
    #
    pass


def heuristic_e2(env, state):
    #
    #
    # TODO: (optionally) implement your 1st heuristic for 8-Puzzle (Ex 3.3c) here
    #
    #
    pass


def a_star(env, heuristic, verbose=True):
    #
    #
    # TODO: implement your A* Search (Ex 3.2e) here
    #
    #
    pass


def main(arglist):
    # TODO: uncomment the section for the current exercise below
    n_trials = 100
    print('== Exercise 3.1 ==============================================================================')
    gridworld = GridWorldEnv()

    print('BFS:')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_bfs = bfs(gridworld, verbose=(i == 0))
    # t_bfs = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_bfs)},\t\tActions: {actions_bfs}')
    # print(f'Time: {t_bfs}')
    # print('\n')

    print('IDDFS:')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_iddfs = iddfs(gridworld, verbose=(i == 0))
    # t_iddfs = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_iddfs)},\t\tActions: {actions_iddfs}')
    # print(f'Time: {t_iddfs}')
    # print('\n')

    print('== Exercise 3.2 ==============================================================================')
    print('UCS:')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_ucs = ucs(gridworld, verbose=(i == 0))
    # t_ucs = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_ucs)},\t\tActions: {actions_ucs}')
    # print(f'Time: {t_ucs}')
    # print('\n')

    print('A*:')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_a_star = a_star(gridworld, manhattan_dist_heuristic, verbose=(i == 0))
    # t_a_star = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    # print(f'Time: {t_a_star}')
    # print('\n')

    print('== Exercise 3.3 ==============================================================================')
    puzzle = EightPuzzleEnv(EightPuzzleState(list('281463_75')), EightPuzzleState(list('1238_4765')))

    print('BFS:')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_bfs = bfs(puzzle, verbose=(i == 0))
    # t_bfs = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_bfs)},\t\tActions: {actions_bfs}')
    # print(f'Time: {t_bfs}')
    # print('\n')

    print('A* (Heuristic 1):')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_a_star = a_star(puzzle, heuristic_e1, verbose=(i == 0))
    # t_a_star = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    # print(f'Time: {t_a_star}')
    # print('\n')

    print('A* (Heuristic 2):')
    # t0 = time.time()
    # for i in range(n_trials):
    #     actions_a_star = a_star(puzzle, heuristic_e2, verbose=(i == 0))
    # t_a_star = (time.time() - t0) / n_trials
    # print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    # print(f'Time: {t_a_star}')
    # print('\n')


def show_visited(env, visited):
    for r in range(9):
        line = ''
        for c in range(9):
            if env.obstacles[r][c] == 1:
                line += 'X'
            elif (r, c) in visited:
                line += '1'
            else:
                line += '0'
        print(line)


if __name__ == '__main__':
    main(sys.argv[1:])



