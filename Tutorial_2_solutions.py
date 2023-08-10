import sys
import copy
import time


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


class EightPuzzle:

    def __init__(self, squares):
        self.squares = tuple(squares)

        idx = -1
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                idx = i
        self.idx = idx

    def __eq__(self, obj):
        if obj is None:
            return False
        return self.squares == obj.squares

    def __hash__(self):
        return hash(self.squares)

    def move_left(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx-1]
        new_squares[self.idx-1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_right(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx+1]
        new_squares[self.idx+1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_up(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx-3]
        new_squares[self.idx-3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_down(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx+3]
        new_squares[self.idx+3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def get_successors(self):
        successors = []

        if self.idx % 3 > 0:
            successors.append(self.move_left())
        else:
            successors.append(None)

        if self.idx % 3 < 2:
            successors.append(self.move_right())
        else:
            successors.append(None)

        if self.idx // 3 > 0:
            successors.append(self.move_up())
        else:
            successors.append(None)

        if self.idx // 3 < 2:
            successors.append(self.move_down())
        else:
            successors.append(None)

        return successors

    def num_inversions(self):
        total = 0
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                continue
            si = int(self.squares[i])
            for j in range(i, len(self.squares)):
                if self.squares[j] == '_':
                    continue
                sj = int(self.squares[j])
                if si > sj:
                    total += 1
        return total

    def get_parity(self):
        return self.num_inversions() % 2

    def __str__(self):
        s = ""
        for c in self.squares:
            s += c
        return s


class ContainerEntry:
    def __init__(self, puzzle, parent, action_from_parent):
        self.puzzle = puzzle
        self.parent = parent
        self.action_from_parent = action_from_parent

    def get_successors(self):
        s = []
        suc = self.puzzle.get_successors()

        if suc[0] is not None:
            s.append(ContainerEntry(suc[0], self, LEFT))
        if suc[1] is not None:
            s.append(ContainerEntry(suc[1], self, RIGHT))
        if suc[2] is not None:
            s.append(ContainerEntry(suc[2], self, UP))
        if suc[3] is not None:
            s.append(ContainerEntry(suc[3], self, DOWN))

        return s

    def __eq__(self, obj):
        return self.puzzle == obj.puzzle


def bfs(initial, goal):
    container = [ContainerEntry(initial, None, None)]
    visited = set([])

    i = 0
    while len(container) > 0:
        # expand node
        node = container.pop(0)
        if node.puzzle == goal:
            actions = []
            while node.action_from_parent is not None:
                actions.append(node.action_from_parent)
                node = node.parent
            return list(reversed(actions))

        # add successors
        suc = node.get_successors()
        for s in suc:
            if s.puzzle not in visited:
                container.append(s)
                visited.add(s.puzzle)
        i += 1

    return None


def dfs(initial, goal):
    container = [ContainerEntry(initial, None, None)]
    visited = set([])

    i = 0
    while len(container) > 0:
        # expand node
        node = container.pop(-1)
        if node.puzzle == goal:
            actions = []
            while node.action_from_parent is not None:
                actions.append(node.action_from_parent)
                node = node.parent
            return list(reversed(actions))

        # add successors
        suc = node.get_successors()
        for s in suc:
            if s.puzzle not in visited:
                container.append(s)
                visited.add(s.puzzle)
        i += 1
        # if i % 10000 == 0:
        #     print(f'container size: {len(container)}')
        #     print(f'visited size: {len(visited)}')

    return None


def main(arglist):
    # p1 = EightPuzzle("1348627_5")
    # p1 = EightPuzzle("281_43765")
    p1 = EightPuzzle("281463_75")

    p2 = EightPuzzle("1238_4765")

    if p1.get_parity() != p2.get_parity():
        print('No Solution')
        return

    t0 = time.time()
    for _ in range(50):
        actions_bfs = bfs(p1, p2)
    t_bfs = (time.time() - t0) / 50
    num_actions_bfs = len(actions_bfs)

    t0 = time.time()
    for _ in range(1):
        actions_dfs = dfs(p1, p2)
    t_dfs = (time.time() - t0) / 1
    num_actions_dfs = len(actions_dfs)

    print(f'BFS: time = {t_bfs} seconds, #actions = {num_actions_bfs}, actions = {actions_bfs}\n'
          f'DFS: time = {t_dfs} seconds, #actions = {num_actions_dfs}')


if __name__ == '__main__':
    main(sys.argv[1:])

