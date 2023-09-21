import random
import math

UP = 'U'
DOWN = 'D'
LEFT = 'L'
RIGHT = 'R'

ACTIONS = [UP, DOWN, LEFT, RIGHT]

ACTION_DELTAS = {
	UP: (-1, 0),
	DOWN: (1, 0),
	LEFT: (0, -1),
	RIGHT: (0, 1)
}

POLICY_CHARS = {
    UP: '↑',
    DOWN: '↓',
    LEFT: '←',
    RIGHT: '→'
}

class GridWorld:
	def __init__(
		self,
		num_rows: int,
		num_columns: int,
		obstacles: list[tuple[int, int]],
		terminal_states: list[tuple[int, int]],
		rewards: dict[tuple[int, int], float],
		transition_probability: float,
		discount: float
	):
		self.num_rows = num_rows
		self.num_cols = num_columns
		self.obstacles = obstacles
		self.states = [(x, y) for x in range(self.num_rows) for y in range(self.num_cols) if (x, y) not in obstacles]
		self.discount = discount
		self.terminal_states = terminal_states
		self.rewards = rewards
		self.p = transition_probability

	def attempt_move(self, state: tuple[int, int], action: str) -> tuple[int, int]:
		if state in self.terminal_states:
			return state

		x, y = state
		dx, dy = ACTION_DELTAS.get(action)
		new_state = max(min(x + dx, self.num_rows - 1), 0), max(min(y + dy, self.num_cols - 1), 0)

		if new_state in self.obstacles:
			return state

		return new_state
		
	def stoch_action(self, a: str) -> dict[str, float]:
		# Stochastic actions probability distributions
		if a == RIGHT: 
			return {RIGHT: 0.8, UP: 0.1, DOWN: 0.1}
		elif a == UP:
			return {UP: 0.8, LEFT: 0.1, RIGHT: 0.1}
		elif a == LEFT:
			return {LEFT: 0.8, UP: 0.1, DOWN: 0.1}
		elif a == DOWN:
			return {DOWN: 0.8, LEFT: 0.1, RIGHT: 0.1}

	def perform_action(self, state: tuple[int, int], action: str) -> tuple[int, int]:
		actions = list(self.stoch_action(action).items())
		action_chosen = random.choices([item[0] for item in actions], weights=[item[1] for item in actions])[0]
		next_state = self.attempt_move(state, action_chosen)
		return next_state

	def get_reward(self, state: tuple[int, int]) -> float:
		return self.rewards.get(state, 0.0)

class MCTS:
	# Adapted with credit to Nick Collins
	VISITS_PER_SIM = 1
	MAX_ROLLOUT_DEPTH = 200
	TRIALS_PER_ROLLOUT = 1
	EXP_BIAS = 4000 # This is set very high for demo so can easily predict next selected

	def __init__(self, env):
		self.env = env
		self.q_sa = {}
		self.n_s = {}
		self.n_sa = {}

	def selection(self, state):
		""" Given a state, selects the next action based on UCB1. """
		unvisited = []
		for a in ACTIONS:
			if (state, a) not in self.n_sa:
				unvisited.append(a)
		if unvisited:
			# If theres an unvisited, go there to see what it's like
			return random.choice(unvisited)

		# They've all been visited, so pick which one to try again based on UCB1
		best_u = -float('inf')
		best_a = None
		for a in ACTIONS:
			u = self.q_sa.get((state, a), 0) + (self.EXP_BIAS * math.sqrt(math.log(self.n_s.get(state, 0))/self.n_sa.get((state, a), 1)))
			if u > best_u:
				best_u = u
				best_a = a
		return best_a if best_a is not None else random.choice(ACTIONS)

	def simulate(self, initial_state):
		# self.initial_state = initial_state
		visited = {}
		return self.mcts_search(initial_state, 0, visited)

	def plan_online(self, state, iters=10000):
		max_iter = iters
		for i in range(max_iter):
			self.simulate(state)
		return self.mcts_select_action(state)

	def mcts_search(self, state, depth, visited):
		# Check for non-visit conditions
		if (state in visited and visited[state] >= self.VISITS_PER_SIM) or (depth > self.MAX_ROLLOUT_DEPTH):
			# Choose the best Q-value if one exists
			best_q = -float('inf')
			best_a = None
			for a in ACTIONS:
				if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
					best_q = self.q_sa[(state, a)]
					best_a = a
			if best_a is not None:
				return best_q
			else:
				return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
		else:
			visited[state] = visited.get(state, 0) + 1

		# Check for terminal state
		if state in self.env.terminal_states:
			self.n_s[state] = 1
			return self.env.get_reward(state)

		# Check for leaf node:
		if state not in self.n_s:
			# Reached an unexpanded state (i.e. simulation time) so perform rollout from here
			self.n_s[state] = 0
			return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
		else:
			action = self.selection(state)

			# Update counts
			self.n_sa[(state, action)] = self.n_sa.get((state, action), 0) + 1
			self.n_s[state] += 1

			# Execute the selected action and recurse
			new_state = self.env.perform_action(state, action)
			r = self.env.get_reward(new_state) + self.env.discount * self.mcts_search(new_state, depth+1, visited)

			# update node statistics
			if (state, action) not in self.q_sa:
				self.q_sa[(state, action)] = r
			else:
				self.q_sa[(state, action)] = ((self.q_sa[(state, action)] * self.n_sa[(state, action)]) + r) / (self.n_sa[(state, action)] + 1)

			return r

	def mcts_random_rollout(self, state, max_depth, trials):
		total = 0
		s = state
		for i in range(trials):
			d = 0
			while d < max_depth and not s in self.env.terminal_states:
				action = random.choice(ACTIONS)
				new_state = self.env.perform_action(s, action)
				reward = self.env.get_reward(new_state)
				total += (self.env.discount ** (d+1)) * (reward)
				s = new_state
				d += 1
		return total / trials

	def mcts_select_action(self, state):
		best_q = -float('inf')
		best_a = None
		for a in ACTIONS:
			if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
				best_q = self.q_sa[(state, a)]
				best_a = a
		return best_a

	def extract_policy(self):
		policy = {}
		for row in range(self.env.num_rows):
			for col in range(self.env.num_cols):
				state = (row, col)
				action = self.mcts_select_action(state)
				policy[state] = action
		return policy

	def print_policy(self):
		policy = self.extract_policy()
		for row in range(self.env.num_rows):
			for col in range(self.env.num_cols):
				action = policy.get((row, col))
				action = ' ' if action is None else POLICY_CHARS.get(action)
				print(action + ' | ', end='')
			print('\n---------------')

	def __str__(self):
		return str(self.q_sa) + ':' + str(self.n_s) + ':' + str(self.n_sa)

	def __repr__(self):
		return str(self)

# Very simple text-based demo of MCTS running on the specific GridWorld from the tutorial
if __name__ == '__main__':
	# Change the following as you wish to change the environment or the agents starting position
	num_rows = 3
	num_cols = 4
	obstacles = [(1, 1)]
	rewards = {
		(1, 3): -1,
		(0, 3): 1,
	}
	terminal_states = list(rewards.keys())
	current_state = (2, 0) # Agent starting position

	grid = GridWorld(num_rows, num_cols, obstacles, terminal_states, rewards, 0.8, 0.9)
	mcts = MCTS(grid)

	while True:
		print('Agent at', current_state)
		mcts.print_policy()

		move = input("Enter a number of iterations to run, or an action (U, D, L, or R), or Q to quit: ")
		if move == 'Q':
			break
		elif move in ('U', 'D', 'L', 'R'):
			current_state = grid.attempt_move(current_state, move)
		else:
			try:
				num_iters = int(move)
			except ValueError:
				print('Invalid move')
				continue
			mcts.plan_online(current_state, iters=num_iters)
			print('Planned for', num_iters, 'iterations')
