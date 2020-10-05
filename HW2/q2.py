from copy import deepcopy as copy
import numpy as np

"""
Name		: Setu Gupta
Roll no.	: 2018190
Question 2 of HW2. Solving for v_pi(s) for gridworld example.
"""


"""
This class has the mapping from every state to all possible actions and next states corresponding to them.
"""
class states_and_actions:

	states = set()		# Set of all states
	action_set = {}		# Dictionary having mapping between a state and possible actions. {state: set(actions)}
	next_states_rewards = {}	# Dictionary having mapping between state action pair and next state and reward pair(s). {(state, action) : [state_reward, ...]}
	state_value = {}	# Dictionary having mapping between state and its state value. {state: v_pi(s)}

	def __init__(self):
		pass

	"""
	Adds a state to the set of states.
	Args:
		state	: State to be added
	Rets:
		None 	
	"""
	def add_state(self, state):
		self.states.add(state)	# Add another state
		self.state_value[state] = 0	# Initialize with 0

	"""
	Adds action to action set of a state.
	Args:
		state				: State of whos action set is to be used
		action				: Action to be added
		next_states_rewards	: [(state, reward), ...] A list of next states
	Rets:
		None 	
	"""
	def add_action(self, state, action, next_states_rewards):
		if state not in self.action_set:
			self.action_set[state] = []

		self.action_set[state].append(action)
		
		state_action_pair = (state, action)
		if state_action_pair not in self.next_states_rewards:
			self.next_states_rewards[state_action_pair] = []

		self.next_states_rewards[state_action_pair].append(next_states_rewards)


	"""
	Resets state values for all states
	"""
	def reset_state_value(self):
		for s in self.states:
			self.state_value[s] = 0

	"""
	Sets state values for a state
	"""
	def set_state_value(self, state, value):
		self.state_value[state] = value


	"""
	Getters below
	"""
	def get_states(self):
		return self.states

	def get_actions(self, state):
		return self.action_set[state]

	def get_next_states_rewards(self, state, action):
		return self.next_states_rewards[(state, action)]

	def get_state_value(self, state):
		return self.state_value[state]



"""
This class holds all the information about the environment.
This class has all the probabilities.
"""
class MDP:
	mdp = {}	# This is a dictionary mapping {(current_state, action, next_state, reward): probability}

	def __init__(self):
		pass

	def set_probability(self, state, action, next_state, reward, probability):
		self.mdp[(state, action, next_state, reward)] = probability

	"""
	Getters below
	"""
	def get_probability(self, state, action, next_state, reward):
		return self.mdp[(state, action, next_state, reward)]

"""
This class defines a policy
A policy is a mapping between a state and all possible actions from that state to probabilities
"""
class policy:
	pi = {}		# A mapping between {(state, action): probability}
	states_and_actions = None	# A reference to states and actions


	"""
	Constructor initializes policy as equiprobable for every state
	"""
	def __init__(self, states_and_actions):

		self.states_and_actions = states_and_actions
		states = states_and_actions.get_states()

		for s in states:
			actions = states_and_actions.get_actions(s)
			
			total_actions = len(actions)
			for a in actions:
				state_action_pair = (s, a)
				self.pi[state_action_pair] = 1/total_actions


	"""
	Greedily sets action for state i.e. pi(a | s) = 1
	"""
	def set_greedy(self, state, action):

		for a in self.states_and_actions.get_actions(state):
			state_action_pair = (state, a)
			self.pi[state_action_pair] = 0

		state_action_pair = (state, action)
		self.pi[state_action_pair] = 1

	"""
	Getters below
	"""
	def get_probability(self, state, action):
		state_action_pair = (state, action)
		return self.pi[state_action_pair]


class policy_evaluator:
	states_and_actions = None
	policy = None
	mdp = None
	discount = 1

	def __init__(self, policy, states_and_actions, mdp, discount):
		self.policy = policy
		self.states_and_actions = states_and_actions
		self.discount = discount
		self.mdp = mdp


	"""
	Evaluates v_pi(s) via solving linear equations 
	NOTE: ONLY WORKS FOR GRIDWORLD
	"""
	def linear(self):
		A = np.zeros((25,25))
		B = np.zeros(25)

		# Construct equations
		for s in self.states_and_actions.get_states():
			s_idx = self.__get_index_from_state(s)
			A[s_idx][s_idx] = 1
			for a in self.states_and_actions.get_actions(s):
				for (ns, r) in self.states_and_actions.get_next_states_rewards(s, a):
					ns_idx = self.__get_index_from_state(ns)
					A[s_idx][ns_idx] -= self.policy.get_probability(s, a) * self.mdp.get_probability(s,a,ns,r) * self.discount
					B[s_idx] += self.policy.get_probability(s, a) * self.mdp.get_probability(s,a,ns,r) * r

		# Solve equations
		solution = np.linalg.solve(A,B)

		# Store results
		for  s_idx in range(len(solution)):
			s = self.__get_state_from_idx(s_idx)
			self.states_and_actions.set_state_value(s, solution[s_idx])

	def __get_index_from_state(self, state):
		return state[0] + state[1]*5
	
	def __get_state_from_idx(self, idx):
		return (idx%5, idx//5)


class policy_improver:
	states_and_actions = None

	def __init__(self, policy, states_and_actions):
		self.policy = policy
		self.states_and_actions = states_and_actions


class gridworld:
	mdp = None	# MDP
	sa = None	# States and actions

	# Action definitions
	NORTH	= 0
	SOUTH	= 1
	EAST	= 2
	WEST	= 3
	
	"""
	Defines the problem for gridworld example of book
	Args:
		None
	Note the naming convention of cells:
		origin at top left.
		x increases towards right
		y increases downwards
	"""
	def __init__(self):
		self.sa = states_and_actions()
		self.mdp = MDP()

		# Add states
		for i in range(25):
			state = (i%5, i//5)
			self.sa.add_state(state)

		# Add actions, next_states, rewards and MDP
		for s in self.sa.get_states():
			for a in range(4):
				ns, r = self.__get_next_state(s, a)
				self.sa.add_action(s, a, (ns, r))
				self.mdp.set_probability(s, a, ns, r, 1)


	"""
	Returns next state and reward given a current action and state.
	"""
	def __get_next_state(self, state, action):

		if(state == (1,0)):	# State A
			next_state = (1,4)	# A'
			reward = 10
			return next_state, reward

		if(state == (3,0)):	# State B
			next_state = (3,2)	# B'
			reward = 5
			return next_state, reward

		next_state = None
		reward = 0

		if(action == self.NORTH):
			next_state = (state[0], state[1] - 1)
		if(action == self.SOUTH):
			next_state = (state[0], state[1] + 1)
		if(action == self.EAST):
			next_state = (state[0] - 1, state[1])
		if(action == self.WEST):
			next_state = (state[0] + 1, state[1])

		if(next_state[0] > 4 or next_state[0] < 0 or next_state[1] > 4 or next_state[1] < 0):	# Boundary
			next_state = state
			reward = -1

		return next_state, reward


	"""
	Solves question 2 of assignment
	"""
	def q2(self):
		p = policy(self.sa)
		pe = policy_evaluator(p, self.sa, self.mdp, 0.9)
		pe.linear()

		# Pretty printing
		for i in range(25):
			state = (i%5, i//5)
			val = round(self.sa.get_state_value(state), 1)
			print(val, "", end = "\t")
			if(i % 5 == 4):
				print()



def main():
	gw = gridworld()
	gw.q2()


if __name__ == '__main__':
	main()