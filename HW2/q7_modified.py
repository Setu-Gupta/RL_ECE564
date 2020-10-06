from copy import deepcopy as copy
import numpy as np
from math import inf, factorial, e

"""
Name		: Setu Gupta
Roll no.	: 2018190
Question 4 of HW2. Solving for v_*(s) and pi_*(s) for gridworld example.
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

		self.next_states_rewards[state_action_pair].extend(next_states_rewards)


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
	greedily_set = False	# This variable is true if the policy is greedily set

	GREEDY_COMPARE_THRESH = 1e-2	# Comparing threshold. If probability is greater than this, then we assume that it's non zero

	"""
	Constructor initializes policy as equiprobable for every state
	"""
	def __init__(self, states_and_actions):

		self.states_and_actions = states_and_actions
		states = states_and_actions.get_states()

		for s in states:
			actions = states_and_actions.get_actions(s)
			
			total_actions = len(actions)
			first_action = True
			for a in actions:
				state_action_pair = (s, a)
				self.pi[state_action_pair] = 1 if first_action else 0
				first_action = False


	"""
	Greedily sets action for state i.e. pi(a | s) = 1
	It returns true if the policy was greedily set and didn't change
	"""
	def set_greedy(self, state, actions):

		# Find the set of greedy actions
		old_greedy_actions = []
		if(self.greedily_set):
			for a in self.states_and_actions.get_actions(state):
				state_action_pair = (state, a)
				if(self.pi[state_action_pair] >= self.GREEDY_COMPARE_THRESH):
					old_greedy_actions.append(a)

		for a in self.states_and_actions.get_actions(state):
			state_action_pair = (state, a)
			self.pi[state_action_pair] = 0

		for a in actions:
			state_action_pair = (state, a)
			self.pi[state_action_pair] = 1/len(actions)

		if(sorted(actions) == sorted(old_greedy_actions) and self.greedily_set):	# This ensures that that true is is returned only if old and new policy were identical 
			return True

		self.greedily_set = True
		return False

	"""
	Getters below
	"""
	def get_probability(self, state, action):
		state_action_pair = (state, action)
		return self.pi[state_action_pair]

	def get_action(self, state):
		prob = []	# Probability distribution
		actions = []
		for a in states_and_actions.get_action(state):
			actions.append(a)
			prob.append(self.pi[(state, a)])

		return np.random.choice(actions, p = prob)

	"""
	Returns a list of all greeily chosen actions for state
	"""
	def get_all_greedy_actions(self, state):
		if(self.greedily_set):
			greedy_actions = []
			for a in self.states_and_actions.get_actions(state):
				state_action_pair = (state, a)
				if(self.pi[state_action_pair] >= self.GREEDY_COMPARE_THRESH):
					greedy_actions.append(a)
			return greedy_actions
		return []


class policy_evaluator:
	states_and_actions = None
	policy = None
	mdp = None
	discount = 1

	def __init__(self, policy, states_and_actions, mdp, discount):
		self.policy = policy
		self.states_and_actions = states_and_actions
		self.mdp = mdp
		self.discount = discount

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

	"""
	Estimates v_pi(s) iteratively
	Args:
		theta	: Error bound for comparision and breaking
	"""
	def iterative(self, theta):
		iteration_count = 0
		delta = inf
		while(delta > theta):
			iteration_count += 1
			delta = 0
			for s in self.states_and_actions.get_states():
				old_state_value = self.states_and_actions.get_state_value(s)
				
				# Calculate new state value
				new_state_value = 0
				for a in self.states_and_actions.get_actions(s):
					for (ns, r) in self.states_and_actions.get_next_states_rewards(s, a):
						new_state_value += self.policy.get_probability(s,a) * self.mdp.get_probability(s,a,ns,r) * (r + self.discount*self.states_and_actions.get_state_value(ns))

				self.states_and_actions.set_state_value(s, new_state_value)	# Update to new value

				delta = max(delta, abs(new_state_value - old_state_value))
			
			print("Policy evaluated", iteration_count, "times with delta", delta)
			# self.__pretty_printing()

	def __pretty_printing(self):
		for i in range(25):
			state = (i%5, i//5)
			val = round(self.states_and_actions.get_state_value(state), 1)
			print(val, end = "\t")
			if(i % 5 == 4):
				print()



class policy_improver:
	states_and_actions = None
	policy = None
	discount = 1

	ARGMAX_BOUND = 1e-2	# The bound for maximum comparision

	def __init__(self, policy, states_and_actions, mdp, discount):
		self.policy = policy
		self.states_and_actions = states_and_actions
		self.mdp = mdp
		self.discount = discount

	"""
	Runs policy improvement step
	Returns true of policy was stable
	"""
	def improve(self):
		stable = True
		
		# Find argmax for all states
		for s in self.states_and_actions.get_states():
			action_returns = []	# [(action, expected return), ...] for all actions from state s
			for a in self.states_and_actions.get_actions(s):
				expected_return = 0
				for (ns, r) in self.states_and_actions.get_next_states_rewards(s, a):
					expected_return += self.mdp.get_probability(s,a,ns,r) * (r  + self.discount * self.states_and_actions.get_state_value(ns))
				action_returns.append((a, expected_return))

			greedy_actions = self.__argmax(action_returns)
			stable &= self.policy.set_greedy(s, greedy_actions)

		return stable


	"""
	Takes in a list of [(action, expected return)] and returns a list [action] of for actions with maximum return
	"""
	def __argmax(self, action_returns):
		max_actions = []
		max_return = -inf

		for (a, er) in action_returns:
			if(max_return + self.ARGMAX_BOUND < er):
				max_actions.clear()
				max_actions.append(a)
				max_return = er
			elif(max_return - self.ARGMAX_BOUND < er):
				max_actions.append(a)

		return max_actions



class policy_iterator:
	policy = None
	states_and_actions = None
	discount = None
	mdp = None
	policy_evaluator = None
	policy_improver = None

	theta = 1e-2	# Error bound for policy evaluation (-1 if doing value iterations)

	def __init__(self, policy, states_and_actions, mdp, discount):
		self.policy = policy
		self.states_and_actions = states_and_actions
		self.mdp = mdp
		self.discount = discount
		self.policy_evaluator = policy_evaluator(self.policy, self.states_and_actions, self.mdp, self.discount)
		self.policy_improver = policy_improver(self.policy, self.states_and_actions, self.mdp, self.discount)


	"""
	Runs policy iterations
	Policy evaluation is done by solving linear equations (ONLY FOR Q4)
	"""
	def iterate(self):
		policy_stable = False
		iteration_count = 0
		while(not policy_stable):
			iteration_count += 1
			# Policy evaluation step
			self.policy_evaluator.iterative(self.theta)

			# Policy improvement
			policy_stable = self.policy_improver.improve()

			print("Improved policy", iteration_count, "times")



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
				self.sa.add_action(s, a, [(ns, r)])
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

	def q4(self):
		p = policy(self.sa)
		pi = policy_iterator(p, self.sa, self.mdp, 0.9)
		pi.iterate()

		# pretty printing
		# v_*(s)
		for i in range(25):
			state = (i%5, i//5)
			val = round(self.sa.get_state_value(state), 1)
			print(val, end = "\t")
			if(i % 5 == 4):
				print()

		# pi_*(s)
		for i in range(25):
			state = (i%5, i//5)
			all_greedy_actions = p.get_all_greedy_actions(state)
			optimal_actions = ""
			if(self.NORTH in all_greedy_actions):
				optimal_actions += "N"
			else:
				optimal_actions += "-"
			if(self.SOUTH in all_greedy_actions):
				optimal_actions += "S"
			else:
				optimal_actions += "-"
			if(self.EAST in all_greedy_actions):
				optimal_actions += "E"
			else:
				optimal_actions += "-"
			if(self.WEST in all_greedy_actions):
				optimal_actions += "W"
			else:
				optimal_actions += "-"
			print(optimal_actions, end="\t")
			if(i % 5 == 4):
				print()

	def q6(self):
		p = policy(self.sa)
		pi = policy_iterator(p, self.sa, self.mdp, 0.9)
		pi.iterate()

		print("Completed policy iterations")
		# pretty printing
		# v_*(s)
		for i in range(25):
			state = (i%5, i//5)
			val = round(self.sa.get_state_value(state), 1)
			print(val, end = "\t")
			if(i % 5 == 4):
				print()

		# pi_*(s)
		for i in range(25):
			state = (i%5, i//5)
			all_greedy_actions = p.get_all_greedy_actions(state)
			optimal_actions = ""
			if(self.NORTH in all_greedy_actions):
				optimal_actions += "N"
			else:
				optimal_actions += "-"
			if(self.SOUTH in all_greedy_actions):
				optimal_actions += "S"
			else:
				optimal_actions += "-"
			if(self.EAST in all_greedy_actions):
				optimal_actions += "E"
			else:
				optimal_actions += "-"
			if(self.WEST in all_greedy_actions):
				optimal_actions += "W"
			else:
				optimal_actions += "-"
			print(optimal_actions, end="\t")
			if(i % 5 == 4):
				print()

class jacks_rental:
	mdp = None	# MDP
	sa = None	# States and actions

	
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

		self.__add_states()
		print("Added states")
		self.__add_actions()
		print("Added actions")


	def __add_states(self):
		# Add states
		for i in range(21):
			for j in range(21):
				state = (i,j)
				self.sa.add_state(state)

	def __add_actions(self):
		# Add actions, probabilities, next states and rewards
		done_count = 0	# Variabe to keep track of progress
		for s in self.sa.get_states():
			# Actions are integers in [-5, 5]. They represent cars moved form A to B.
			for a in range(-5,6):
				intermidiate_state = (s[0] - a, s[1] + a)	# This intermidiate state is affected solely by action. We will add returned and rented cars next.
				next_states_rewards = []
				for ns in self.sa.get_states():
					p, r = self.__get_probability_and_reward(intermidiate_state, ns)
					assert p <= 1
					r -= max(0, abs(a) - 1)	# Cost of moving. Subtract 1 for free moving by employee
					if(ns[0] > 10): # Cost for parking
						r -= 4
					if(ns[1] > 10): # Cost for parking
						r -= 4
					next_states_rewards.append((ns,r))
					self.mdp.set_probability(s,a,ns,r,p)	# Store probability
					done_count += 1
					print("Added for", done_count, "out of 2139291\t\t\t", end = "\r")
				self.sa.add_action(s, a, next_states_rewards)	# store next state reward pair
		print()


	"""
	Gives the probability and rewrds for going from state to next_state
	"""
	def __get_probability_and_reward(self, state, next_state):
		pa, ra = self.__get_probabilty_and_reward_for_car_delta(state[0], next_state[0], 0)
		pb, rb = self.__get_probabilty_and_reward_for_car_delta(state[1], next_state[1], 1)
		p = pa*pb
		r = ra+rb
		return p, r


	"""
	GIves the reward and probability for changing the car count from current to next
	Args:
		current_count	: Current number of cars at place
		next_count		: New number of cars at place
		place			: 0 for A and 1 for B
	"""
	def __get_probabilty_and_reward_for_car_delta(self, current_count, next_count, place):
		r = 0
		p = 0
		lambda_returned = 3 if place == 0 else 2
		lambda_rented = 3 if place == 0 else 4
		possible_rented_returned_pairs = []
		for rented in range(current_count+1):	 # I can rent from 0 to current_count
			returned = next_count - (current_count - rented) # These many vehicles must be returned
			
			# Probability and rewards for renting
			p_rented = self.__poisson(lambda_rented, rented)
			r_rented = 10 * rented

			# Probability and reward for returns
			p_returned = self.__poisson(lambda_returned, returned)
			r_returned = 0
			if(next_count == 20): # Handle the case where extra cars are sent to national facility
				p_returned = self.__poisson_greater_than_equal(lambda_returned, returned)

			# Rewards and probabilities for current rented returned pair
			r_cur = r_returned + r_rented
			p_cur = p_returned * p_rented

			r += r_cur * p_cur
			p += p_cur

		return p, r


	"""
	Returns probability for poisson distribution
	Args:
		n	: RV for which probability is needed
		lam	: Expected value of poisson distribution
	Rets:
		Probability of n
	"""
	def __poisson(self, lam, n):
		if(n < 0):
			return 0
		return ((lam**n)/factorial(n)) * (e**(-lam))
	
	"""
	Returns probability for poisson distribution of getting a number >= n
	Args:
		n	: RV for which probability is needed
		lam	: Expected value of poisson distribution
	Rets:
		Probability of >= n
	"""
	def __poisson_greater_than_equal(self, lam, n):
		less_than_prob = 0
		for i in range(n):
			less_than_prob += self.__poisson(lam, n)
		return 1 - less_than_prob

	def q7_modified(self):
		p = policy(self.sa)
		pi = policy_iterator(p, self.sa, self.mdp, 0.3)
		pi.iterate()

		print("Completed policy iterations")

		for s in self.sa.get_states():
			val = round(self.sa.get_state_value(s), 1)
			print(s, ":", val)


def main():
	# gw = gridworld()
	# gw.q6()
	jr = jacks_rental()
	jr.q7_modified()


if __name__ == '__main__':
	main()