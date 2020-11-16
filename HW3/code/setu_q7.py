"""
Name	: Setu Gupta
R. No.	: 2018190
RL HW3 Q7
"""
from random import randint as randi
from matplotlib import pyplot as plt
from math import inf
from numpy.random import choice

"""
Environment to simulate MPR
"""
class CliffWalk:
	"""
	x coordinate increases from left to right
	y coordinate increases from bottom to up
	"""
	curState = (-1,-1)
	START = (0,0)
	GOAL = (11,0)

	def __init__(self):		
		self.START = (0,0)
		self.GOAL = (11,0)
		self.curState = self.START

	"""
	Takes action in current state and returns reward and next state. Also returns true if terminal state
	"""
	def step(self, action):
		nextState = ()
		if(action == 'up'):
			nextState = (self.curState[0], self.curState[1]+1)
		elif(action == 'down'):
			nextState = (self.curState[0], self.curState[1]-1)
		elif(action == 'left'):
			nextState = (self.curState[0]-1, self.curState[1])
		elif(action == 'right'):
			nextState = (self.curState[0]+1, self.curState[1])

		assert(nextState[0] >= 0 and nextState[1] <= 11)
		assert(nextState[1] >= 0 and nextState[1] <= 3)

		terminal = False
		reward = -1
		if(nextState[0] >= 1 and nextState[0] <= 10 and nextState[1] == 0):	# Cliff
			# terminal = True
			reward = -100
			nextState = self.START

		if(nextState == self.GOAL):
			terminal = True

		self.curState = nextState

		return terminal, reward, nextState

	"""
	Returns possible actions from a given state
	"""
	def getPossibleActions(self, state):
		possibleActions = []
		if(state[0]+1 <= 11):
			possibleActions.append('right')
		if(state[0]-1 >= 0):
			possibleActions.append('left')
		if(state[1]+1 <= 3):
			possibleActions.append('up')
		if(state[1]-1 >= 0):
			possibleActions.append('down')
		return possibleActions

	"""
	Returns a list of all possible states
	"""
	def getAllStates(self):
		states = []
		for x in range(12):
			for y in range(4):
				states.append((x,y))
		return states

	"""
	Resets the environment and returns current state
	"""
	def reset(self):
		self.curState = self.START
		return self.curState

class Q:
	stateActionPairs = {}
	environment = None
	"""
	Uses environment to make all possible state action pairs and initialize them to 0
	"""
	def __init__(self, env):
		self.env = env
		self.stateActionPairs = {}
		for s in self.env.getAllStates():
			for a in self.env.getPossibleActions(s):
				self.stateActionPairs[(s,a)] = 0	# Initialize to 0

	"""
	Increments a given state action pair
	"""
	def update(self, state, action, increment):
		self.stateActionPairs[(state, action)] += increment

	"""
	Returns the value for a given state action pair
	"""
	def get(self, state, action):
		return self.stateActionPairs[(state, action)]

	"""
	Returns the max value over actions for a given state
	"""
	def getMax(self, state):
		maxVal = -inf
		for a in self.env.getPossibleActions(state):
			maxVal = max(self.stateActionPairs[(state, a)], maxVal)
		return maxVal
	
	"""
	Returns the action corresponding to max value over actions for a given state
	"""
	def getArgMax(self, state):
		maxVal = -inf
		argMaxAction = 'NULL'
		for a in self.env.getPossibleActions(state):
			if maxVal < self.stateActionPairs[(state, a)]:
				argMaxAction = a
				maxVal = self.stateActionPairs[(state, a)]
		return argMaxAction

	"""
	Returns the list of all possible actions from a given state
	"""
	def getPossibleActions(self, state):
		return self.env.getPossibleActions(state)

class EpsilonGreedyPolicy:
	Q = None
	EPSILON = 0.1

	def __init__(self, Q):
		self.EPSILON = 0.1
		self.Q = Q

	"""
	Returns an action from state depending on a epsilon greedy distribution depending on Q
	"""
	def getAction(self, state):
		greedy = self.Q.getArgMax(state)	# Find greedy choice
		exploration = self.Q.getPossibleActions(state)	# Find non greedy choices
		exploration.remove(greedy)
		
		actions = exploration + [greedy]	# Complete list of arms in the following order: [non greedy, greedy]
		
		prob = [self.EPSILON/len(actions)] * (len(actions) - 1)	# Generate probability distribution
		prob += [1 - self.EPSILON + (self.EPSILON/len(actions))]
		
		return choice(actions, p = prob)	# Pick an arm according to probability

class Q7:
	ALPHA = 0.2
	returns = []	# Sum of rewards indexed by episode number
	EPISODES = 500
	RUNS = 100

	def __init__(self):
		self.returns = [0]* self.EPISODES
		self.plotInit()
		self.runSARSA()
		self.plot("SARSA")
		self.runQLearning()
		self.plot("Q-Learning")
		self.plotShow()

	"""
	Sets up plot
	"""
	def plotInit(self):
		plt.xlabel("Episodes")
		plt.ylabel("Sum of rewards during episode")
		plt.grid()
		plt.ylim(-150, 0)

	"""
	Adds a plot
	"""
	def plot(self, label):
		plt.plot(self.returns, label=label)

	"""
	Displays the plot
	"""
	def plotShow(self):
		plt.legend()
		plt.show()

	def runSARSA(self):
		print("Running SARSA")
		self.returns = [0]* self.EPISODES

		for r in range(self.RUNS):
			env = CliffWalk()
			q = Q(env)
			policy = EpsilonGreedyPolicy(q)

			for ep in range(self.EPISODES):
				print("Running episode number", (ep+1), "\tout of", self.EPISODES, "for run", r+1, "\tout of", self.RUNS, "\t\t", end='\r')

				S = env.reset()
				A = policy.getAction(S)

				G = 0 # Total return for the episode
				terminal = False
				while not terminal:
					terminal, R, nextS = env.step(A)
					G += R
					nextA = policy.getAction(nextS)
					inc = self.ALPHA*(R + q.get(nextS, nextA) - q.get(S,A))	# Calculate delta
					q.update(S,A,inc)  # Update Q
					S = nextS
					A = nextA
				self.returns[ep] += G
		
		for ep in range(self.EPISODES):	# Take average over runs
			self.returns[ep] /= self.RUNS
		print()


	def runQLearning(self):
		print("Running Q-Learning")
		self.returns = [0]* self.EPISODES

		for r in range(self.RUNS):
			env = CliffWalk()
			q = Q(env)
			policy = EpsilonGreedyPolicy(q)

			for ep in range(self.EPISODES):
				print("Running episode number", (ep+1), "\tout of", self.EPISODES, "for run", r+1, "\tout of", self.RUNS, "\t\t", end='\r')

				S = env.reset()

				G = 0 # Total return for the episode
				terminal = False
				while not terminal:
					A = policy.getAction(S)
					terminal, R, nextS = env.step(A)
					G += R
					inc = self.ALPHA*(R + q.getMax(nextS) - q.get(S,A))	# Calculate delta
					q.update(S,A,inc)  # Update Q
					S = nextS
				self.returns[ep] += G

		for ep in range(self.EPISODES):	# Take average over runs
			self.returns[ep] /= self.RUNS
		print()

def main():
	Q7()

if __name__ == '__main__':
	main()
