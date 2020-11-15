"""
Name	: Setu Gupta
R. No.	: 2018190
RL HW3 Q6
"""
from random import randint as randi
from matplotlib import pyplot as plt

"""
Environment to simulate MPR
"""
class Environment:
	"""
	Returns the state and reward sequence for a walk
	"""
	def getSequence(self):
		states = []
		rewards = []

		curState = 2

		while (curState != 5) and (curState != -1):
			nextState = curState - 1 if randi(1,2) == 1 else curState + 1
			curReward = 1 if (nextState == 5) else 0
			states.append(curState)
			rewards.append(curReward)
			curState = nextState

		assert(len(rewards) == len(states))
		return states, rewards

class Fig1:

	ALPHA = 0.1
	stateValue = []
	def __init__(self):
		self.env = Environment()
		self.plotInit()
		self.plotTrue()
		self.stateValue = [0.5] * 5
		self.plot("0")
		self.run(1)
		self.plot("1")
		self.run(9)
		self.plot("10")
		self.run(90)
		self.plot("100")
		self.plotShow()


	"""
	Sets up plot
	"""
	def plotInit(self):
		plt.xticks([0 ,1, 2, 3, 4], ['A', 'B', 'C', 'D', 'E'])
		plt.xlabel("State")
		plt.ylabel("Estimates")
		plt.grid()

	"""
	Adds a plot
	"""
	def plot(self, label):
		plt.plot(self.stateValue, label=label)

	"""
	Adds a plot for true values
	"""
	def plotTrue(self):
		self.stateValue = [1/6, 2/6, 3/6, 4/6, 5/6]
		self.plot("Truth")

	"""
	Displays the plot
	"""
	def plotShow(self):
		plt.title("Estimated values of states")
		plt.legend()
		plt.show()

	"""
	Generated TD estimates with ep number of episodes
	"""
	def run(self, ep):
		for e in range(ep):
			states, rewards = self.env.getSequence()	# Generates state reward sequence
			for idx in range(len(states)):
				s = states[idx]
				r = rewards[idx]
				nextStateValue = 0
				if(idx < len(states) - 1):	# Bootstrap for next state value. If at terminal state, next state value is 0
					ns = states[idx+1]
					nextStateValue = self.stateValue[ns]
				self.stateValue[s] = self.stateValue[s] + self.ALPHA*(r + nextStateValue - self.stateValue[s])	# Update at each step in episode

def main():
	Fig1()

if __name__ == '__main__':
	main()
