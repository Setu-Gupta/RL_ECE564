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

class Fig2:

	errors = []
	def __init__(self):
		self.env = Environment()
		self.errors = [0] * 100
		self.plotInit()
		self.runTD(0.05)
		self.plot("TD with alpha = 0.05")
		self.runTD(0.10)
		self.plot("TD with alpha = 0.10")
		self.runTD(0.15)
		self.plot("TD with alpha = 0.15")
		self.runMC(0.01)
		self.plot("MC with alpha = 0.01")
		self.runMC(0.02)
		self.plot("MC with alpha = 0.02")
		self.runMC(0.03)
		self.plot("MC with alpha = 0.03")
		self.runMC(0.04)
		self.plot("MC with alpha = 0.04")
		self.plotShow()


	"""
	Sets up plot
	"""
	def plotInit(self):
		plt.xlabel("Walks")
		plt.ylabel("Error")
		plt.grid()
		plt.autoscale(enable=True, axis='x', tight=True)

	"""
	Adds a plot
	"""
	def plot(self, label):
		plt.plot(self.errors, label=label)

	"""
	Displays the plot
	"""
	def plotShow(self):
		plt.title("Emperical RMS error, averaged over states")
		plt.legend()
		plt.show()


	"""
	Gives RMS error averaged over states goven a stateValue fucntion
	"""
	def getRMSerror(self, stateValue):
		truth = [1/6, 2/6, 3/6, 4/6, 5/6]
		err = 0
		for idx in range(len(stateValue)):
			err += (truth[idx] - stateValue[idx])**2
		return (err/5)**(1/2)

	"""
	Generates 100 runs of 100 episodes of TD and updates self.errors with step size = alpha
	"""
	def runTD(self, alpha):
		self.errors = [0] * 100
		for run in range(100):
			stateValue = [0.5] * 5
			for episode in range(100):
				states, rewards = self.env.getSequence()	# Generates state reward sequence
				for idx in range(len(states)):
					s = states[idx]
					r = rewards[idx]
					nextStateValue = 0
					if(idx < len(states) - 1):	# Bootstrap for next state value. If at terminal state, next state value is 0
						ns = states[idx+1]
						nextStateValue = stateValue[ns]
					stateValue[s] += alpha*(r + nextStateValue - stateValue[s])	# Update at each step in episode
				self.errors[episode] += self.getRMSerror(stateValue)	# Store the error for current episode
		
		for i in range(100):
			self.errors[i] /= 100	# Average over 100 runs

	"""
	Generates 100 runs of 100 episodes of MC and updates self.errors with step size = alpha
	"""
	def runMC(self, alpha):
		self.errors = [0] * 100
		for run in range(100):
			stateValue = [0.5] * 5
			for episode in range(100):
				states, rewards = self.env.getSequence()	# Generates state reward sequence

				G = 0
				for idx in range(len(states)-1, -1, -1):	# Iterate backwards
					s = states[idx]
					r = rewards[idx]
					G += r

					stateValue[s] += alpha*(G - stateValue[s])	# Simple MC update. No bootstrapping with next state
					
				self.errors[episode] += self.getRMSerror(stateValue)	# Store the error for current episode
		
		for i in range(100):
			self.errors[i] /= 100	# Average over 100 runs


def main():
	Fig2()

if __name__ == '__main__':
	main()
