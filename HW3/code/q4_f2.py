"""
Name	: Setu Gupta
R. No.	: 2018190
RL HW3 Q4
"""
from random import randint as randi

class Deck:
	def draw(self):
		return min(randi(1,13), 10)	#J,Q,K are treated as face cards with value 10

class Person:
	cards = []
	deck = None
	
	def __init__(self):
		self.deck = Deck()
		self.cards = []
		self.cards.append(self.deck.draw())	# Draw two cards
		self.cards.append(self.deck.draw())
	
	"""
	Returns value of current hand.
	Maintains two sums: One with usable ace (if possible), and one without
	Rets:	
		usedAce	: True if ace was considered usable while calculaing value
		value	: Value of the hand
	"""
	def getValue(self):
		val = 0		# tracks the value when ace was not usable
		valAce = 0	# tracks the value when ace was usable
		usedAce = False	# true if usable ace was used
		
		for c in self.cards:
			val += c
			valAce += c
			if(c == 1) and not usedAce:	# There can only be one usable ace
				usedAce = True
				valAce += 10
		
		if(valAce <= 21):	# If not bust
			return usedAce, valAce
		return False, val

	"""
	Draw a single card from deck
	"""
	def draw(self):
		self.cards.append(self.deck.draw())

class Dealer(Person):
	def getFaceUp(self):
		return self.cards[1]

	def act(self):
		while(self.getValue()[1] < 17):
			self.draw()

	"""
	Forces faceUp card
	"""
	def forceInitialState(self, faceUp):
		self.cards[1] = faceUp


class Player(Person):
	"""
	Append a special card which satisfies initial value. This value might be larger than 10
	"""
	def forceInitialState(self, value, usable):
		self.cards.clear()
		if(usable):
			self.cards.append(1)	# Insert ace
			value -= 11
			self.cards.append(value)
		else:
			self.cards.append(value)
			

"""
This class simulates a lackjack game
Sates are of the form (player's value, dealers's showing card, whether player has a usable ace)
"""
class Blackjack:
	player = None
	dealer = None

	def __init__(self):
		self.player = Player()
		self.dealer = Dealer()

	def getInitialState(self):
		playerUsedAce, playerValue = self.player.getValue()
		faceUp = self.dealer.getFaceUp()
		return (playerValue, faceUp, playerUsedAce)
	
	"""
	Takes in current action and returns the reward and next state. Terminal is true if state is terminal
	"""
	def step(self, action):
		
		terminal = False
		reward = 0
		nextState = None
		stick = False

		if(action == "hit"):
			self.player.draw()
		elif(action == "stick"):	# Game ends here
			self.dealer.act()
			stick = True
		else:
			print("Incorrect action")
			exit(0)

		playerUsedAce, playerValue = self.player.getValue()
		_, dealerValue = self.dealer.getValue()
		faceUp = self.dealer.getFaceUp()

		terminal = (playerValue > 21) or stick	# If player goes bust or it was dealers turn

		if(not stick): # If it was players turn
			reward = -1 if (playerValue > 21) else 0
		else:
			if(dealerValue > 21 or dealerValue < playerValue):
				reward =  1
			elif(dealerValue == playerValue):
				reward = 0
			else:
				reward = -1

		nextState = (playerValue, faceUp, playerUsedAce)
		return terminal, reward, nextState

	"""
	Forces initial state to a predefined value
	"""
	def forceInitialState(self, state):
		self.player.forceInitialState(state[0], state[2])	# Force player value and usable ace
		self.dealer.forceInitialState(state[1])			# Force dealer face up

class Policy:
	def getAction(self, state):
		pass

class PolicySimple(Policy):

	def getAction(self, state):
		if(state[0] == 20 or state[0] == 21):
			return "stick"
		return "hit"

class PolicyGreedy(Policy):
	stateActionPair = {}

	def __init__(self):
		self.stateActionPair = {}
		for playerSum in  range(1, 22):
			for dealerUp in range(1, 11):
				for usableAce in [True, False]:
					self.stateActionPair[(playerSum, dealerUp, usableAce)] = "hit" if playerSum < 20 else "stick"

	def setAction(self, state, action):
		self.stateActionPair[state] = action

	def getAction(self, state):
		return self.stateActionPair[state]


class Fig1:
	NUM_GAMES = 500000
	stateValue = {}	# A dictionary keeping account of state values
	stateCount = {} # A dictionary keeping account of how may times a state was seen

	"""
	Generates stateValue, a dictionary index by state (player's value, dealers's showing card, whether player has a usable ace) whoch stores the corresponding state value
	"""
	def generate(self):
		for gameNumber in range(self.NUM_GAMES):
			print("Running game number:", str(gameNumber + 1), "out of", self.NUM_GAMES, "\t", end="\r")
			blackjack = Blackjack()
			policy = PolicySimple()
			
			# Generate episode
			states = []
			rewards = [None]	# Rewards start form time step = 1. Hence insert dummy R_0
			currentState = blackjack.getInitialState()
			terminal = False
			while not terminal:
				action = policy.getAction(currentState)
				terminal, reward, nextState = blackjack.step(action)
				states.append(currentState)
				rewards.append(reward)
				currentState = nextState
			states.append(nextState)	# Append the terminal state

			assert(len(states) == len(rewards))

			G = 0
			T = len(rewards) - 2	# Index of last timestep
			for t in range(T, -1, -1):
				G += rewards[t+1]
				St = states[t]
				
				if(St not in self.stateValue):	# Encountered state first time
					self.stateValue[St] = 0
					self.stateCount[St] = 0

				self.stateValue[St] += G
				self.stateCount[St] += 1

		for s in self.stateValue:	# Take avg
			self.stateValue[s] /= self.stateCount[s]
		print()

	def prettyPrint(self):	# Pretty prints the value of states
		print("After 500,000 games")

		x = [' ', 'A'] + [str(i) for i in range(2, 11)]
		y = [str(i) for i in range(21, 11, -1)]
		# Initialize grids ---------------------------------------------
		outputGrid = []	# Output without usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("NA")
			outputGrid.append(row)

		outputGridUsable = []	# Output with usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("N/A")
			outputGridUsable.append(row)

		# Populate grids -----------------------------------------------
		for s in self.stateValue:
			# print("player:", s[0], ",", 21-s[0], "dealer:", s[1], ",", s[1]-1, "usable:", s[2], "Value:", self.stateValue[s])

			if(s[2]):
				outputGridUsable[21-s[0]][s[1]-1] = str(round(self.stateValue[s], 2))
			else:
				outputGrid[21-s[0]][s[1]-1] = str(round(self.stateValue[s], 2))

		# Print grids --------------------------------------------------
		print("With usable ace")
		for idx, row in enumerate(outputGridUsable[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))

		print()
		print("Without usable ace")
		for idx, row in enumerate(outputGrid[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))

class Fig2:
	NUM_GAMES = 500000
	totalStateActionValue = {}	# A dictionary keeping account of state action values
	totalStateActionCount = {}	# A dictionary keeping account of how may times a state action pair was seen
	stateValue = {}				# A dictionary keeping account of state values
	policy = None

	"""
	Returns a random initial state for exploring starts
	"""
	def getRandomState(self):
		playerSum = randi(12, 21)
		dealerUp = randi(1,10)
		usedAce = randi(1,2) == 1
		return (playerSum, dealerUp, usedAce)

	def getRandomAction(self):
		return "hit" if randi(1,2) == 1 else "stick"

	"""
	Generates stateValue, a dictionary index by state (player's value, dealers's showing card, whether player has a usable ace) whoch stores the corresponding state value
	"""
	def generate(self):
		self.policy = PolicyGreedy()
		for gameNumber in range(self.NUM_GAMES):
			print("Running game number:", str(gameNumber + 1), "out of", self.NUM_GAMES, "\t", end="\r")
			blackjack = Blackjack()
			
			# Generate episode
			states = []
			actions = []
			rewards = [None]	# Rewards start form time step = 1. Hence insert dummy R_0

			# Exploring starts
			blackjack.forceInitialState(self.getRandomState())
			currentState = blackjack.getInitialState()
			currentAction = self.getRandomAction()
			states.append(currentState)
			actions.append(currentAction)
		
			# Generate episode
			terminal = False
			while not terminal:
				terminal, reward, nextState = blackjack.step(currentAction)
				rewards.append(reward)
				states.append(nextState)
				if(not terminal):
					nextAction = self.policy.getAction(nextState)
				else:
					nextAction = None
				actions.append(nextAction)
				currentState = nextState
				currentAction = nextAction

			assert(len(states) == len(rewards) and len(states) == len(actions))

			# Store the index of first occurance of states for first visit MC
			firstVisitIdx = {}
			time = 0
			for s, a in zip(states, actions):
				if((s,a) not in firstVisitIdx):
					firstVisitIdx[(s, a)] = time
				time += 1
				if(a != None and (s,a) not in self.totalStateActionValue):
					self.totalStateActionValue[(s,a)] = 0
					self.totalStateActionCount[(s,a)] = 0
			
			# Estimate State Action Values
			G = 0
			T = len(rewards) - 2	# Index of last timestep
			for t in range(T, -1, -1):
				G += rewards[t+1]
				St = states[t]
				At = actions[t]

				if(firstVisitIdx[(St, At)] == t):	# First Visit MC
					self.totalStateActionValue[(St, At)] += G
					self.totalStateActionCount[(St, At)] += 1

			# Take average and get Q values
			Q = {}
			for (s,a) in self.totalStateActionValue:	# Take avg
				Q[(s,a)] = self.totalStateActionValue[(s,a)] / self.totalStateActionCount[(s,a)]

			for (s,a) in Q:
				if(a == "hit"):
					if((s,"stick") in Q):	# Both actions present
						# Take argmax and break ties randomly
						if(Q[(s, "hit")] > Q[(s, "stick")]):
							self.policy.setAction(s, "hit")
						elif(Q[(s, "hit")] < Q[(s, "stick")]):
							self.policy.setAction(s, "stick")
						else:
							self.policy.setAction(s, "hit" if randi(1,2) == 1 else "stick")
					else:	# Only hit present
						self.policy.setAction(s, "hit")
				else:
					if((s,"hit") not in Q):	# Only stick present
						self.policy.setAction(s, "stick")
		self.generateStateValue()
		print()

	def generateStateValue(self):
		for (s,a) in self.totalStateActionValue:
			self.totalStateActionValue[(s,a)] /= self.totalStateActionCount[(s,a)]
			self.stateValue[s] = self.totalStateActionValue[(s,self.policy.getAction(s))]

	def prettyPrint(self):	# Pretty prints the value of states
		print("After 500,000 games")

		x = [' ', 'A'] + [str(i) for i in range(2, 11)]
		y = [str(i) for i in range(21, 11, -1)]
		# Initialize grids ---------------------------------------------
		outputGrid = []	# Output without usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("N/A")
			outputGrid.append(row)

		outputGridUsable = []	# Output with usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("N/A")
			outputGridUsable.append(row)

		policyGrid = []	# Policy without usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("N/A")
			policyGrid.append(row)

		policyGridUsable = []	# Policy with usable ace
		for playerSum in range(1, 22):
			row = []
			for dealerShowing in range(1, 11):
				row.append("N/A")
			policyGridUsable.append(row)

		# Populate grids -----------------------------------------------
		for s in self.stateValue:
			if(s[2]):
				outputGridUsable[21-s[0]][s[1]-1] = str(round(self.stateValue[s], 2))
				policyGridUsable[21-s[0]][s[1]-1] = self.policy.getAction(s)
			else:
				outputGrid[21-s[0]][s[1]-1] = str(round(self.stateValue[s], 2))
				policyGrid[21-s[0]][s[1]-1] = self.policy.getAction(s)

		# Print grids --------------------------------------------------
		print("With usable ace")
		for idx, row in enumerate(outputGridUsable[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))

		print()
		print("Without usable ace")
		for idx, row in enumerate(outputGrid[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))

		# Print grids --------------------------------------------------
		print("Policy with usable ace")
		for idx, row in enumerate(policyGridUsable[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))

		print()
		print("Policy without usable ace")
		for idx, row in enumerate(policyGrid[:10]):
			print(y[idx] + "\t" + "\t".join(row))
		print("\t".join(x))



def main():
	fig2 = Fig2()
	fig2.generate()
	fig2.prettyPrint()

if __name__ == '__main__':
	main()

