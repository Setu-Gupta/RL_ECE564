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


class Player(Person):
	pass

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

class Policy:
	def getAction(self, state):
		pass

class PolicySimple(Policy):

	def getAction(self, state):
		if(state[0] == 20 or state[0] == 21):
			return "stick"
		return "hit"

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
				row.append("N/A")
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

def main():
	fig1 = Fig1()
	fig1.generate()
	fig1.prettyPrint()

if __name__ == '__main__':
	main()

