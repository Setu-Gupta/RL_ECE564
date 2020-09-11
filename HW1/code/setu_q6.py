from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
import itertools
from math import log as ln


EPSILON = 0.1
"""
Gives action depending on epsilon and current estimates.
Args:
	confidence		: c for UCB (if -1, use epsilon greedy with epsilon = EPSILON)
	q				: Current estimates. A list with k elements each describing estimate of expectaion for that arm
	t				: Time step
	choice_count	: COunt of how many times each action has been chosen
Rets:
	action	: Chosen action
"""
def get_action(confidence, q, t, choice_count):
	if(confidence == -1):	# Choose epsilon greedy
		epsilon = EPSILON
		greedy = np.argmax(q)	# Find greedy choice
		exploration = list(range(len(q)))	# Find non greedy choices
		exploration.remove(greedy)
	
		arms = exploration + [greedy]	# Complete list of arms in the following order: [non greedy, greedy]
	
		prob = [epsilon/len(q)] * (len(q) - 1)	# Generate probability distribution
		prob += [1 - epsilon + (epsilon/len(q))]	

		return np.random.choice(arms, p = prob)	# Pick an arm according to probability

	ucb = copy(q)
	for i in range(len(ucb)):
		if(choice_count[i] == 0):	# Avoid div by 0
			return i
		ucb[i] += confidence*((ln(t)/choice_count[i]) ** (1/2))
	return np.argmax(ucb)	# Find greedy choice

"""
Returns reward based in chosen action.
Args:
	action	: Chosen action
	arms	: [[expectaion, variance], ...] a list decribing each arm.
Rets:
	reward	: reward according to action
"""
def get_reward(action, arms):
	mean = arms[action][0]
	std_dev = arms[action][1] ** (1/2)
	return np.random.normal(mean, std_dev)


"""
Upadtes estimates for next time step
Args:
	q				: Current estimates (Modified in place)
	choice_count	: A list describing number of times a partical action was chosen
	step_size		: Step size to use while updating. None if sample mean
	action			: Chosen action
	reward			: Obtained reward
"""
def update_estimates(q, choice_count, step_size, action, reward):
	alpha = step_size;	# Select alpha
	if(step_size == None):	# Sample mean
		alpha = 1/choice_count[action]

	q[action] += alpha * (reward - q[action])	# Update estimate




"""
Gives the error between current estimate and truth.
Args:
	q		: Current estimates. A list with k values where each value is the estimate of expecation for that arm.
	arms	: [[expectaion, variance], ...] a list decribing each arm.
Rets:	
	error	: A list with k elements denoting errors.
"""
def get_error(q, arms):
	errors = []
	for (estimate, truth) in zip(q, arms):
		errors.append(abs(truth[0] - estimate))
	return errors



"""
Runs one instance of k armed bandit problem. It uses UCB or epsilon greedy to learn.
Conventions:
	At each step i, I choose action[i] and get reward[i] based on estimates q[i]
Args:	
	arms		: [[expectation, variance], ...] A list with k elements, each describing arm_i [1 <= i <= k] by expectation and variance
	confidence	: c in UCB (if -1 => epsilon greedy with epsilon = 0.1)
	steps		: number of steps to run
	initial		: A list of length k denoting inital estimates for expectation
	step_size	: If step size is mentioned, then use this instead of sample mean
Rets:
	rewards	: A list of size steps where each element is award obtained at that step
"""
def run(arms, confidence, steps, initial, step_size = None):
	q = copy(initial)	# Running estimates for all actions
	
	k = len(arms)

	rewards = []
	choice_count = [0] * k	# Keeps track of number of times a particular arm is chosen

	for t in range(1, steps + 1):	# Run for "steps" time steps
		
		# Choose an action based on estimate
		action = get_action(confidence, q, t, choice_count)
		choice_count[action] += 1

		# Get a reward
		reward = get_reward(action, arms)
		rewards.append(reward)

		# Update estimates
		update_estimates(q, choice_count, step_size, action, reward)

	return rewards



VARIANCE_OF_EACH_ARM = 1
VARIANCE = 1
MEAN = 0
"""
Returns a set of arms which have their mean picked from a normal distribution(MEAN, VARIANCE) and variance = VARIANCE_OF_EACH_ARM
Args:
	k	: Number of arms
Rets:
	arms	: Generated arms
"""
def get_arms(k):
	std_dev = VARIANCE ** (1/2)
	arms = [[np.random.normal(MEAN, std_dev), VARIANCE_OF_EACH_ARM] for a in range(k)]
	return arms


"""
Gives the optimal arm based on arms
Args:	
	arms	: Description of arms
Rets:	
	optimal arm
"""
def get_optimal(arms):
	optimal_arm = 0;
	optimal_exp = arms[optimal_arm][0]

	for i in range(len(arms)):
		if(arms[i][0] > optimal_exp):
			optimal_arm = i
			optimal_exp = arms[optimal_arm][0]

	return optimal_arm



NUMBER_OF_EXPERIMENTS	= 2000
NUMBER_OF_STEPS			= 1000
NUMBER_OF_ARMS			= 10
"""
Runs experiment spits out data for plotting
Args:
	confidence	: c to choose for experiment (-1 if I want to use epsilon greedy)
Rets:
	rewards	: Average of rewards indexed by timestep across runs
"""
def run_experiment(confidence):
	print("Running for confidence =", confidence)
	
	rewards = [0] * NUMBER_OF_STEPS

	initial = [0] * NUMBER_OF_ARMS
	for exp in range(NUMBER_OF_EXPERIMENTS):
		
		print("Running experiment number", exp+1, "out of", NUMBER_OF_EXPERIMENTS, "\t\t\t", end = "\r")
		arms = get_arms(NUMBER_OF_ARMS)
		r = run(arms, confidence, NUMBER_OF_STEPS, initial)	# Run experiment

		for t in range(NUMBER_OF_STEPS):
			rewards[t] += r[t]


	print("Done!" + " " * 40)
	rewards = [x/NUMBER_OF_EXPERIMENTS for x in rewards]

	return rewards


def main():
	confidence_choices = [-1, 1, 2, 4]
	rewards = []	# Indexed as rewards[epsilon][timestep]

	# Collect data
	for c in confidence_choices:
		r = run_experiment(c)
		rewards.append(r)

	# Start plotting
	plt.figure()
	plt.title("Average reward v/s steps")
	plt.xlabel("Steps")
	plt.ylabel("Average reward")
	for (r, c) in zip(rewards, confidence_choices):
		label = "c = " + str(c)
		if(c == -1):
			label = "epsilon = 0.1"
		plt.plot(r, label = label)
		plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	plt.tight_layout()

	plt.show()



if __name__ == '__main__':
	main()