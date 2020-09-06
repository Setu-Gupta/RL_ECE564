from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
import itertools

"""
Gives action depending on epsilon and current estimates.
Args:
	epsilon	: epsilon for epsilon greedy (if -1, then epsilon changes with time t as epsilon = 1/t)
	q		: Current estimates. A list with k elements each describing estimate of expectaion for that arm
	t		: Time step
Rets:
	action	: Chosen action
"""
def get_action(epsilon, q, t):
	if(epsilon == -1):	# Choose epsilon
		epsilon = 1/t

	greedy = np.argmax(q)	# Find greedy choice
	exploration = list(range(len(q)))	# Find non greedy choices
	exploration.remove(greedy)
	
	arms = exploration + [greedy]	# Complete list of arms in the following order: [non greedy, greedy]
	
	prob = [epsilon/len(q)] * (len(q) - 1)	# Generate probability distribution
	prob += [1 - epsilon + (epsilon/len(q))]
	
	return np.random.choice(arms, p = prob)	# Pick an arm according to probability


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



UPDATE_MEAN = 0
UPDATE_STD_DEV = 0.01
"""
Updates arm characteristics.
Args:
	arms	: Arms to be upadted (in place). [[expectation, variance], ...k]
"""
def update_arms(arms):
	for a in arms:
		a[0] += np.random.normal(UPDATE_MEAN, UPDATE_STD_DEV)



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



"""
Runs one instance of k armed bandit problem. It uses epsilon greedy to learn.
Armstake random walks.
Conventions:
	At each step i, I choose action[i] and get reward[i] based on estimates q[i]
Args:
	k			: number of arms
	epsilon		: epsilon in epsilon greedy (if -1 => variable epsilon with epsilon = 1/n)
	steps		: number of steps to run
	initial		: A list of length k denoting inital estimates for expectation
	step_size	: If step size is mentioned, then use this instead of sample mean
Rets:
	rewards	: A list of size steps where each element is award obtained at that step
	optimal	: A list of size steps where each element is 1 if optimal arm was chosen, else 0
"""
def run(k, epsilon, steps, initial, step_size = None):
	q = copy(initial)	# Running estimates for all actions
	
	arms = get_arms(k)	# Get initial set of arms

	rewards = []
	optimal = [0] * steps
	choice_count = [0] * k	# Keeps track of number of times a particular arm is chosen

	for t in range(1, steps + 1):	# Run for "steps" time steps

		# Choose an action based on estimate
		action = get_action(epsilon, q, t)
		choice_count[action] += 1

		# Get a reward
		reward = get_reward(action, arms)
		rewards.append(reward)

		# Update optimal
		optimal_action = get_optimal(arms)
		optimal[t-1] = 1 if action == optimal_action else 0

		# Update estimates
		update_estimates(q, choice_count, step_size, action, reward)

		# Update arms
		update_arms(arms)

	return rewards, optimal




NUMBER_OF_EXPERIMENTS	= 2000
NUMBER_OF_STEPS			= 10000
NUMBER_OF_ARMS			= 10
"""
Runs experiment spits out data for plotting
Args:
	epsilon		: Epsilon to choose for experiment 
	step_size	: Step size for estimate updation. None if using sample mean
Rets:
	rewards	: Average of rewards indexed by timestep across runs
	optimal	: % of times that optimal action was chosen indexed by timestep
"""
def run_experiment(epsilon, step_size = None):
	print("Running for epsilon =", epsilon)
	
	rewards = [0] * NUMBER_OF_STEPS
	optimal = [0] * NUMBER_OF_STEPS

	initial = [0] * NUMBER_OF_ARMS
	k = NUMBER_OF_ARMS

	for exp in range(NUMBER_OF_EXPERIMENTS):
		
		print("Running experiment number", exp+1, "out of", NUMBER_OF_EXPERIMENTS, "\t\t\t", end = "\r")
		r, o = run(k, epsilon, NUMBER_OF_STEPS, initial, step_size)	# Run experiment

		for t in range(NUMBER_OF_STEPS):
			rewards[t] += r[t]
			optimal[t] += o[t]


	print("Done!" + " " * 40)
	rewards = [x/NUMBER_OF_EXPERIMENTS for x in rewards]
	optimal = [100*x/NUMBER_OF_EXPERIMENTS for x in optimal]

	return rewards, optimal


def main():
	# epsilon_choices = [0.1, 0.01, 0, -1]
	epsilon = 0.1
	rewards = []	# Indexed as rewards[epsilon][timestep]
	optimal = []	# Indexed as optimal[epsilon][timestep]
	error = []		# Indexed as error[epsilon][timestep][arm]

	reward_sample_mean, optimal_sample_mean = run_experiment(epsilon)
	reward_alpha, optimal_alpha = run_experiment(epsilon, 0.1)

	# Start plotting
	plt.figure()
	plt.title("Average reward v/s steps")
	plt.xlabel("Steps")
	plt.ylabel("Average reward")
	plt.plot(reward_sample_mean, label = "Sample mean")
	plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.plot(reward_alpha, label = "Alpha = 0.1")
	plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	plt.tight_layout()

	plt.figure()
	plt.title("% Optimal action v/s steps")
	plt.xlabel("Steps")
	plt.ylabel("% Optimal action")
	plt.plot(optimal_sample_mean, label = "Sample mean")
	plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.plot(optimal_alpha, label = "Alpha = 0.1")
	plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	plt.tight_layout()

	plt.show()



if __name__ == '__main__':
	main()