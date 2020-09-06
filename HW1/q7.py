from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
import itertools
from math import exp

"""
Gives action and probability distribution according to soft-max distribution.
Args:
	h		: Current preference
Rets:
	action	: Chosen action
	pi	: probability distribution
"""
def get_action_and_prob(h):
	pi = [exp(x) for x in h]
	total = sum(pi)
	pi = [x/total for x in pi]

	action = np.random.choice(range(len(pi)), p = pi)
	return action, pi



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
Upadtes preferences for next time step
Args:
	h			: Current preferences (Modified in place)
	pi			: Probability distribution
	alpha		: Step size
	reward 		: Obtained reward
	avg_rwd		: Average reward uptill now
	chosen_arm	: Index of chosen arm
"""
def update_preference(h, pi, alpha, reward, avg_rwd, chosen_arm):
	for i in range(len(h)):
		if(i == chosen_arm):
			h[i] += alpha * (reward - avg_rwd) * (1 - pi[i])
		else:
			h[i] -= alpha * (reward - avg_rwd) * pi[i]


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
Runs one instance of k armed bandit problem. It uses soft max to learn.
Conventions:
	At each step i, I choose action[i] and get reward[i] based on preferences q[i]
Args:	
	arms		: [[expectation, variance], ...] A list with k elements, each describing arm_i [1 <= i <= k] by expectation and variance
	alpha		: step size
	steps		: number of steps to run
	initial		: A list of length k denoting inital preferences
	baseline	: True of False. Indicate whether to use baseline
Rets:
	optimal	: A list of size steps where each element is 1 if optimal choice was taken, else 0
"""
def run(arms, alpha, steps, initial, baseline):
	h = copy(initial)	# Running estimates for all actions
	pi = [] # Probability distribution
	
	k = len(arms)
	optimal = [0] * steps

	total_rwd = 0
	for t in range(1, steps + 1):	# Run for "steps" time steps

		# Choose an action based on estimate
		action, pi = get_action_and_prob(h)

		optimal_arm = get_optimal(arms)
		optimal[t-1] = 1 if (action == optimal_arm) else 0

		# Get a reward
		reward = get_reward(action, arms)
		total_rwd += reward

		# Update estimates
		avg_rwd = 0 if baseline else total_rwd/t
		update_preference(h, pi, alpha, reward, avg_rwd, action)

	return optimal



VARIANCE_OF_EACH_ARM = 1
VARIANCE = 1
MEAN = 4
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



NUMBER_OF_EXPERIMENTS	= 2000
NUMBER_OF_STEPS			= 1000
NUMBER_OF_ARMS			= 10
"""
Runs experiment spits out data for plotting
Args:
	alpha		: Step size to use for updates
	baseline	: True of False. Whether to use basline for updates or not.
Rets:
	rewards	: % of times optimal action was taken indexed by timestep across runs
"""
def run_experiment(alpha, baseline):
	print("Running for alpha =", alpha, "baseline =", baseline)
	
	optimal = [0] * NUMBER_OF_STEPS
	initial = [0] * NUMBER_OF_ARMS
	for exp in range(NUMBER_OF_EXPERIMENTS):
		
		print("Running experiment number", exp+1, "out of", NUMBER_OF_EXPERIMENTS, "\t\t\t", end = "\r")
		arms = get_arms(NUMBER_OF_ARMS)
		o = run(arms, alpha, NUMBER_OF_STEPS, initial, baseline)	# Run experiment

		for t in range(NUMBER_OF_STEPS):
			optimal[t] += o[t]


	print("Done!" + " " * 40)
	optimal = [100*x/NUMBER_OF_EXPERIMENTS for x in optimal]

	return optimal


def main():
	alpha_baseline_choices = [[0.1, True], [0.1, False], [0.4, True], [0.4, False]]
	optimal = []	# Indexed as rewards[epsilon][timestep]

	# Collect data
	for ab in alpha_baseline_choices:
		o = run_experiment(ab[0], ab[1])
		optimal.append(o)

	# Start plotting
	plt.figure()
	plt.title("% Optimal action v/s steps")
	plt.xlabel("Steps")
	plt.ylabel("% Optimal action")
	for (o, ab) in zip(optimal, alpha_baseline_choices):
		label = "alpha = " + str(ab[0])
		label += " with baseline" if ab[1] else ""
		plt.plot(o, label = label)
		plt.gca().autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	plt.tight_layout()

	plt.show()

if __name__ == '__main__':
	main()