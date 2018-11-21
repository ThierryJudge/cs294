import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import argparse
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
import gym

from BC import Agent


parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

# Load exprt policy
print('loading and building expert policy')
policy_fn = load_policy.load_policy(args.expert_policy_file)
print('loaded and built')

# Get inital data
with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
	expert_data = pickle.load(f)

obs = expert_data['observations']
actions = expert_data['actions']
actions = actions.squeeze()
expert_mean_return =expert_data['mean_return']
print("obs shape {}".format(obs.shape))
print("action shape {}".format(actions.shape))
print("mean_return {}".format(expert_mean_return))

# Create agent
agent = Agent(args.envname, obs.shape[1], actions.shape[1])
history = agent.train(obs, actions)


all_losses = history.history['loss']
# Setup environment
env = gym.make(args.envname)
max_steps = env.spec.timestep_limit
dagger_iterations = 10
num_rollouts = 50
with tf.Session():
	tf_util.initialize()

	# Run dagger algorithm
	all_observations = obs.tolist()
	all_actions = actions.tolist()
	all_returns = []
	all_stds = []
	try:	
		for j in range(dagger_iterations):
			print('Dagger interation {}/{}'.format(j, dagger_iterations))
			print("Running policy ({} rollouts)...".format(num_rollouts))
			returns = []
			observations = []
			actions = []
			for i in range(num_rollouts):
				#print('iter', i)
				obs = env.reset()
				done = False
				totalr = 0.
				steps = 0
				while not done:
					action = agent.act(obs[None,:])
					observations.append(obs)
					#actions.append(action)
					obs, r, done, _ = env.step(action)
					totalr += r
					steps += 1
					if args.render:
						env.render()
					#if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
					if steps >= max_steps:
						break
				returns.append(totalr)

			#print('returns', returns)
			print('mean return', np.mean(returns))
			print('std of return', np.std(returns))

			all_returns.append(np.mean(returns))
			all_stds.append(np.std(returns))

			#print("all_obs: {}".format(np.array(all_observations).shape))
			#print("all_actions: {}".format(np.array(all_actions).shape))
			
			print("Expert annotating data...")
			for obs in observations:
				actions.append(policy_fn(obs[None,:]).squeeze())

			#print("obs: {}".format(np.array(observations).shape))
			#print("actions: {}".format(np.array(actions).shape))

			#print(actions)
			all_observations.extend(observations)
			all_actions.extend(actions)

			#print("all_obs: {}".format(np.array(all_observations).shape))
			#print("all_actions: {}".format(np.array(all_actions).shape))
			print("Training policy...")
			history = agent.train(np.array(all_observations), np.array(all_actions))
			all_losses.extend(history.history['loss'])
	except KeyboardInterrupt as ki:
		print("\nTraining cancelled!\n")


print('All returns', all_returns)
print('mean All returns', np.mean(all_returns))
print('std of all returns', np.std(all_returns))

plt.figure(1)
plt.plot(all_losses)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show(block=False)

plt.figure(2)
x = np.arange(1, len(all_returns)+1)
plt.errorbar(x, all_returns, all_stds, linestyle='None', marker='^')
plt.axhline(y=expert_mean_return, c='r')
plt.ylabel('Episode returns')
plt.xlabel('dagger iterations')
plt.show()
