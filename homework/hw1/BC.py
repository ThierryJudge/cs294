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


class Agent:
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.model = self.build_model(state_size, action_size)

	def build_model(self, state_size, action_size):
		model = Sequential()
		model.add(Dense(32, activation='relu', input_dim=state_size))
		model.add(Dense(action_size, activation='linear'))
		model.compile(optimizer='sgd',
			      loss='mean_squared_error',
			      metrics=['accuracy'])
		return model
	def train(self, obs, expert_actions, epochs = 10, plot=False):
		history = self.model.fit(obs, expert_actions, epochs=epochs, batch_size=128)
		if plot:		
			plt.plot(history.history['loss'])
			plt.title('model loss')
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.show()

		return history
	def act(self, state):
		return self.model.predict(state)
	def save(self):
		pass
	def load(self):
		pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	args = parser.parse_args()

	with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
		expert_data = pickle.load(f)

	obs = expert_data['observations']
	actions = expert_data['actions']
	actions = actions.squeeze()

	print("obs shape {}".format(obs.shape))
	print("action shape {}".format(actions.shape))


	agent = Agent(args.envname, obs.shape[1], actions.shape[1])
	agent.train(obs, actions, plot=True)


	print("------------------TEST--------------------")
        
	env = gym.make(args.envname)
	max_steps = env.spec.timestep_limit

	returns = []
	observations = []
	actions = []
	for i in range(10):
		print('iter', i)
		obs = env.reset()
		done = False
		totalr = 0.
		steps = 0
		while not done:
			action = agent.act(obs[None,:])
			observations.append(obs)
			actions.append(action)
			obs, r, done, _ = env.step(action)
			totalr += r
			steps += 1
			if args.render:
				env.render()
			if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
			if steps >= max_steps:
				break
		returns.append(totalr)

	print('returns', returns)
	print('mean return', np.mean(returns))
	print('std of return', np.std(returns))


if __name__ == '__main__':
	main()


