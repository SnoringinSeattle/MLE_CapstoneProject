#!/usr/bin/python

from __future__ import print_function
import gym, random, json, argparse
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import backend as K
import skimage as skimage
from skimage import transform, color, exposure
from os import path


SAVE_PATH = "/Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/05_Storage"
MH5 = path.join(SAVE_PATH, "model.h5")
MJS = path.join(SAVE_PATH, "model.json")
EXPLORATION_STEPS = float(1e6) # dqn, p.6
TOTAL_STEPS = float(10e6) # dqn, p.6
REPLAY_MEMORY = 1e6 # dqn, p.6


def _parse_args():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("-env", type=str, required=True)
	parser.add_argument("-D", nargs='+', type=int, required=True)
	parser.add_argument('-epsilon', nargs='+', type=float, required=True)
	parser.add_argument("-gamma", type=float, required=True)
	# To what extent the newly acquired information will override the old one: Learning rate alpha
	parser.add_argument("-alpha", type=float, required=True) 
	parser.add_argument("-batch_size", type=int, required=True)
	parser.add_argument("-fpa", type=int, required=True)
	parser.add_argument("-reward_clip", type=str, required=True)
	parser.add_argument("-render", type=str, required=True)
	return parser.parse_args() 
	

def preprocess_img(x, args, s_t=None):
	x = color.rgb2gray(x)
	x = transform.resize(x, (args.D[1], args.D[1]))
	x = exposure.rescale_intensity(x, out_range=(0,255))
	if np.any(s_t) == None:
		# Initial processing: Stack one single frame 4 times on top of each other
		s = np.stack([x for _ in range(args.D[0])], axis=0) #s = np.stack((x, x, x, x), axis=0) 
		s = s.reshape(1, s.shape[0], s.shape[1], s.shape[2])
	else:
		# Non-init processing: Replace 1 out of n frames
		x = x.reshape(1, 1, x.shape[0], x.shape[1])
		s = np.append(x, s_t[:, :args.D[0]-1, :, :], axis=1)
	return s


def _create_network(env, args):
	# Secure dimension ordering for Theano, even if running on Tensorflow
	K.set_image_dim_ordering('th')
	# Create model
	model = Sequential()
	input_shape = (args.D[0], args.D[1], args.D[1])
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', init='glorot_normal', activation='relu', input_shape=input_shape))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', init='glorot_normal', activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', init='glorot_normal', activation='relu'))
	model.add(Flatten()) 
	model.add(Dense(512, init='glorot_normal', activation='relu'))
	model.add(Dense(env.action_space.n, init='glorot_normal'))
	adam = Adam(lr=1e-6)
	model.compile(loss='mse', optimizer=adam) #model.compile(sgd(lr=self.learning_rate), "mse") ###
	return model


def train_network(model, env, args):
	actions = env.action_space.n
	# 1) Prepare training
	# Instantiate replay memory (D)
	D = deque()
	# Initialize epsilon and alpha
	epsilon = args.epsilon[0]
	alpha = args.alpha
	
	# Start episode
	# Initialize the number of episodes (E) played, reset step counter (t) over all episodes
	E = t = 0
	while t <= TOTAL_STEPS:
		env.reset()
		# Get the first state by doing random
		x_t, r_0, done, info = env.step(env.action_space.sample())
		# Preprocess the initial state to grayscale and resized dimensions, if 2d
		s_t = preprocess_img(x_t, args)  
		
		# Start training steps for each episode
		while not done:
			if args.render == 't': env.render()
			# Initialize trial variables
			loss = Q_sa = r_t = 0
			
			# Choose an action epsilon greedy if in training, if needed #frames has passed
			if t % args.fpa == 0:
				# Choose random action
				if random.random() <= epsilon:
					action_type = "R"
					a_t = random.randrange(actions)
				# Choose maxQ action
				else:
					action_type = "maxQ"
					q = model.predict(s_t)
					# Select action at index of highest Q
					a_t = np.argmax(q)
				
			# Run the selected action and observe next state and reward     
			x_t1_raw, r_t, done, info = env.step(a_t)
			s_t1 = preprocess_img(x_t1_raw, args, s_t)
			
			# Clip rewards
			if args.reward_clip == 't' and r_t != 0: r_t = abs(r_t)/r_t
				
			# Store the transition in D
			D.append((s_t, a_t, r_t, s_t1, done))
			if len(D) > REPLAY_MEMORY: D.popleft()
				
			# Anneal epsilon during exploration steps, and alpha during total steps
			# http://stackoverflow.com/questions/1854659/alpha-and-gamma-parameters-in-qlearning
			if epsilon > args.epsilon[1]: epsilon -= (args.epsilon[0] - args.epsilon[1]) / EXPLORATION_STEPS
			alpha -= alpha / TOTAL_STEPS
				
			# We only train when D is at least the size of the batch
			if t >= args.batch_size:
				# Sample minibatch to train on
				minibatch = random.sample(D, args.batch_size)
				# Instantiate inputs
				inputs = np.zeros((args.batch_size, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
				targets = np.zeros((args.batch_size, actions))
				
				# Perform experience replay on minibatch
				for i in range(0, len(minibatch)):
					state_t = minibatch[i][0]
					action_t = minibatch[i][1]
					reward_t = minibatch[i][2]
					state_t1 = minibatch[i][3]
					done = minibatch[i][4]
					inputs[i:i+1] = state_t
					
					# Compute target model (for supervised nn)
					targets[i] = model.predict(state_t) # Initialize targets for all actions
					Q_sa = model.predict(state_t1)
					targets[i, action_t] = args.alpha * (reward_t + args.gamma * np.max(Q_sa)) if not done else reward_t
			
				# Calculate loss for complete minibatch (backpropagation?) ###
				loss += model.train_on_batch(inputs, targets)
					
			# Move on with the transitions
			s_t = s_t1
			t += 1
			
			# Save progress every 100 iterations
			if t % 100 == 0:
				model.save_weights(MH5, overwrite=True)
				with open(MJS, "w") as outfile: json.dump(model.to_json(), outfile)
			
			# Print step info
			print("Episode: {}, Step: {}, Explore: {}, Action: {} ({}), Reward: {}, Loss: {}".format(E, t, t <= EXPLORATION_STEPS, a_t, action_type, r_t, loss))
			  
		E += 1
		
if __name__ == "__main__":
	args = _parse_args()
	
	# Prepare games
	env = gym.make(args.env)
	env.reset()
	net = _create_network(env, args)

	# Go for one epoch/episode
	train = train_network(net, env, args)