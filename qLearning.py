#!/usr/bin/python

import gym
import numpy as np


D_RANGE = np.arange(1, 20) # Constant over one epoch
GAMMA = 0.99
ALPHA = 0.1
N_EPISODES = 10

Q_TABLE = {}
VALUE_INIT = 0

ENV = gym.make("LunarLander-v2")
INPUT_DIM = ENV.reset().shape[0]
N_ACTIONS = ENV.action_space.n
ACTIONS = np.arange(0, N_ACTIONS)

def best_action(state):
	if state not in Q_TABLE:
		# Bookkeeping: Revisiting states in an exploding state-space
		revisit_state = 0
		# Initialize q function; Do random action
		q_function = {}
		for A in ACTIONS: q_function[A] = VALUE_INIT
		Q_TABLE[state] = q_function
		action = np.random.choice(ACTIONS, 1)[0]
	else: 
		revisit_state = 1
		# Select action according to max q
		action = max(Q_TABLE[state], key=Q_TABLE[state].get)
	# Get q value for action selected
	q = Q_TABLE[state][action]
	return action, q, revisit_state


def train_00(render):
	# Start a new epoch
	revisited_state = 0
	episode = 0
	success = 0
	solved = False

	for episode in range(N_EPISODES):
		
		x_t = ENV.reset()
		s_t = tuple(x_t)
		done = False

		while not done:
			if render: ENV.render()
	
			# Look up values for each action at s_t
			a_t, q, r_s  = best_action(s_t)
			
			# Observe after action a_t
			x_t, r_t, done, info = ENV.step(a_t)
			s_t1 = tuple(x_t)
		
			a_t1, Q_sa, _ = best_action(s_t1)
			
			# Update q for s_t and a_t, in hindsight
			Q_TABLE[s_t][a_t] = q + ALPHA * (r_t + GAMMA * Q_sa - q)
			
			# Update state and episode
			s_t = s_t1
			episode += 1
			revisited_state += r_s
			
			print episode, revisited_state

train_00(False)			