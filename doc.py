#!/usr/bin/python

python /Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/04_DQN_PG/DQN.py -env 'Pong-v0' -D 4 84 -epsilon 1 0.1 -gamma 0.99 -alpha 0.1 -batch_size 32 -fpa 4 -reward_clip 't' -render 't'


-- alpha (between 1 and 3)



import gym

env = gym.make('Acrobot-v1')
print(env.action_space.n)
for i_episode in range(1):
	observation = env.reset()
	print(env.observation_space)
	for t in range(1):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print(observation.shape)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
