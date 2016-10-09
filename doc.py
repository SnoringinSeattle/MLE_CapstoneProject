#!/usr/bin/python

python /Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/04_DQN_PG/DQN.py -env 'Pong-v0' -D 4 84 -epsilon 1 0.1 -gamma 0.99 -alpha 0.1 -batch_size 32 -fpa 4 -reward_clip 't' -render 't'

python /Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/04_DQN_PG/DQN.py -env 'LunarLander-v2' -D 4 84 -epsilon 1 0.1 -gamma 0.99 -alpha 0.1 -batch_size 32 -fpa 4 -reward_clip 't' -render 't'

# -- alpha (between 1 and 3)

## Box2d for LunarLander:
# https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md

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

http://www.cloudera.com/documentation/enterprise/5-5-x/topics/spark_python.html
https://aws.amazon.com/emr/details/spark/
https://districtdatalabs.silvrback.com/getting-started-with-spark-in-python
https://github.com/osh/kerlym/blob/master/kerlym/networks.py
http://karpathy.github.io/2016/05/31/rl/
https://gist.github.com/cadurosar/bd54c723c1d6335a43c8
https://gist.github.com/EderSantana/1ad56b7720af8d706e7f22cbcb8c6d70
reinforcement learning policy gradient
https://keras.io/backend/
https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html#encoding-and-decoding