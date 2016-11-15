#!/usr/bin/python

'''
def _create_network():
	model = Sequential()
	model.add(Dense(NUM_HIDDEN_NEURON, init=INITIALIZATION, input_shape=(STEP_MEM*NUM_INPUT,))) 
	model.add(Activation(ACTIVATION))
	model.add(Dense(NUM_ACTION, init=INITIALIZATION))
	model.compile(optimizer='rmsprop', loss='mse') # sgd(lr=self.learning_rate)
	return model

'''


from keras.models import model_from_json

'''
###NEW Set up model, weight, and stats save
	MH5 = path.join(SAVE_PATH, "DQN_"+EPOCH+".h5")
	MJS = path.join(SAVE_PATH, "DQN_"+EPOCH+".json") 
	
		# Save model and model weights
		with open(MJS, "w") as outfile: json.dump(model.to_json(), outfile) 
		model.save_weights(MH5_V, overwrite=True)
'''

'''
	if args['mode'] == 'Run':
		OBSERVE = 999999999    #We keep observe, never train
		epsilon = FINAL_EPSILON
		print ("Now we load weight")
		model.load_weights("model.h5")
		adam = Adam(lr=1e-6)
		model.compile(loss='mse',optimizer=adam)
		print ("Weight load successfully")  
		
# Visualize
stdout.write("\rEpoch {}, Maximum Reward {}, Successful Episodes {}, Solved Episodes {}".format(\
	EPOCH, highest_reward, success_episodes, solved_episodes))
'''

#INITIALIZATIONS = [] # Constant over one epoch
# 'uniform', 'lecun_uniform', 'normal', 'identity', 'orthogonal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'
#ACTIVATIONS = ['softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] # Constant over one epoch


# Take action with highest estimated reward with probability 1-epsilon ("epsilon greedy" policy)
###NEW Act only every n-th step
if step % SPA == 0: a_t = np.argmax(q) if np.random.random() > epsilon else np.random.choice(ACTIONS, 1)[0]


###NEW Clip rewards
if R_CLIP and r_t != 0: r_t = abs(r_t) / r_t


# Plot learning process

'''
%matplotlib inline
import matplotlib.pyplot as plt


episodes = [v+1 for v in range(NUM_EPISODE)]
titles = ['Sum Reward', 'Steps', '# Successes', '# Solved']
fontsize = 14

from os import listdir
file_count = 0
for file in listdir(SAVE_PATH):
	if "AAC_Stats" in file: # only AAC ###TODO
		file_count += 1
		json_data = open(path.join(SAVE_PATH, file))
		data = json.load(json_data)
		
		plt.figure(file_count, figsize=(18,6), dpi=320)
		for i in range(len(titles)):
			plt.subplot(2,2,i+1)
			plt.plot(episodes, [ v[i] for v in chain.from_iterable([v.values() for v in data.values()]) ], color='blue')
			plt.xlabel("Episodes", fontsize=fontsize)
			plt.ylabel("AAC: "+titles[i], fontsize=fontsize)
'''




