import gym
import numpy as np
import tensorflow as tf

class DeepQNetwork:
	
	def __init__(self,
                 qnet_online,
                 qnet_target,
                 qnet_test,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 batch_size):
		self.qnet_online = qnet_online
		self.qnet_target = qnet_target
		self.qnet_test = qnet_test
		self.preprocessor = preprocessor
		self.memory = memory
		self.gamma = gamma
		self.policy = policy
		self.target_update_freq = target_update_freq
		self.batch_size = batch_size
		self.num_actions = 4

 	def select_action(self, state):
 		zero_state = np.zeros(state.shape)  	
 		input_state = np.concatenate((zero_state, zero_state, zero_state, state), axis=2)
 		input_state = np.reshape(input_state, [1, input_state.shape[0], input_state.shape[1], input_state.shape[2]])
		q_values = self.qnet_target.predict_val(input_state)
		return self.policy.get_lineargreedyeps(q_values[0]) 

	def fit(self, env, num_iterations = 1000000, max_episode_length=1000):
		global_count = 0
		for i in range(0, num_iterations):
			state = env.reset()
			returns = 0
			done = False
			count = 0
			while(done == False & count < max_episode_length):
				count += 1
				global_count += 1

				env.render()

				network_state = self.preprocessor.preprocess_network(state)
				action = self.select_action(network_state)
				next_state, reward, done, _ = env.step(action)

				memory_state = self.preprocessor.preprocess_memory(state)
				self.memory.append([memory_state, action, reward, done])

				state = next_state

				if(global_count % self.target_update_freq == 0 and global_count != 0):
					print "Updating target"
					self.qnet_target.set_weights("online")

				# one more than batch size to sample next state without error
				if(global_count <= self.batch_size):
					continue

				states, actions, rewards, nstates, is_terminals = self.memory.sample(self.batch_size)
				target = self.calc_target(states, actions, rewards, nstates, is_terminals)

				self.qnet_online.train_net(states, target)

				returns += reward

			print "Episode : ", i, "return : ", returns

	def calc_target(self, states, actions, rewards, nstates, is_terminals):
		q_values = self.qnet_online.predict_val(states)
		nq_values = self.qnet_target.predict_val(nstates)

		for i in range(0, len(states)):
		  action = actions[i]
		  reward = rewards[i]
		  if is_terminals[i] == True:
		  	q_values[i,action] = reward
		  else:
		  	q_values[i,action] = reward + self.gamma*(np.max(nq_values[i]))


		return q_values
