import numpy as np

class Policy():

	def __init__(self, num_actions = 4, start_eps = 0.8, end_eps = 0.2, num_steps = 100):
		self.num_actions = num_actions
		self.init_eps = start_eps
		self.start_eps = start_eps
		self.end_eps = end_eps
		self.num_steps = num_steps

	def get_random(self):
		return np.random.randint(0, self.num_actions)

	def get_greedy(self, q_values):
		return np.argmax(q_values)

	def get_greedyeps(self, q_values):
		rand = np.random.uniform(0, 1)

		if(rand <= self.init_eps):
			return np.random.randint(0, self.num_actions)
		else:
			return np.argmax(q_values)

	def get_lineargreedyeps(self, q_values):
		if(self.start_eps <= self.end_eps):
			self.start_eps = self.init_eps
        
		self.start_eps = self.start_eps - (self.init_eps - self.end_eps)/self.num_steps
        
		rand = np.random.uniform(0, 1)

		if(rand <= self.start_eps):
			return np.random.randint(0, self.num_actions)
		else:
			return np.argmax(q_values)
