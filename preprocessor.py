import numpy as np

class Preprocessor:

	def __init__(self, state_size, history_length = 4, ds = 3):
		self.state_size = state_size
		self.history_length = history_length		
		self.ds = ds

	def reset(self):
		self.history_states = np.zeros((self.state_size[0]/self.ds, self.state_size[1]/self.ds, self.history_length), dtype=float)

	def preprocess_memory(self, state):
		# uint images
		downsample_state = state[::self.ds, ::self.ds]
		grayscale_state = np.mean(downsample_state, axis=2).astype(np.uint8)
		grayscale_state = np.reshape(grayscale_state, (grayscale_state.shape[0], grayscale_state.shape[1], 1))
		return grayscale_state

	def preprocess_network(self, state):
		# float images
		downsample_state = state[::self.ds, ::self.ds]
		grayscale_state = np.mean(downsample_state, axis=2)
		# np.savetxt('test2.out', grayscale_state, fmt='%d')
		grayscale_state = np.reshape(grayscale_state, (grayscale_state.shape[0], grayscale_state.shape[1], 1))
		return grayscale_state

	def history_save(self, state):
		# push back state into history
		state = np.reshape(state,(state.shape[0], state.shape[1], 1))
		self.history_states = np.delete(self.history_states, 0, 2)
		self.history_states = np.append(self.history_states, state, 2)
		return self.history_states
