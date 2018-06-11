import numpy as np
from numpy import array
import random

class ReplayMemory:

    def __init__(self, size = 10000, history_length = 4, state_shape = [100, 100]):
        self.size = size
        self.history_length = history_length
        self.state_shape = state_shape
        self.data = [None] * (size)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def sample(self, batch_size = 32):
        max_num = self.size - 1 if self.end < self.start else self.end - 1
        nums = range(0, max_num)
        chosen_nums = random.sample(nums, batch_size)

        states = []
        actions = []
        rewards = []
        nstates = []
        is_terminals = []

        for i in chosen_nums:

            if self.end > self.start and i < 3:
                i = 3

            actions.append(self.data[i][1])
            rewards.append(self.data[i][2])
            is_terminals.append(self.data[i][3])

            if self.history_length == 4:
                stack_state = np.concatenate((self.data[i - 3][0], self.data[i - 2][0], self.data[i - 1][0], self.data[i][0]), axis=2)
                stack_nstate = np.concatenate((self.data[i - 2][0], self.data[i - 1][0], self.data[i][0], self.data[i+1][0] ), axis=2)
            else:
                print "Error in history length"
                            
            states.append(stack_state.astype(float))
            nstates.append(stack_nstate.astype(float))

        return states, actions, rewards, nstates, is_terminals

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
