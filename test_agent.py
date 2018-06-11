import gym
from gym.envs import snake
import numpy as np
import tensorflow as tf

import deepq
import networks
import policy
import preprocessor
import replay_memory

# define useful params
batch_size = 32
ds = 5
gamma = 0.99
learning_rate = 0.001
max_episode_length = 1000
num_burn_in = 10
num_iterations = 100000
target_update_freq = 100
train_freq = 10

# Initialize Tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.InteractiveSession(config=config)

# load snake environment
env = snake.SnakeEnv()
state_size = env.get_state_size()
num_actions = env.get_num_actions()

# make the preprocessors
prep = preprocessor.Preprocessor(state_size = state_size, ds = 5)

# make the replay memory
mem = replay_memory.ReplayMemory()

# make the policy
pol = policy.Policy(num_actions = 4, start_eps = 0.90, end_eps = 0.05, num_steps = 10000)

# make the q networks
qnet_online = networks.ConvNet(session, num_classes = num_actions, learning_rate = learning_rate, net_name = "online")
qnet_target = networks.ConvNet(session, num_classes = num_actions, learning_rate = learning_rate, net_name = "target")
qnet_test   = networks.ConvNet(session, num_classes = num_actions, learning_rate = learning_rate, net_name = "test")

# make the networks
qnet_online.build_network()
qnet_target.build_network()
qnet_test.build_network()

# initialize global vars
session.run(tf.global_variables_initializer())

# make the dqn agent
agent = deepq.DeepQNetwork(qnet_online, qnet_target, qnet_test, prep, mem, pol, gamma, target_update_freq, num_burn_in, train_freq, batch_size)
agent.fit(env = env, num_iterations = num_iterations, max_episode_length = max_episode_length)