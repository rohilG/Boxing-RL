import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
import retro  # Retro Environment

from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames

import matplotlib.pyplot as plt  # Display graphs

from collections import deque  # Ordered collection with ends

import random

import warnings  # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

# Create our environment
env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(
    np.identity(env.action_space.n, dtype=int).tolist())


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame  # 110x84x1 frame


stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque(
    [np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque(
            [np.zeros((110, 84), dtype=np.int) for i in range(stack_size)],
            maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)


### MODEL HYPERPARAMETERS
state_size = [
    110, 84, 4
]  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n  # 8 possible actions
learning_rate = 0.00025  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50  # Total episodes for training
max_steps = 50000  # Max possible steps in an episode
batch_size = 64  # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9  # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4  # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

############################


class DQNetwork:
    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate,
                 name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size],
                                          name="inputs")
            self.actions_ = tf.placeholder(tf.float32,
                                           [None, self.action_size],
                                           name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.
                                          xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.
                                          xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.
                                          xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=self.action_size,
                activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)
