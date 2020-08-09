import numpy as np
import retro
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

np.random.seed(123)

env = None


def run_neat(mode="1p"):
    global env

    if mode == "1p":
        print("Starting 1P mode!")
        env = retro.make('Boxing-Atari2600', 'Start.state')
    if mode == "2p":
        print("Starting 2P mode!")
        env = retro.make('Boxing-Atari2600', 'boxing_2p.state', players=2)

    #env = gym.make('CartPole-v0')
    env.seed(123)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=10,
                   target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
    dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

    dqn.test(env, nb_episodes=5, visualize=True)


run_neat("1p")