import os
import sys

import retro
import numpy as np
import cv2
import neat
import pickle

import visualize

if len(sys.argv) != 2:
    print("Give me a unique folder name")
    print("WTF are you doing with your life.")
    exit(1)

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        env = retro.make('Boxing-Atari2600', 'Start.state')
        observation = env.reset()

        image_width, image_height, image_color = env.observation_space.shape

        image_width //= 8
        image_height //= 8

        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        score1 = 0
        score2 = 0
        current_fitness = 0

        done = False
        while not done:
            #env.render()

            observation = cv2.resize(observation, (image_width, image_height))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (image_width, image_height))

            imgarray = np.ndarray.flatten(observation)

            action = net.activate(imgarray)

            observation, reward, done, info = env.step(action)

            score1 = info['score1']
            score2 = info['score2']

            score1 = min(score1, 98)
            score2 = min(score2, 98)

            current_fitness = score1 - score2

            # cur_fitness_v1 = -(100 * score1 / 98)
            # cur_fitness_v2 = 0
            # 
            # if score1 < score2:
            #     cur_fitness_v2 = 100 * score1 / (score2 * 2)
            # elif score1 > score2:
            #     cur_fitness_v2 = 50 + (100 * ((score1 - score2)/(100 - score2)) / 2)
            # else:
            #     cur_fitness_v2 = 50
            # 
            # current_fitness = max(cur_fitness_v1, cur_fitness_v2)
            
            self.genome.fitness = current_fitness

            if score1 >= 98 or score2 >= 98:
                done = True
        
        if score1 > score2:
            current_fitness += 100
        self.genome.fitness = current_fitness

        env.close()
        return current_fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-parallel')
p = neat.Population(config)

# print statistics after each generation
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5, filename_prefix=sys.argv[1] + "/neat-checkpoint-"))

pe = neat.ParallelEvaluator(None, eval_genomes)

for i in range(8):
    winner = p.run(pe.evaluate, 5)

    visualize.draw_net(config, winner, view=False, filename=sys.argv[1] + "/checkpoint-" + str(i) + "/Digraph")
    visualize.plot_stats(stats, ylog=False, view=False, filename=sys.argv[1] + "/checkpoint-" + str(i) + "/avg_fitness.svg")
    visualize.plot_species(stats, view=False, filename=sys.argv[1] + "/checkpoint-" + str(i) + "/speciation.svg")

    with open(sys.argv[1] + "/checkpoint-" + str(i) + '/winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

exit(0)