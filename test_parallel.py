import retro
import numpy as np
import cv2
import neat
import pickle

import visualize

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make('Boxing-Atari2600', 'Start.state')
        observation = self.env.reset()

        image_width, image_height, image_color = self.env.observation_space.shape

        image_width //= 8
        image_height //= 8

        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        current_fitness = 0

        done = False
        while not done:
            #env.render()

            observation = cv2.resize(observation, (image_width, image_height))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (image_width, image_height))

            imgarray = np.ndarray.flatten(observation)

            action = net.activate(imgarray)

            observation, reward, done, info = self.env.step(action)

            score1 = info['score1']
            score2 = info['score2']

            score1 = min(score1, 98)
            score2 = min(score2, 98)

            current_fitness = (score1 - score2)**2
            if score1 < score2:
                current_fitness = - current_fitness
            self.genome.fitness = current_fitness

            if score1 >= 98 or score2 >= 98:
                done = True
        return current_fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)

# print statistics after each generation
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

pe = neat.ParallelEvaluator(5, eval_genomes)

winner = p.run(pe.evaluate, 10)

visualize.draw_net(config, winner, True)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)