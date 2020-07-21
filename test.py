import retro
import numpy as np
import cv2
import neat
import pickle


def eval_genomes(env, genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()

        image_width, image_height, image_color = env.observation_space.shape

        image_width //= 8
        image_height //= 8

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

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

            current_fitness = score1**2 + (score1 / max(score2, 1))**2

            genome.fitness = current_fitness

            if score1 >= 98 or score2 >= 98:
                done = True

        print("Score: ", score1, score2)
        print("Genome ID ", genome_id, "Fitness Achieved ", current_fitness)


env = retro.make('Boxing-Atari2600', 'Start.state')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)

# print statistics after each generation
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

winner = p.run(lambda genomes, config: eval_genomes(env, genomes, config))

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)