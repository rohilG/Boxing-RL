import retro
import numpy as np
import cv2
import neat
import pickle

env = None


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()
        action = env.action_space.sample()

        image_width, image_height, image_color = env.observation_space.shape

        # scale down image
        # Atari (or at least the emulator) is originally 210 x 160
        #
        # NOTE: scaling down further (such as by dividing by 16) will _considerably_ speed up training time
        # But this is at the cost of loss of information. Might be something to investigate.
        # The graphics in Boxing are primitive, but even then I'm not sure whether the AI actually benefits
        # in knowing where (for example) the heads and fists are, and maybe it's fine with just knowing the general
        # vicinity where the white and black blobs are located.
        #
        # Don't forget to change num_inputs in config if you do this
        image_width //= 16
        image_height //= 16

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        current_fitness = 0

        frame = 0
        counter = 0

        # score_diff = 0
        # score_diff_max = -100

        last_score1 = 0
        last_score2 = 0

        isGameOver = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not isGameOver:

            #scaledimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            #scaledimg = cv2.resize(scaledimg, (image_width, image_height))

            observation = cv2.resize(observation, (image_width, image_height))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (image_width, image_height))

            #cv2.imshow("main", scaledimg)
            #cv2.waitKey(1)

            # display image - commenting it out will speed up training (slightly)
            #env.render()

            imgarray = np.ndarray.flatten(observation)

            # output of neural network, which is an array representing the controller input
            nn_output = net.activate(imgarray)
            #print(nnOutput)

            observation, reward, isGameOver, info = env.step(nn_output)

            score1 = info['score1']
            score2 = info['score2']

            minutes = info['minutes']
            seconds = info['seconds']

            # currently using difference between scores
            # NOTE: might be better to use score in a given time? or something like - 1 point for landing a hit and -0.5 for getting hit
            # (a score of 0-10 is probably worse than 10-20)
            # also sometimes the player will score a bunch of points in quick succession before slowly losing their lead, might be
            # able to take advantage of these short bursts of the score difference rising by dividing it over time and give higher fitness

            # score_diff = score1 - score2

            # if score_diff > score_diff_max:
            #   counter = 0
            #   score_diff_max = score_diff
            # else:
            #   counter += 1
            '''
            if score1 > last_score1:
                current_fitness += score1 - last_score1
            '''

            if last_score1 == score1 and last_score2 == score2:
                counter += 1
            else:
                score1 = min(score1, 98)
                score2 = min(score2, 98)

                last_score1 = score1
                last_score2 = score2
                counter = 0

            #score_fitness = ((score1 - score2 + 100)//2) + (100 * score1) // max(score2, 1)
            current_fitness = score1

            # stop training if score has not improved for 1000 frames
            # NOTE: maybe this value should be changed? or have some sort of other metric to decide when to stop game
            # if counter == 10000:
            #   done = True

            # if minutes == 0 and seconds == 0:
            #   done = True

            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness

            if score1 >= 98 or score2 >= 98:
                isGameOver = True

            if isGameOver:
                # NOTE: I'm not sure whether for NEAT you can have a negative fitness score
                # So here 0 fitness is 1p losing by 100 points, 100 fitness is a tie, and 200 is 1p winning by 100 points
                #
                # I was considering doing something like - if 1p is losing, the fitness is 1 / abs(score_diff)
                # so that all losing score are within 0 to 1 and 1 to 100 is for when player is winning
                # but I'm not sure how important this is, and whether NEAT is ok with floating numbers for fitness (though this can be fixed pretty easily)
                #
                # I just have a suspiction that this value could be pretty important (though I could be totally wrong)
                # current_fitness = score_diff + 100
                print("Genome ID ", genome_id, "Fitness Achieved ",
                      current_fitness)

            genome.fitness = current_fitness  # score_fitness


def run_neat(mode="1p"):
    global env

    if mode == "1p":
        print("Starting 1P mode!")
        env = retro.make('Boxing-Atari2600', 'Start.state')
    if mode == "2p":
        print("Starting 2P mode!")
        env = retro.make('Boxing-Atari2600', 'boxing_2p.state', players=2)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    p = neat.Population(config)

    # print statistics after each generation
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # adds a checkpoint every 5 generations - generates a file that can later be reused if you don't want to restart from scratch after making a change
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes)


run_neat("1p")