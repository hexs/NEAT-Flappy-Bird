import cv2
import numpy as np
import random
import os
import neat
import pickle
import time
from hexss.constants.terminal_color import *

WIN_WIDTH = 600
WIN_HEIGHT = 800
BG_COLOR = (250, 206, 135)
PIPE_COLOR = (0, 255, 0)
BIRD_WIDTH = 30
BIRD_HEIGHT = 30
PIPE_WIDTH = 80
gen = 0
id_list = []


def convert_color(color):
    return (color[2], color[1], color[0])


def rect_collide(r1, r2):
    return r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1]


class Bird:
    def __init__(self, color=(255, 0, 0)):
        self.x = 230
        self.y = 350
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.color = convert_color(color)

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 0.5 * 3 * self.tick_count ** 2
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2
        self.y += d

    def draw(self, frame):
        cv2.rectangle(frame, (int(self.x), int(self.y)), (int(self.x + BIRD_WIDTH), int(self.y + BIRD_HEIGHT)),
                      self.color, -1)

    def get_rect(self):
        return (self.x, self.y, self.x + BIRD_WIDTH, self.y + BIRD_HEIGHT)


class Pipe:
    GAP = 200
    VEL = 10

    def __init__(self, x):
        self.x = x
        self.height = random.randrange(50, 450)
        self.passed = False

    def move(self):
        self.x -= self.VEL

    def draw(self, frame):
        top_rect = (int(self.x), 0, int(self.x + PIPE_WIDTH), int(self.height))
        bottom_rect = (int(self.x), int(self.height + self.GAP), int(self.x + PIPE_WIDTH), WIN_HEIGHT)
        cv2.rectangle(frame, (top_rect[0], top_rect[1]), (top_rect[2], top_rect[3]), PIPE_COLOR, -1)
        cv2.rectangle(frame, (bottom_rect[0], bottom_rect[1]), (bottom_rect[2], bottom_rect[3]), PIPE_COLOR, -1)

    def collide(self, bird):
        b = bird.get_rect()
        top = (self.x, 0, self.x + PIPE_WIDTH, self.height)
        bottom = (self.x, self.height + self.GAP, self.x + PIPE_WIDTH, WIN_HEIGHT)
        return rect_collide(b, top) or rect_collide(b, bottom)


def draw_window(birds, pipes, score, gen_val):
    frame = np.full((WIN_HEIGHT, WIN_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    for pipe in pipes:
        pipe.draw(frame)
    for bird in birds:
        bird.draw(frame)
    cv2.putText(frame, "Score: " + str(score), (WIN_WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Gens: " + str(gen_val), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Alive: " + str(len(birds)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Flappy Bird", frame)
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
        exit()


def eval_genomes(genomes, config):
    print('------------ eval_genomes ------------')
    global gen, id_list
    gen += 1
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cnt = id_list.count(genome_id)
        if cnt == 0:
            birds.append(Bird())
        elif cnt == 1:
            birds.append(Bird((0, 0, 255)))
        elif cnt == 2:
            birds.append(Bird((0, 50, 255)))
        elif cnt == 3:
            birds.append(Bird((0, 100, 255)))
        elif cnt == 4:
            birds.append(Bird((0, 150, 255)))
        elif cnt == 5:
            birds.append(Bird((0, 200, 255)))
        else:
            birds.append(Bird((0, 255, 255)))
        id_list.append(genome_id)
        ge.append(genome)
    pipes = [Pipe(700)]
    score = 0
    play = True
    while play and len(birds) > 0:
        pipe_ind = 0
        if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
            pipe_ind = 1
        for i, bird in enumerate(birds):
            ge[i].fitness += 0.1
            bird.move()
            output = nets[birds.index(bird)].activate((
                bird.y, pipes[pipe_ind].height, pipes[pipe_ind].height + Pipe.GAP
            ))
            if output[0] > 0.5:
                bird.jump()
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
            if pipe.x + PIPE_WIDTH < 0:
                rem.append(pipe)
            if birds and not pipe.passed and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True
        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))
        for r in rem:
            pipes.remove(r)
        for bird in birds:
            if bird.y < 0 or WIN_HEIGHT < bird.y + BIRD_HEIGHT:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
        draw_window(birds, pipes, score, gen)
        if score > 20:
            pickle.dump(nets[0], open("best.pickle", "wb"))
            print(f'{GREEN}save best.pickle{END}')
            break


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))


def run_best():
    with open("best.pickle", "rb") as f:
        net = pickle.load(f)
    bird = Bird()
    pipes = [Pipe(700)]
    score = 0
    play = True
    while play:
        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + PIPE_WIDTH:
            pipe_ind = 1
        bird.move()
        output = net.activate((bird.y, pipes[pipe_ind].height, pipes[pipe_ind].height + Pipe.GAP))
        if output[0] > 0.5:
            bird.jump()
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                play = False
                print('pipe.collide(bird)')
            if pipe.x + PIPE_WIDTH < 0:
                pipes.remove(pipe)
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                score += 1
                pipes.append(Pipe(WIN_WIDTH))
        if bird.y < 0 or WIN_HEIGHT < bird.y + BIRD_HEIGHT:
            play = False
            print('out')
        draw_window([bird], pipes, score, -1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    # run(config_path)
    run_best()
