import pygame
import random
import os
import neat
import pickle
from hexss.constants.terminal_color import *

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800

STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

BG_COLOR = (135, 206, 250)
PIPE_COLOR = (0, 255, 0)

BIRD_WIDTH = 30
BIRD_HEIGHT = 30
PIPE_WIDTH = 80

gen = 0


class Bird:
    def __init__(self, color=(255, 0, 0)):
        self.x = 230
        self.y = 350
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.color = color

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

    def draw(self, display):
        pygame.draw.rect(display, self.color, (self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT)


class Pipe:
    GAP = 200
    VEL = 10

    def __init__(self, x):
        self.x = x
        self.height = random.randrange(50, 450)
        self.passed = False

    def move(self):
        self.x -= self.VEL

    def draw(self, display):
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.GAP, PIPE_WIDTH, WIN_HEIGHT - (self.height + self.GAP))
        pygame.draw.rect(display, PIPE_COLOR, top_rect)
        pygame.draw.rect(display, PIPE_COLOR, bottom_rect)

    def collide(self, bird):
        bird_rect = bird.get_rect()
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.GAP, PIPE_WIDTH, WIN_HEIGHT - (self.height + self.GAP))
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


def draw_window(display, birds, pipes, score, gen, pipe_ind):
    display.fill(BG_COLOR)
    for pipe in pipes:
        pipe.draw(display)
    for bird in birds:
        bird.draw(display)
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    display.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    gens_label = STAT_FONT.render("Gens: " + str(gen), 1, (255, 255, 255))
    display.blit(gens_label, (10, 10))
    alive_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    display.blit(alive_label, (10, 50))
    pygame.display.update()


id_list = []


def eval_genomes(genomes, config):
    print('------------ eval_genomes ------------')
    global display, gen, id_list
    gen += 1
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        count = id_list.count(genome_id)
        if count == 0:
            birds.append(Bird())
        elif count == 1:
            birds.append(Bird((0, 0, 255)))
        elif count == 2:
            birds.append(Bird((0, 50, 255)))
        elif count == 3:
            birds.append(Bird((0, 100, 255)))
        elif count == 4:
            birds.append(Bird((0, 150, 255)))
        elif count == 5:
            birds.append(Bird((0, 200, 255)))
        else:
            birds.append(Bird((0, 255, 255)))
        id_list.append(genome_id)
        ge.append(genome)

    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()
    play = True
    while play and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
        pipe_ind = 0
        if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
            pipe_ind = 1
        for i, bird in enumerate(birds):
            ge[i].fitness += 0.1
            bird.move()
            output = nets[birds.index(bird)].activate((
                bird.y,
                pipes[pipe_ind].height,
                pipes[pipe_ind].height + Pipe.GAP
            ))

            # bird.y,
            # abs(bird.y - pipes[pipe_ind].height),
            # abs(bird.y - (pipes[pipe_ind].height + Pipe.GAP))

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
        draw_window(display, birds, pipes, score, gen, pipe_ind)
        if score > 20:
            pickle.dump(nets[0], open("best.pickle", "wb"))
            print(f'{GREEN}save best.pickle{END}')
            break


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
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
    clock = pygame.time.Clock()
    play = True
    while play:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + PIPE_WIDTH:
            pipe_ind = 1
        bird.move()
        output = net.activate((
            bird.y,
            pipes[pipe_ind].height,
            pipes[pipe_ind].height + Pipe.GAP
        ))

        if output[0] > 0.5:
            bird.jump()

        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                play = False
            if pipe.x + PIPE_WIDTH < 0:
                pipes.remove(pipe)
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                score += 1
                pipes.append(Pipe(WIN_WIDTH))

        if bird.y < 0 or WIN_HEIGHT < bird.y + BIRD_HEIGHT:
            play = False
        draw_window(display, [bird], pipes, score, -1, pipe_ind)
    pygame.quit()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    run(config_path)
    run_best()
