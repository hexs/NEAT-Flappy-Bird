import pygame
import random
import os
import neat
import pickle

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

BG_COLOR = (135, 206, 250)
BIRD_COLOR = (255, 0, 0)
PIPE_COLOR = (0, 255, 0)
BASE_COLOR = (139, 69, 19)

BIRD_WIDTH = 30
BIRD_HEIGHT = 30
PIPE_WIDTH = 80

gen = 0


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.vel = 0
        self.height = self.y

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel * self.tick_count + 0.5 * 3 * self.tick_count ** 2
        if displacement >= 16:
            displacement = 16
        if displacement < 0:
            displacement -= 2
        self.y += displacement

    def draw(self, display):
        pygame.draw.rect(display, BIRD_COLOR, (self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)

    def move(self):
        self.x -= self.VEL

    def draw(self, display):
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.GAP, PIPE_WIDTH, FLOOR - (self.height + self.GAP))
        pygame.draw.rect(display, PIPE_COLOR, top_rect)
        pygame.draw.rect(display, PIPE_COLOR, bottom_rect)

    def collide(self, bird):
        bird_rect = bird.get_rect()
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.GAP, PIPE_WIDTH, FLOOR - (self.height + self.GAP))
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


class Base:
    VEL = 5

    def __init__(self, y):
        self.y = y
        self.x = 0

    def move(self):
        self.x -= self.VEL
        if self.x <= -WIN_WIDTH:
            self.x = 0

    def draw(self, display):
        pygame.draw.rect(display, BASE_COLOR, (self.x, self.y, WIN_WIDTH, WIN_HEIGHT - self.y))
        pygame.draw.rect(display, BASE_COLOR, (self.x + WIN_WIDTH, self.y, WIN_WIDTH, WIN_HEIGHT - self.y))


def draw_window(display, birds, pipes, base, score, gen, pipe_ind):
    display.fill(BG_COLOR)
    for pipe in pipes:
        pipe.draw(display)
    base.draw(display)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(display, (255, 0, 0),
                                 (bird.x + BIRD_WIDTH / 2, bird.y + BIRD_HEIGHT / 2),
                                 (pipes[pipe_ind].x + PIPE_WIDTH / 2, pipes[pipe_ind].height),
                                 5)
                pygame.draw.line(display, (255, 0, 0),
                                 (bird.x + BIRD_WIDTH / 2, bird.y + BIRD_HEIGHT / 2),
                                 (pipes[pipe_ind].x + PIPE_WIDTH / 2, pipes[pipe_ind].height + Pipe.GAP),
                                 5)
            except:
                pass
        bird.draw(display)
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    display.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    gens_label = STAT_FONT.render("Gens: " + str(gen - 1), 1, (255, 255, 255))
    display.blit(gens_label, (10, 10))
    alive_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    display.blit(alive_label, (10, 50))
    pygame.display.update()


def eval_genomes(genomes, config):
    global display, gen
    gen += 1
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()
    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        pipe_ind = 0
        if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
            pipe_ind = 1
        for i, bird in enumerate(birds):
            ge[i].fitness += 0.1
            bird.move()
            output = nets[birds.index(bird)].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - (pipes[pipe_ind].height + Pipe.GAP)))
            )
            if output[0] > 0.5:
                bird.jump()
        base.move()
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
            if bird.y + BIRD_HEIGHT >= FLOOR or bird.y < 0:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
        draw_window(display, birds, pipes, base, score, gen, pipe_ind)
        if score > 20:
            pickle.dump(nets[0], open("best.pickle", "wb"))
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


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
