import math
import random


class Particle:

    def __init__(self, x, y, inertia, cognitive_constant, social_constant):
        self.x = x
        self.y = y
        self.fitness = math.inf
        self.velocity = 0
        self.best_x = x
        self.best_y = y
        self.best_fitness = math.inf
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.update_fitness()

    def get_fitness(self):
        return self.fitness

    @staticmethod
    def calculate_booth(x, y):
        return math.pow(x + 2 * y - 7, 2) + math.pow(2 * x + y - 5, 2)

    def update_fitness(self):
        self.fitness = self.calculate_booth(self.x, self.y)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_x = self.x
            self.best_y = self.y

    @staticmethod
    def calculate_inertia(inertia, current_velocity):
        return inertia * current_velocity

    def calculate_cognitive(self,
                            cognitive_constant,
                            cognitive_random,
                            particle_best_position_x,
                            particle_best_position_y,
                            particle_current_position_x,
                            particle_current_position_y):
        cognitive_acceleration = self.calculate_acceleration(cognitive_constant, cognitive_random)
        cognitive_distance = math.sqrt(((particle_best_position_x-particle_current_position_x)**2)
                                       + ((particle_best_position_y-particle_current_position_y)**2))
        return cognitive_acceleration * cognitive_distance

    def calculate_social(self,
                         social_constant,
                         social_random,
                         swarm_best_position_x,
                         swarm_best_position_y,
                         particle_current_position_x,
                         particle_current_position_y):
        social_acceleration = self.calculate_acceleration(social_constant, social_random)
        social_distance = math.sqrt(((swarm_best_position_x-particle_current_position_x)**2)
                                    + ((swarm_best_position_y-particle_current_position_y)**2))
        return social_acceleration * social_distance

    @staticmethod
    def calculate_acceleration(constant, random_factor):
        return constant * random_factor

    @staticmethod
    def calculate_updated_velocity(inertia, cognitive, social):
        return inertia + cognitive + social

    @staticmethod
    def calculate_position(current_position_x, current_position_y, updated_velocity):
        return current_position_x + updated_velocity, current_position_y + updated_velocity

    def update(self, swarm_best_x, swarm_best_y):
        i = self.calculate_inertia(self.inertia, self.velocity)
        print('Inertia: ', i)
        c = self.calculate_cognitive(self.cognitive_constant, random.random(), self.x, self.y, self.best_x, self.best_y)
        print('Cognitive: ', c)
        s = self.calculate_social(self.social_constant, random.random(), self.x, self.y, swarm_best_x, swarm_best_y)
        print('Social: ', s)
        v = self.calculate_updated_velocity(i, c, s)
        self.velocity = v
        print('Velocity: ', v)
        p = self.calculate_position(self.x, self.y, v)
        self.x = p[0]
        self.y = p[1]
        print('Position: ', p)

    def to_string(self):
        print('Inertia: ', self.inertia)
        print('Velocity: ', self.velocity)
        print('Position: ', self.x, ',', self.y)


class Swarm:

    def __init__(self,
                 inertia,
                 cognitive_constant,
                 social_constant,
                 random_swarm,
                 number_of_particles,
                 number_of_iterations):
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.number_of_iterations = number_of_iterations
        self.swarm = []
        if random_swarm:
            self.swarm = self.get_random_swarm(number_of_particles)
        else:
            self.swarm = self.get_sample_swarm()

    def get_best_in_swarm(self):
        best = math.inf
        best_particle = None
        for p in self.swarm:
            p.update_fitness()
            if p.fitness < best:
                best = p.fitness
                best_particle = p
        return best_particle

    @staticmethod
    def get_sample_swarm():
        p1 = Particle(7, 1, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p2 = Particle(-1, 9, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p3 = Particle(5, -1, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p4 = Particle(-2, -5, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        particles = [p1, p2, p3, p4]
        return particles

    @staticmethod
    def get_random_swarm(number_of_particles):
        particles = []
        for p in range(number_of_particles):
            particles.append(Particle(random.randint(-10, 10),
                                      random.randint(-10, 10),
                                      INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT))
        return particles

    def run_pso(self):
        for t in range(0, self.number_of_iterations):
            best_particle = self.get_best_in_swarm()
            for p in self.swarm:
                p.update(best_particle.x, best_particle.y)
            print('BEST: ', best_particle.fitness)


INERTIA = 0.4
COGNITIVE_CONSTANT = 0.3
SOCIAL_CONSTANT = 0.7
RANDOM_CHANCE = True
NUMBER_OF_PARTICLES = 200
NUMBER_OF_ITERATIONS = 500

swarm = Swarm(INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT, RANDOM_CHANCE, NUMBER_OF_PARTICLES, NUMBER_OF_ITERATIONS)
swarm.run_pso()
