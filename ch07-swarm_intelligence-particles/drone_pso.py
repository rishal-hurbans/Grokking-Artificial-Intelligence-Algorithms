import math
import random


# The function that is being optimized. Namely the Booth function.
# Reference: https://en.wikipedia.org/wiki/Test_functions_for_optimization
def calculate_booth(x, y):
    return math.pow(x + 2 * y - 7, 2) + math.pow(2 * x + y - 5, 2)


# Particle Swarm Optimization (PSO)
# Representing the concept of a particle:
# - Position: The position of the particle in all dimensions.
# - Best position: The best position found using the fitness function.
# - Velocity: The current velocity of the particle’s movement.
class Particle:

    # Initialize a particle; including its position, inertia, cognitive constant, and social constant
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

    # Get the fitness of the particle
    def get_fitness(self):
        return self.fitness

    # Update the particle's fitness based on the function we're optimizing for
    def update_fitness(self):
        self.fitness = calculate_booth(self.x, self.y)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_x = self.x
            self.best_y = self.y

    # Calculate the inertia component for the particle
    # inertia * current velocity
    @staticmethod
    def calculate_inertia(inertia, current_velocity):
        return inertia * current_velocity

    # Calculate the cognitive component for the particle
    # cognitive acceleration * (particle best solution - current position)
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

    # Calculate the social component for the particle
    # social acceleration * (swarm best position - current position)
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

    # Calculate acceleration for the particle
    @staticmethod
    def calculate_acceleration(constant, random_factor):
        return constant * random_factor

    # Calculate the new velocity for the particle
    @staticmethod
    def calculate_updated_velocity(inertia, cognitive, social):
        return inertia + cognitive + social

    # Calculate the new position for the particle
    @staticmethod
    def calculate_position(current_position_x, current_position_y, updated_velocity):
        return current_position_x + updated_velocity, current_position_y + updated_velocity

    # Perform the update on inertia component, cognitive component, social component, velocity, and position
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


# Ant Colony Optimization (ACO).
# The general lifecycle of a particle swarm optimization algorithm is as follows:
# - Initialize the population of particles: This involves determining the number of particles to be used and initialize
# each particle to a random position in the search space.

# - Calculate the fitness of each particle: Given the position of each particle, determine the fitness of that particle
# at that position.

# - Update the position of each particle: This involves repetitively updating the position of all the particles using
# principles of swarm intelligence. Particles will explore then converge to good solutions.

# - Determine stopping criteria: This involves determining when the particles stop updating and the algorithm stops.
class Swarm:

    # Initialize a swarm of particles randomly
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

    # Return a static swarm of particles
    @staticmethod
    def get_sample_swarm():
        p1 = Particle(7, 1, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p2 = Particle(-1, 9, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p3 = Particle(5, -1, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        p4 = Particle(-2, -5, INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT)
        particles = [p1, p2, p3, p4]
        return particles

    # Return a randomized swarm of particles
    @staticmethod
    def get_random_swarm(number_of_particles):
        particles = []
        for p in range(number_of_particles):
            particles.append(Particle(random.randint(-10, 10),
                                      random.randint(-10, 10),
                                      INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT))
        return particles

    # Get the best particle in the swarm based on its fitness
    def get_best_in_swarm(self):
        best = math.inf
        best_particle = None
        for p in self.swarm:
            p.update_fitness()
            if p.fitness < best:
                best = p.fitness
                best_particle = p
        return best_particle

    # Run the PSO lifecycle for every particle in the swarm
    def run_pso(self):
        for t in range(0, self.number_of_iterations):
            best_particle = self.get_best_in_swarm()
            for p in self.swarm:
                p.update(best_particle.x, best_particle.y)
            print('Best particle fitness: ', best_particle.fitness)


# Set the hyper parameters for the PSO
INERTIA = 0.4
COGNITIVE_CONSTANT = 0.3
SOCIAL_CONSTANT = 0.7
RANDOM_CHANCE = True
NUMBER_OF_PARTICLES = 200
NUMBER_OF_ITERATIONS = 500

# Initialize and execute the PSO algorithm
swarm = Swarm(INERTIA, COGNITIVE_CONSTANT, SOCIAL_CONSTANT, RANDOM_CHANCE, NUMBER_OF_PARTICLES, NUMBER_OF_ITERATIONS)
swarm.run_pso()
