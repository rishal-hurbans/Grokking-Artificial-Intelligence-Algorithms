import random
import numpy as np

# Reinforcement learning
# The simulator should provide the following functionality and information as a minimum.

# - Initialize the environment: This involves resetting the environment including the agent to the starting state.

# - Get the current state of the environment: This function should provide the current state of the environment.
# The state of the environment will change after each action is performed.

# - Apply action to environment: This involves having the agent applying an action to the environment.
# The environment will be impacted by the action which may result in a reward.

# - Calculate the reward of an action: This is related to the "apply action to environment” function.
# The reward for the action and impact in the environment needs to be calculated.

# - Determine if the goal is achieved: This function results in whether or not the agent has achieved the goal.
# This can also sometimes be represented as “is complete”; in the case of an environment where the goal could
# potentially not be achieved, the simulator needs to signal completion when it deems necessary.

# Set constants that represent symbols, rewards, and actions in the environment
ROAD_AGENT = '*'

ROAD_EMPTY = ' '
ROAD_EMPTY_REWARD = 100

ROAD_OBSTACLE_CAR = '#'
ROAD_OBSTACLE_CAR_REWARD = -100

ROAD_OBSTACLE_PERSON = '!'
ROAD_OBSTACLE_PERSON_REWARD = -1000

ROAD_GOAL = '@'
ROAD_GOAL_REWARD = 500

ROAD_OUT_OF_BOUNDS = '-'
ROAD_OUT_OF_BOUNDS_REWARD = -5

COMMAND_NORTH = 0
COMMAND_SOUTH = 1
COMMAND_EAST = 2
COMMAND_WEST = 3


# Get a random action
def get_random_action():
    return random.randint(0, 3)


# Initialize a map with the starting point and goal
DEFAULT_START_X = 0
DEFAULT_START_Y = 0
DEFAULT_GOAL_X = 9
DEFAULT_GOAL_Y = 5
DEFAULT_ROAD_SIZE_X = 10
DEFAULT_ROAD_SIZE_Y = 10
DEFAULT_ROAD = [[' ', '#', '#', '#', ' ', '#', ' ', ' ', '#', ' '],
                [' ', '!', ' ', ' ', ' ', ' ', ' ', '!', ' ', '!'],
                [' ', '#', '#', ' ', '#', '#', ' ', ' ', ' ', '#'],
                [' ', ' ', ' ', ' ', ' ', ' ', '#', '#', ' ', '#'],
                ['#', '!', '!', '#', ' ', '!', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', '#', ' ', ' ', '#', '#', '#', ' '],
                [' ', '#', '#', '#', '!', ' ', '#', '#', '!', ' '],
                [' ', ' ', ' ', '!', '#', ' ', '!', ' ', ' ', ' '],
                ['#', '!', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                [' ', '#', ' ', '#', '#', '@', ' ', '#', ' ', '!']]


# The Simulator class encompasses the functions for the simulation environment
class Simulator:

    # Initialize the simulator with a road, its size, the starting point, and the goal point
    def __init__(self, road, road_size_x, road_size_y, agent_start_x, agent_start_y, goal_x, goal_y):
        self.road_size_x = road_size_x
        self.road_size_y = road_size_y
        self.road = road
        self.rewards = 0
        self.agent_x = agent_start_x
        self.agent_y = agent_start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.states = []

    # Move the agent and return the reward based on a command. The command is an action (North, South, East, West)
    def move_agent(self, command):
        reward_update = 0
        next_x = 0
        next_y = 0
        if command == COMMAND_NORTH:
            next_x = self.agent_x - 1
            next_y = self.agent_y
        elif command == COMMAND_SOUTH:
            next_x = self.agent_x + 1
            next_y = self.agent_y
        elif command == COMMAND_EAST:
            next_x = self.agent_x
            next_y = self.agent_y + 1
        elif command == COMMAND_WEST:
            next_x = self.agent_x
            next_y = self.agent_y - 1
        if self.is_within_bounds(next_x, next_y):
            reward_update = self.calculate_movement_reward(next_x, next_y)
            self.agent_x = next_x
            self.agent_y = next_y
            # print('Origin ', self.player_x, ',', self.player_y)
            # print('Target ', next_x, ',', next_y)
        else:
            reward_update = ROAD_OUT_OF_BOUNDS_REWARD
        self.rewards += reward_update
        #print('Immediate Reward: ', reward_update)
        #print('Lifetime Reward: ', self.rewards)
        return reward_update

    # Move the agent to a target point
    def move(self, target_x, target_y):
        self.agent_x = target_x
        self.agent_y = target_y

    # Calculate the reward for the movement based on an action
    def calculate_movement_reward(self, next_x, next_y):
        if self.road[next_x][next_y] == ROAD_OBSTACLE_PERSON:
            return ROAD_OBSTACLE_PERSON_REWARD
        elif self.road[next_x][next_y] == ROAD_OBSTACLE_CAR:
            return ROAD_OBSTACLE_CAR_REWARD
        elif self.road[next_x][next_y] == ROAD_GOAL:
            return ROAD_GOAL_REWARD
        else:
            return ROAD_EMPTY_REWARD

    # Determine if the target point is within bounds
    def is_within_bounds(self, target_x, target_y):
        if self.road_size_x > target_x >= 0 and self.road_size_y > target_y >= 0:
            return True
        return False

    # Determine if the goal is achieved
    def is_goal_achieved(self):
        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            return True
        return False

    # Get the state. This is a string encoding based on the immediate neighbors around the current point
    def get_state(self):
        state = ''
        for x in range(-1, 2):
            for y in range(-1, 2):
                if self.is_within_bounds(x, y):
                    state += self.road[x][y]
                else:
                    state += ROAD_OUT_OF_BOUNDS
        if state not in self.states:
            self.states.append(state)
        return self.states.index(state)

    def print_road(self):
        output = ''
        for x in range(self.road_size_x):
            for y in range(self.road_size_y):
                if x == self.agent_x and y == self.agent_y:
                    output += ROAD_AGENT
                else:
                    output += self.road[x][y]
            output += '\n'
        print('Agent x: ', self.agent_x)
        print('Agent y: ', self.agent_y)
        print(output)


# Execute a known most rewarding path
def execute_happy_path():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.print_road()


# Execute a known path with penalties
def execute_sad_path():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    simulator.move_agent(COMMAND_NORTH)
    simulator.move_agent(COMMAND_WEST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.print_road()


# Execute a random brute force approach until the goal is found
def execute_random_brute_force():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    while not simulator.is_goal_achieved():
        simulator.move_agent(get_random_action())
        simulator.print_road()
    simulator.print_road()


# Train the Q-table using Q-learning
# - Initialize simulator:
# This involves resetting the environment to the starting state with the agent in a neutral state.

# - Get environment state:
# This function should provide the current state of the environment. The state of the environment will change after
# each action is performed.

# - Is goal achieved:
# Determine if the goal is achieved (or the simulator deems the exploration to be complete). In our example, this
# would be picking up the owner of the self-driving car. If the goal is achieved, the algorithm will end.

# - Pick a random action:
# Determine if a random action should be selected; if so, a random action will be selected out of north, south, east,
# and west. Random actions are useful to explore the possibilities in the environment instead of learn only a narrow
# subset.

# - Reference action in Q-table: If the decision to select a random action is not selected, the current environment
# state is transposed to the Q-table and the respective action is selected based on the values in table. More about
# the Q-table is coming up.

# - Apply action to environment: This involves applying the selected action to the environment; whether that is a
# random action or an action selected from the Q-table. An action will have a consequence in the environment and
# yield a reward.

# - Update Q-table: The following describes the concepts involved in updating the Q-table and the steps that are
# carried out.
# Learning rate is represented as alpha in the book.
# Discount is represented as gamma in the book.
def train_with_q_learning(observation_space, action_space, number_of_iterations, learning_rate, discount,
                          chance_of_random_move):
    # Initialize the Q-table
    q_table = np.zeros([observation_space, action_space], dtype=np.int8)

    # Repeat for a number of iterations
    for i in range(number_of_iterations):
        # Reset the simulator
        simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                              DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
        state = simulator.get_state()
        done = False

        # Continue while the simulator is not terminated
        while not done:
            action = COMMAND_SOUTH
            # Choose a random action or choose the best move from the Q-table given the current state
            if random.uniform(0, 1) > chance_of_random_move:
                action = get_random_action()
            else:
                action = np.argmax(q_table[state])

            # Apply the selected action to the simulation and get the changed state and reward
            reward = simulator.move_agent(action)
            next_state = simulator.get_state()
            done = simulator.is_goal_achieved()

            print(simulator.get_state())
            print('State: ', state)
            print('Action: ', action)

            # Calculate the Q-table value for the selected action
            current_value = q_table[state, action]
            next_state_max_value = np.max(q_table[next_state])
            new_value = (1 - learning_rate) * current_value + learning_rate * (reward + discount * next_state_max_value)
            q_table[state, action] = new_value
            state = next_state
            print(q_table)

    return q_table


# Use the trained Q-table to navigate the map to the goal, to test the training
def execute_with_q_learning(q_table, number_of_episodes):
    total_epochs = 0
    total_penalties_person = 0
    total_penalties_car = 0

    # Repeat for a number of episodes
    for episode in range(number_of_episodes):
        # Initialize a simulator
        simulator = Simulator(DEFAULT_ROAD,
                              DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y,
                              DEFAULT_START_X, DEFAULT_START_Y,
                              DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
        state = simulator.get_state()
        epochs = 0
        penalties_person = 0
        penalties_car = 0
        reward = 0
        done = False

        # Choose an action from the Q-table while the simulator is not terminated
        while not done:
            action = np.argmax(q_table[state])

            reward = simulator.move_agent(action)
            state = simulator.get_state()
            done = simulator.is_goal_achieved()

            if reward == ROAD_OBSTACLE_PERSON_REWARD:
                penalties_person += 1
            elif reward == ROAD_OBSTACLE_CAR_REWARD:
                penalties_car += 1

            epochs += 1

        # Calculate the penalties accrued
        total_penalties_person += penalties_person
        total_penalties_car += penalties_car
        total_epochs += epochs

    print('Results after ', number_of_episodes, 'episodes:')
    print('Average time steps per episode: ', total_epochs / number_of_episodes)
    print('Average person penalties per episode: ', total_penalties_person / number_of_episodes)
    print('Average car penalties per episode: ', total_penalties_person / number_of_episodes)


if __name__ == '__main__':
    #execute_happy_path()
    #execute_sad_path()
    #execute_random_brute_force()
    LEARNING_RATE = 0.1
    DISCOUNT = 0.6
    CHANCE_OF_RANDOM_ACTION = 0.1
    trained_q_table = train_with_q_learning(4*4*4*4*4*4*4*4, 4, 100, LEARNING_RATE, DISCOUNT, CHANCE_OF_RANDOM_ACTION)
    #execute_q_learning(trained_q_table, 1)
