import torch
import random
import numpy as np
from collections import deque
from snake_base_game import SnakeGameAI, Direction, Point
from model import Linear_Q, Trainer_Q
from plotter import plotter

MAX_MEMORY = 100000
BATCH_SIZE = 1000 # can be adjusted
LR = 0.001 # can be adjusted
BLOCK_SIZE = 20
class Game_Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 
        self.discount_rate = 0.9 #Can be adjusted must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Q(11,256,3) # Index 1 can be adjusted
        self.trainer = Trainer_Q(self.model,lr=LR, discount_rate=self.discount_rate)

    # Get the current state of the snake and its surroundings
    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y -BLOCK_SIZE)
        point_down = Point(head.x, head.y +BLOCK_SIZE)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_right and game.collision(point_right)) or
            (dir_left and game.collision(point_left)) or
            (dir_up and game.collision(point_up)) or
            (dir_down and game.collision(point_down)),
            # Danger right
            (dir_up and game.collision(point_right)) or
            (dir_down and game.collision(point_left)) or
            (dir_left and game.collision(point_up)) or
            (dir_right and game.collision(point_down)), 
            # Danger left  
            (dir_down and game.collision(point_right)) or
            (dir_up and game.collision(point_left)) or
            (dir_right and game.collision(point_up)) or
            (dir_left and game.collision(point_down)),

            # Directions
            dir_left, dir_right, dir_up, dir_down,

            # Locate Food
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state,dtype=int)
    # Save and train the snake movements past of short term and long term memory
    def save_moves(self, state, action, reward, next_state, game_over):
        self.memory.append((state,action,reward,next_state,game_over))


    def train_long_term(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_turn(states, actions, rewards, next_states, game_overs)

    def train_short_term(self, state, action, reward, next_state, game_over):
        self.trainer.train_turn(state, action, reward, next_state, game_over)

    # Determing the snakes next move
    def next_move(self, state):
        self.epsilon = 80 - self.num_games # can be adjusted
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # can be adjusted
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

# Activate the learning model and game
def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Game_Agent()
    game = SnakeGameAI()

    while True:

        old_state = agent.get_state(game)

        final_move = agent.next_move(old_state)

        reward, game_over, score = game.play(final_move)
        new_state = agent.get_state(game)


        # Train short term memory
        agent.train_short_term(old_state, final_move, reward, new_state, game_over)
        # Remember
        agent.save_moves(old_state, final_move, reward, new_state, game_over)

        if game_over:
            # Train long term memory
            game.reset()
            agent.num_games+=1
            agent.train_long_term()

            if score > record:
                record = score
                agent.model.save()

                print('Game', agent.num_games, 'Score', score, 'Record:', record)

                scores.append(score)
                total_score += score
                mean_score = total_score/agent.num_games
                mean_scores.append(mean_score)
                plotter(scores, mean_scores)



if __name__ == '__main__':
    train()