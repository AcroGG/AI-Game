import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FN
import numpy as np
import os

# Learning model 
class Linear_Q(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    # Forward propagation using RELU activation function
    def forward(self, x):
        x = FN.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        file_name = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_name)
# Training and optimizing the model 
class Trainer_Q:
    def __init__(self, model, lr, discount_rate):
        self.lr = lr
        self.discount_rate = discount_rate
        self.model =model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_turn(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # if only in 1 dimension forces a 2d shape (tuple)
            state = torch.unsqueeze(state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )
        
        # Predicted Q value with current state
        prediction = self.model(state)
        target = prediction.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                Q_new = reward[index] + self.discount_rate * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action[index]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(target, prediction)
        loss.backward()
        self.optimizer.step()






