"""
Created on Fri Mar 18 14:24:20 2022

Copyright (C) <year>  <name of author>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
from torch.autograd import Variable
import numpy as np
import copy
import warnings
import sys


from numba import jit


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        super(Net, self).__init__()
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum         # 本游戏中，posNum = xdim*yDim
        self.actionNum = actionNum   # 本游戏中，actionNum = posNum
        '''----------------------------------------'''

        # common layers
        self.conv1 = nn.Conv2d(self.StateNum, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(self.StateNum, 36, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(36, 72, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(100, 50, kernel_size=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, self.StateNum, kernel_size=1)

        # action policy layers
        # self.act_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.act_fc1 = nn.Linear(self.StateNum * self.posNum, self.actionNum)

        # state value layers
        # self.val_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.val_fc1 = nn.Linear(self.StateNum * self.posNum, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        # action policy layers
        # x_act = F.relu(self.act_conv1(x))
        x_act = x.view(-1, self.StateNum * self.posNum)
        x_act = F.softmax(self.act_fc1(x_act), dim=-1)

        # state value layers
        # x_val = F.relu(self.val_conv1(x))
        x_val = x.view(-1, self.StateNum * self.posNum)
        # x_val = self.val_fc1(x_val)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val


class DeeperNet(nn.Module):
    """"一个更深的神经网络"""
    """policy-value network module"""

    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        super(DeeperNet, self).__init__()
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum         # 本游戏中，posNum = xdim*yDim
        self.actionNum = actionNum   # 本游戏中，actionNum = posNum
        '''----------------------------------------'''

        # common layers
        self.conv1 = nn.Conv2d(self.StateNum, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(50, self.StateNum, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, self.StateNum, kernel_size=1)

        # action policy layers
        # self.act_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.act_fc1 = nn.Linear(self.StateNum * self.posNum, self.actionNum)

        # state value layers
        # self.val_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.val_fc1 = nn.Linear(self.StateNum * self.posNum, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # action policy layers
        # x_act = F.relu(self.act_conv1(x))
        x_act = x.view(-1, self.StateNum * self.posNum)
        x_act = F.softmax(self.act_fc1(x_act), dim=-1)

        # state value layers
        # x_val = F.relu(self.val_conv1(x))
        x_val = x.view(-1, self.StateNum * self.posNum)
        # x_val = self.val_fc1(x_val)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val


class DeeperValAct_net(nn.Module):
    """"A deeper neural network"""
    """policy-value network module"""

    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        super(DeeperValAct_net, self).__init__()
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum  # In this game, posNum = xdim*yDim
        self.actionNum = actionNum  # In this game, actionNum = posNum
        '''----------------------------------------'''
        # common layers
        self.conv1 = nn.Conv2d(self.StateNum, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.act_fc1 = nn.Linear(self.StateNum * self.posNum, self.actionNum)

        # state value layers
        self.val_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.val_fc1 = nn.Linear(self.StateNum * self.posNum, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input, available_pos):
        # Construct BoolTensor for illegal pos elimination.
        av_pos = torch.BoolTensor([available_pos])

        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, self.StateNum * self.posNum)
        x_act = self.act_fc1(x_act)
        x_act = F.softmax(x_act.where(av_pos, torch.tensor(-float('inf'))), dim=1)
        # TODO: remove custom softmax and av_pos so you can create a model.summary()

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.StateNum * self.posNum)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val


class DeepestNet(nn.Module):
    """"一个更深的神经网络"""
    """policy-value network module"""

    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        super(DeeperNet, self).__init__()
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum         # 本游戏中，posNum = xdim*yDim
        self.actionNum = actionNum   # 本游戏中，actionNum = posNum
        '''----------------------------------------'''

        # common layers
        self.conv1 = nn.Conv2d(self.StateNum, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(50, self.StateNum, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(64, self.StateNum, kernel_size=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.act_fc1 = nn.Linear(self.StateNum * self.posNum, self.actionNum)

        # state value layers
        self.val_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.val_fc1 = nn.Linear(self.StateNum * self.posNum, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, self.StateNum * self.posNum)
        x_act = F.softmax(self.act_fc1(x_act), dim=-1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.StateNum * self.posNum)
        # x_val = self.val_fc1(x_val)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val


class PolicyValueNet:
    """Create neural networks and train neural networks."""

    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum  # In this game, posNum = xdim*yDim
        self.actionNum = actionNum  # In this game, actionNum = posNum
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module

        # Create a neural network
        self.policy_value_net = DeeperValAct_net(xDim, yDim, stateNume, posNum, actionNum)

        # Initialize the parameters of the neural network
        for m in self.policy_value_net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight.data, 0, 1)
                nn.init.xavier_uniform_(m.weight.data, gain=1)
                # print("a")
                # pass
            elif isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    lr=3e-3,
                                    betas=(0.9, 0.999),
                                    weight_decay=self.l2_const)

        # if model_file:
        #    net_params = torch.load(model_file)
        #    self.policy_value_net.load_state_dict(net_params)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.lossValue = []
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state, availablePos, epsilon=None):
        """Forward propagate the input through the network and select an action according to a
           categorical distribution created from the network's output."""
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy_value_net(state, availablePos)

        # Create the categorical distribution from the network's probabilities.
        m = Categorical(probs)
        action = m.sample()

        # Save the action to a list.
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def add_reward(self, this_reward):
        self.rewards.append(this_reward)

    def train_step(self, lr, discount_factor):
        """Perform a training step after an episode has ended. Calculates the discounted reward,
           loss values and then performs backpropagation."""
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment

        # Iterate backwards over the rewards.
        for idx, r in enumerate(self.rewards[::-1]):
            # calculate the discounted value
            R = r + discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        # Subtract from the rewards their mean, then divide by their stdev + epsilon.
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Below, log_prob is the probability of an action in the categorical distribution,
        # value is the value of the state as given by the Critic network,
        # R is the normalized, discounted reward as observed in the simulation.
        for (log_prob, value), R in zip(saved_actions, returns):
            # Here they use the "Advantage" function
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # FIX: warning keeps appearing.
            warnings.simplefilter(action='ignore', category=UserWarning)
            # calculate critic (value) loss using L1 smooth loss
            # value_losses.append(F.mse_loss(value, torch.tensor([R])))  # I added this.
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # Normalize, reduce size.
        policy_losses = torch.stack(policy_losses)
        value_losses = torch.stack(value_losses)

        # reset gradients
        self.optimizer.zero_grad()
        self.set_learning_rate(self.optimizer, lr)

        # sum up all the values of policy_losses and value_losses
        loss = policy_losses.sum() + value_losses.sum()

        # Store the loss values.
        self.lossValue.append(loss.detach().numpy().tolist())
        self.actor_losses.append(policy_losses.sum().detach().numpy().tolist())
        self.critic_losses.append(value_losses.sum().detach().numpy().tolist())

        # Perform backpropagation.
        loss.backward()

        # Perform gradient clipping to potentially avoid exploding gradient problem.
        # torch.nn.utils.clip_grad_value_(self.policy_value_net.parameters(), clip_value=0.5)

        # Optimize.
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def get_policy_param(self):
        # Gets and saves the model
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        # Gets and saves the model
        # net_params = self.get_policy_param()  # get model params
        torch.save(self.policy_value_net, model_file)

    def set_learning_rate(self, optimizer, lr):
        """Sets the learning rate to the given value"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def getLossValue(self):
        return self.lossValue

    def getActorLosses(self):
        return self.actor_losses

    def getCriticLosses(self):
        return self.critic_losses
