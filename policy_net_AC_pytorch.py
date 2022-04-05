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
        # self.conv2 = nn.Conv2d(50, self.StateNum, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, self.StateNum, kernel_size=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.act_fc1 = nn.Linear(self.StateNum * self.posNum, self.actionNum)

        # state value layers
        self.val_conv1 = nn.Conv2d(64, self.StateNum, kernel_size=1)
        self.val_fc1 = nn.Linear(self.StateNum * self.posNum, 64)
        self.val_fc2 = nn.Linear(64, 1)

    # TODO: rewrite method outside of class, then use @jit(nopython=True)
    # @jit(forceobj=True)
    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

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


class PolicyValueNet:  # Create neural networks and train neural networks
    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum  # In this game, posNum = xdim*yDim
        self.actionNum = actionNum  # In this game, actionNum = posNum
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module

        # Create a neural network
        # self.policy_value_net = Net(xDim, yDim, stateNume, posNum, actionNum)
        self.policy_value_net = DeeperValAct_net(xDim, yDim, stateNume, posNum, actionNum)
        # self.policy_value_net = DeeperNet(xDim, yDim, stateNume, posNum, actionNum)
        """Initialize the parameters of the neural network"""
        # for m in self.policy_value_net.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.uniform_(m.weight, 0, 1)
        # #     elif isinstance(m, nn.BatchNorm2d):
        # #         nn.init.constant_(m.weight, 1)
        # #         nn.init.constant_(m.bias, 0)
        # print(self.policy_value_net)
        # # print(self.policy_value_net.weight)
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
                                    lr=3e-3, weight_decay=self.l2_const)
        # self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

        # if model_file:
        #    net_params = torch.load(model_file)
        #    self.policy_value_net.load_state_dict(net_params)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.lossValue = []

    # TODO: consider rewriting for @jit(nopython=True) compilation.
    # @jit(forceobj=True)
    def select_action(self, state, availablePos):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print("state value in Array(input to the neural network):", state)
        probs, state_value = self.policy_value_net(state)
        # probs_aaaaa = copy.deepcopy(probs.data[0])
        # print("probs_before:", probs)
        max_prob = 0
        max_prob_index = 0
        for temp_i in range(len(availablePos)):
            # if probs.data[0][temp_i] <= 1e-8:  # Prevent the appearance of nan, indeterminate
            #     probs.data[0][temp_i] = 1e-8
            if availablePos[temp_i] == 0:
                probs.data[0][temp_i] = 0

            if probs.data[0][temp_i] >= max_prob:
                max_prob = probs.data[0][temp_i]
                max_prob_index = temp_i

        if max_prob <= 1e-8:  # Prevent the appearance of nan, indeterminate
            probs.data[0][max_prob_index] = 1e-8

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # m = Categorical(probs.clip_by_value(probs, 1e-8, 1.0))

        # and sample an action using the distribution

        try:
            action = m.sample()
            temp_counter = 0
            while True:
                if temp_counter >= 10:
                    # print("temp_i:", temp_i)
                    # print("action:", action.item())
                    # print("state:", state)
                    # print("availablePos:", availablePos)
                    # print("probs:", probs)
                    # print("probs_aaaaa:", probs_aaaaa)
                    # print("state_value:", state_value)
                    # for label, p in enumerate(probs[0]):
                    #     print(f'{label:2}: {100*probs[0][label]}%')
                    # sys.exit()
                    return
                if availablePos[action] == 0:
                    action = m.sample()
                    # print(temp_counter)
                    temp_counter = temp_counter+1
                else:
                    break
        except:
            print("state:", state)
            print("availablePos:", availablePos)
            print("probs:", probs)
            print("state_value:", state_value)
        # print(action)

        # save to action buffer
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def add_reward(self, this_reward):
        self.rewards.append(this_reward)

    def train_step(self, lr):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment

        parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor (default: 0.99)')
        parser.add_argument('--seed', type=int, default=543, metavar='N',
                            help='random seed (default: 543)')
        parser.add_argument('--render', action='store_true',
                            help='render the environment')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='interval between training status logs (default: 10)')
        args = parser.parse_args()

        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        # Subtract from the rewards their mean, then divide by their stdev + epsilon.
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            # print("R:", R)
            # print("value:", value)

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            warnings.simplefilter(action='ignore', category=UserWarning)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()
        self.set_learning_rate(self.optimizer, lr)

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        self.lossValue.append(loss.detach().numpy().tolist())
        # perform backprop
        loss.backward()
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
