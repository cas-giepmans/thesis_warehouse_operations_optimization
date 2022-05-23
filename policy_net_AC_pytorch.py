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
    def forward(self, state_input, available_pos):
        # Construct BoolTensor for illegal pos elimination.
        av_pos = torch.BoolTensor([available_pos])
        # print(av_pos)

        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, self.StateNum * self.posNum)
        x_act = self.act_fc1(x_act)
        # TODO: check for exploding gradients during backprop
        # x_act = self.masked_softmax(x_act, av_pos, -1)
        # if x_act.isnan().any():
        #     print("state: \n" + state_input)
        #     sys.exit()
        # print(x_act)
        x_act = F.softmax(x_act.where(av_pos, torch.tensor(-float('inf'))))

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        # x_val = x_val * shelf_available
        x_val = x_val.view(-1, self.StateNum * self.posNum)
        # x_val = self.val_fc1(x_val)
        # x_val
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val

    def forward_alt(self, state_input, available_pos):
        av_pos = torch.BoolTensor([available_pos])
        print("state_input:")
        print(state_input)
        x = F.relu(self.conv1(state_input))
        print("conv1 output:")
        print(x)
        x = F.relu(self.conv2(x))
        print("conv2 output:")
        print(x)
        x = F.relu(self.conv3(x))
        print("conv3 output:")
        print(x)

        x_act = F.relu(self.act_conv1(x))
        print("act_conv1 output:")
        print(x_act)
        x_act = x_act.view(-1, self.StateNum * self.posNum)
        print("reshape using view:")
        print(x_act)
        x_act = self.act_fc1(x_act)
        print("linear1 gradients:")
        print(self.act_fc1.weight.grad)
        print(f"linear1 contains nans: {self.act_fc1.weight.grad.isnan().any()}")
        # TODO: check for exploding gradients during backprop
        x_act = self.masked_softmax_alt(x_act, av_pos, -1)
        print("masked softmax output:")
        print(x_act)

    def masked_softmax(self, in_tensor, mask_tensor, dim=1):
        # TODO: just use regular softmax, where you set all the unavailable actions as -inf.
        def log_sum_exp_trick(x):
            c = torch.max(x)
            log_tensor = c + torch.log(torch.sum(torch.exp(x - c)))
            return torch.exp(torch.sub(x, log_tensor))

        exps = log_sum_exp_trick(in_tensor)
        # exps = torch.nan_to_num(exps)
        masked_exps = torch.mul(exps, mask_tensor)
        masked_sum = masked_exps.sum(dim, keepdim=True)
        # masked_sums = torch.nan_to_num(masked_sums)
        # Avoid division by 0. here.
        # masked_sums = torch.max(torch.tensor([masked_sums, 1.e-7]))
        print(f"masked_sum: {masked_sum}")
        # print(f"Gradient: {masked_sums.grad}")
        out_tensor = torch.div(masked_exps, masked_sum)
        if out_tensor.isnan().any() or out_tensor.isinf().any():
            print("found Nan's in probs tensor. Exiting...")
            print(f"mask: {mask_tensor}\n")
            print(f"in_tensor: {in_tensor}\n")
            print(f"exps: {exps}\n")
            print(f"masked_sum: {masked_sum}\n")
            print(f"out_tensor: {out_tensor}")
            # sys.exit()
            # raise Exception("Exiting masked softmax")
        return out_tensor

    def masked_softmax_alt(self, in_tensor, mask_tensor, dim=1):
        exps = torch.exp(in_tensor)
        exps = torch.nan_to_num(exps)
        masked_exps = torch.mul(exps, mask_tensor)
        masked_sums = masked_exps.sum(dim, keepdim=True)
        out_tensor = torch.div(masked_exps, masked_sums)
        # if out_tensor.isnan().any():
        print("found Nan's in probs tensor. Exiting...")
        print(f"mask: {mask_tensor}\n")
        print(f"in_tensor: {in_tensor}\n")
        print(f"exps: {exps}\n")
        print(f"masked_sums: {masked_sums}\n")
        print(f"out_tensor: {out_tensor}")
        # sys.exit()
        # raise Exception("Exiting masked softmax")
        return out_tensor


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
                                    lr=3e-3,
                                    betas=(0.9, 0.999),
                                    weight_decay=self.l2_const)
        # self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

        # if model_file:
        #    net_params = torch.load(model_file)
        #    self.policy_value_net.load_state_dict(net_params)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.lossValue = []
        self.actor_losses = []
        self.critic_losses = []

    def select_action_alt(self, state, availablePos, epsilon=None):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy_value_net.forward_alt(state, availablePos)

        sys.exit()

    # TODO: consider rewriting for @jit(nopython=True) compilation.
    # @jit(forceobj=True)
    def select_action(self, state, availablePos, epsilon=None):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print("state value in Array(input to the neural network):", state)
        probs, state_value = self.policy_value_net(state, availablePos)
        # if probs.isnan().any():
        #     print(state)
        #     sys.exit()
        # probs_aaaaa = copy.deepcopy(probs.data[0])
        # print("probs_before:", probs)

        # max_prob_index = np.argmax(probs.data[0])
        # max_prob = probs.data[0][max_prob_index]

        # for idx, pos_av in enumerate(availablePos):
        # if pos_av is False:
        #     probs.data[0][idx] = 0
        #     if probs.data[0][idx] >= max_prob:
        #         max_prob = probs.data[0][idx]
        #         max_prob_index = idx

        # for temp_i in range(len(availablePos)):
        #     # if probs.data[0][temp_i] <= 1e-8:  # Prevent the appearance of nan, indeterminate
        #     #     probs.data[0][temp_i] = 1e-8
        #     if availablePos[temp_i] == 0:
        #         probs.data[0][temp_i] = 0

        #     if probs.data[0][temp_i] >= max_prob:
        #         max_prob = probs.data[0][temp_i]
        #         max_prob_index = temp_i

        # if max_prob <= 1e-8:  # Prevent the appearance of nan, indeterminate
        #     probs.data[0][max_prob_index] = 1e-8

        # create a categorical distribution over the list of probabilities of actions
        # try:
        #     m = Categorical(probs)
        # except ValueError:
        #     print("Encountered an error.")
        #     print("probs tensor:")
        #     print(probs)
        #     print("state:")
        #     print(state)
        #     print("available positions:")
        #     print(availablePos)
        #     sys.exit()
        # try:
        #     m = Categorical(probs)
        # except ValueError:
        #     print("Encountered a value error, probably due to nans")
        #     print("state: ")
        #     print(state)
        #     print("av_pos: ")
        #     print(availablePos)
        m = Categorical(probs)
        action = m.sample()
        # print(m.probs)
        # print(availablePos)

        # rand_val = np.random.uniform()
        # if rand_val > epsilon:
        #     action = torch.tensor(max_prob_index)
        # else:
        #     action = m.sample()
        # print(action)

        # m = Categorical(probs.clip_by_value(probs, 1e-8, 1.0))

        # and sample an action using the distribution
        # print("probs: ", probs)
        # print("probs_aaaa: ", probs_aaaaa)

        # try:
        #     action = m.sample()
        #     temp_counter = 0
        #     while True:
        #         if temp_counter >= 10:
        #             # print("temp_i:", temp_i)
        #             # print("action:", action.item())
        #             # print("state:", state)
        #             # print("availablePos:", availablePos)
        #             # print("probs:", probs)
        #             # print("probs_aaaaa:", probs_aaaaa)
        #             # print("state_value:", state_value)
        #             # for label, p in enumerate(probs[0]):
        #             #     print(f'{label:2}: {100*probs[0][label]}%')
        #             # sys.exit()
        #             return
        #         if availablePos[action] == 0:
        #             action = m.sample()
        #             # print(temp_counter)
        #             temp_counter = temp_counter+1
        #         else:
        #             break
        # except:
        #     print("state:", state)
        #     print("availablePos:", availablePos)
        #     print("probs:", probs)
        #     print("state_value:", state_value)
        # print(action)

        # save to action buffer
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def add_reward(self, this_reward):
        self.rewards.append(this_reward)

    def train_step(self, lr, discount_factor):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
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

            # returns.insert(0, R/(idx+1))
            # returns.insert(0, R/(72-idx))

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
            # Idea: Use discount factor
            # print("R:", R)
            # print("value:", value)

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # TODO: check if log_probs always > 0

            # FIX: warning keeps appearing.
            warnings.simplefilter(action='ignore', category=UserWarning)
            # calculate critic (value) loss using L1 smooth loss
            # FIX: They don't use torch.nn.MSEloss like they told us over email.
            # value_losses.append(F.mse_loss(value, torch.tensor([R])))  # I added this.
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        # print(policy_losses)
        # Normalize, reduce size.
        policy_losses = torch.stack(policy_losses)
        value_losses = torch.stack(value_losses)

        # div = torch.max(torch.tensor([torch.max(policy_losses), 1]))
        # policy_losses = torch.div(policy_losses, div)
        # value_losses = torch.div(value_losses, torch.max(value_losses))
        # policy_losses = policy_losses / max(policy_losses)
        # value_losses = value_losses / max(value_losses)

        # if policy_losses.isnan().any():
        #     print("Policy losses contain nan")
        #     print(policy_losses)
        #     sys.exit()
        # if value_losses.isnan().any():
        #     print("Value losses contain nan")
        #     print(value_losses)
        #     sys.exit()

        # policy_losses = torch.zeros_like(policy_losses)
        # value_losses = torch.zeros_like(value_losses)

        # reset gradients
        self.optimizer.zero_grad()
        self.set_learning_rate(self.optimizer, lr)

        # sum up all the values of policy_losses and value_losses
        # loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss = policy_losses.sum() + value_losses.sum()

        self.lossValue.append(loss.detach().numpy().tolist())
        self.actor_losses.append(policy_losses.sum().detach().numpy().tolist())
        self.critic_losses.append(value_losses.sum().detach().numpy().tolist())
        # print(f"loss value: {loss}")
        # print(f"policy loss: {torch.stack(policy_losses).sum()}")
        # print(f"value loss: {torch.stack(value_losses).sum()}")

        # perform backprop
        loss.backward()

        # Perform gradient clipping to potentially avoid exploding gradient problem.
        # torch.nn.utils.clip_grad_value_(self.policy_value_net.parameters(), clip_value=0.5)
        # torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), 2.0)

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
