import numpy as np

import torch
import torch.optim as optim
from torch.distributions import Categorical

from machineLearning_AC.policy_net_AC_pytorch import Net

class PolicyValueNet():#创建神经网络和训练神经网络
    def __init__(self, xDim, yDim, stateNume, posNum, actionNum):
        self.XNum = xDim
        self.YNum = yDim
        self.StateNum = stateNume
        self.posNum = posNum  # 本游戏中，posNum = xdim*yDim
        self.actionNum = actionNum  # 本游戏中，actionNum = posNum
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module

        #创建神经网络
        self.policy_value_net = Net(xDim, yDim, stateNume, posNum, actionNum)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const, lr=5e-4)

        # action & reward buffer
        self.saved_log_probs = []
        self.rewards = []
        self.lossValue = []

    def select_action(self, state, availablePos):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy_value_net(state) #statevalue是ac使用的，再此处无用

        # 剔除不可用的位置
        for temp_i in range(len(availablePos)):
            if availablePos[temp_i] == 0:
                probs.data[0][temp_i] = 0

        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def addReward(self, thisReward):
        self.rewards.append(thisReward)

    def train_step(self, lr):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.95 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        # print("policy_loss:",policy_loss)
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def get_policy_param(self):
        #获取并保存模型
        net_params = self.policy_value_net.state_dict()
        return net_params


    def save_model(self, model_file):
        #获取并保存模型
        # net_params = self.get_policy_param()  # get model params
        torch.save(self.policy_value_net, model_file)

    def set_learning_rate(self, optimizer, lr):
        """Sets the learning rate to the given value"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def getLossValue(self):
        return self.lossValue

