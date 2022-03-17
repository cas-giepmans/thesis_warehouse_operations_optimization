"""
train PlantSimulation model
@author: stone
"""
import sys
import matplotlib.pyplot as plt
from policy_net_AC_pytorch import PolicyValueNet as trainNet
# from policyNet_PG_pytorch import PolicyValueNet as trainNet
from MathTreeGame import MathTreeGame as mathTreeGame
# from PlantGame_AVSRS import PlantGame_AVSRS as plantGame
from newPlantGame import PlantGame_AVSRS as plantGame # Sim replacement
import numpy as np
from datetime import datetime


class TrainGameModel():
    def __init__(self, my_game_model):
        # 创建神经网络
        self.My_Game_Model = my_game_model
        self.My_Train_NET = trainNet(my_game_model.X_dim, my_game_model.Y_dim, my_game_model.StateNum,
                                    my_game_model.StateBoard, my_game_model.ActionNum)
        dt = datetime.now()  # 创建一个datetime类对象
        self.startTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        self.endTime = 0
        self.lr = 4.e-3
        self.lr_decay = 0.9
        self.epsiode_count = 0
        self.change_count = 1500
        # self.lr_decay = 1.

    def run_training(self, train_episodes):
        all_episode_reward = []
        for i_episode in range(train_episodes):
            my_state = self.My_Game_Model.get_init_state()
            available_pos = self.My_Game_Model.get_available_pos()
            episode_reward = 0
            all_action = []
            all_reward = []

            while True:
                # print("state value in List:", my_state)
                action = self.My_Train_NET.select_action(np.array(my_state), available_pos) # Here we go from the list state representation to a numpy array representation.
                # if action in all_action:
                #     print("same action")
                #     print(action)
                #     print(all_action)
                #     return
                all_action.append(action)
                my_state, this_reward, is_end = self.My_Game_Model.do_action(action)
                available_pos = self.My_Game_Model.get_available_pos()
                self.My_Train_NET.add_reward(this_reward)
                # all_reward.append(this_reward)
                episode_reward += this_reward

                if is_end:
                    break

            # adjust lr
            # if i_episode == int(0.7*train_episodes):
                # self.lr = self.lr*self.lr_decay
            # elif i_episode == int(0.5*train_episodes):
            #     self.lr = self.lr*self.lr_decay
            # elif i_episode == int(0.75 * train_episodes):
            #     self.lr = self.lr * self.lr_decay
            # elif i_episode == int(0.8 * train_episodes):
            #     self.lr = self.lr * self.lr_decay
            # print(self.lr)
            self.epsiode_count = self.epsiode_count + 1 # self.epsiode_count += 1
            if self.epsiode_count == self.change_count:
                self.lr = self.lr*self.lr_decay
                self.change_count = 500
                self.epsiode_count = 0

            # train nn
            self.My_Train_NET.train_step(self.lr)
            self.My_Game_Model.dolastAction()
            all_episode_reward.append(episode_reward)
            # self.saveBestModel()
            print(f"Finished episode {i_episode}/4000")
            print("i_episode:", i_episode,
                  "episode_reward", episode_reward,
                  "max_reward:", max(all_episode_reward),
                  "all_action:", all_action,
                  "thisTime", self.My_Game_Model.episodeTime[
                      len(self.My_Game_Model.episodeTime)-1],
                  "minTime:", min(self.My_Game_Model.episodeTime),
                  "maxTime:", max(self.My_Game_Model.episodeTime),
                  "lr", self.lr)

            # print("i_episode:", i_episode, "episode_reward", episode_reward, "max_reward:", max(all_episode_reward), "all_action:", all_action)

        dt = datetime.now()  # 创建一个datetime类对象
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("开始时间：", self.startTime, "结束时间", self.endTime)
        print("均值：", np.mean(self.My_Game_Model.episodeTime), "方差：", np.var(self.My_Game_Model.episodeTime), "标准差:", np.std(self.My_Game_Model.episodeTime) )
        self.drawResult(self.My_Game_Model.episodeTime)
        # print(self.My_Game_Model.episodeTime)

    def saveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if self.My_Game_Model.episodeTime[len(self.My_Game_Model.episodeTime)-1] == min(self.My_Game_Model.episodeTime):
            savePath = "./plantPolicy_" + str(self.My_Game_Model.XDim) + "_" + str(self.My_Game_Model.YDim) + "_" + str(
            self.train_episodes) + ".model"
            self.My_Train_NET.save_model(savePath)  # 保存模型

    def drawResult(self, all_episode_reward):
        # 画图
        print("drawResult")

        plt.figure()
        plt.title("AC algorithm on SBS/RS")
        plt.xlabel("training rounds")
        plt.ylabel("spent time")
        plt.plot(all_episode_reward, color="g")

        # x = np.array(range(len(all_episode_reward)))
        mean_list = [np.mean(all_episode_reward)] * len(all_episode_reward)
        # desirable_list = [660]*len(all_episode_reward)
        # desirable_list = [540] * len(all_episode_reward)
        # plt.plot(mean_list, linestyle='--', color="k")
        # plt.plot(desirable_list, linestyle='--', color="k")
        # plt.text(1, np.mean(all_episode_reward)+1, "mean:" + str(np.mean(all_episode_reward)))
        # plt.text(1, np.mean(all_episode_reward)+3, "min:" + str(min(all_episode_reward))+"__std:"+str(np.std(self.My_Game_Model.episodeTime)))

        # x = np.array(range(len(all_episode_reward)))
        # smooth_func = np.poly1d(np.polyfit(x, all_episode_reward, 3))
        # plt.plot(x, smooth_func(x), label='Mean', linestyle='--', color="k")

        # plt.hist(self.My_Train_NET.getLossValue())
        plt.figure()
        plt.title("Training loss (limited at 50 and -50)")
        loss_list = self.My_Train_NET.getLossValue()
        for i in range(len(loss_list)):
            if loss_list[i] >= 50:
                loss_list[i] = 50
            elif loss_list[i] <= -50:
                loss_list[i] = -50
            # loss_list[i] = loss_list[i]+650
            # loss_list[i] = loss_list[i] + 530
        plt.plot(loss_list)

        zero_list = [0] * len(all_episode_reward)
        plt.plot(zero_list, linestyle='--', color="k")
        plt.show()


def main():
    x_dim = 8  # (column-1)
    y_dim = 6 * 2  # row * 2
    # my_game_model = mathTreeGame(xDim, yDim)
    my_game_model = plantGame(x_dim, y_dim)

    train_episodes = 1  # 不建议该值超过5000
    train_plant_model = TrainGameModel(my_game_model)
    train_plant_model.run_training(train_episodes)
    sys.exit("training end")


if __name__ == '__main__':
    main()
