"""
train PlantSimulation model
@author: stone
"""
import matplotlib.pyplot as plt
from policy_net_AC_pytorch import PolicyValueNet as trainNet
# from policyNet_PG_pytorch import PolicyValueNet as trainNet
from MathTreeGame import MathTreeGame as mathTreeGame
from PlantGame_AVSRS import PlantGame_AVSRS as plantGame
import numpy as np
from datetime import datetime

class TrainGameModel():
    def __init__(self, MyGameModel, Train_episodes):
        # 创建神经网络
        self.My_Game_Model = MyGameModel
        self.My_Train_NET = trainNet(MyGameModel.XDim, MyGameModel.YDim, MyGameModel.StateNum,
                                    MyGameModel.StateBoard, MyGameModel.ActionNum)
        dt = datetime.now()  # 创建一个datetime类对象
        self.startTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        self.endTime = 0
        self.lr =5e-3
        self.lr_decay = 0.5
        # self.lr_decay = 1.

    def runTraining(self, Train_episodes):
        all_episode_reward = []
        for i_episode in range(Train_episodes):
            # print("i_episode:", i_episode)
            myState = self.My_Game_Model.getInitState()
            availablePos = self.My_Game_Model.getAvailablePos()
            episode_reward = 0
            all_action = []
            all_reward = []

            while True:
                action = self.My_Train_NET.select_action(np.array(myState), availablePos)
                all_action.append(action)
                myState, thisReward, isEnd = self.My_Game_Model.doAction(action)
                availablePos = self.My_Game_Model.getAvailablePos()
                self.My_Train_NET.addReward(thisReward)
                # all_reward.append(thisReward)
                episode_reward += thisReward

                if isEnd:
                    break
            if i_episode == int(0.4*Train_episodes):
                self.lr = self.lr*self.lr_decay
            elif i_episode == int(0.8*Train_episodes):
                self.lr = self.lr * self.lr_decay
            # elif i_episode == int(0.6 * Train_episodes):
            #     self.lr = self.lr * self.lr_decay
            # print(self.lr)
            self.My_Train_NET.train_step(self.lr)
            self.My_Game_Model.dolastAction()
            all_episode_reward.append(episode_reward)
            # self.saveBestModel()
            # print("i_episode:", i_episode, "episode_reward", episode_reward, "max_reward:" ,max(all_episode_reward),
            #       "all_action:", all_action, "thisTime", self.My_Game_Model.episodeTime[len(self.My_Game_Model.episodeTime)-1],
            #       "minTime:", min(self.My_Game_Model.episodeTime), "maxTime:", max(self.My_Game_Model.episodeTime),"lr",self.lr)

            print("i_episode:", i_episode, "episode_reward", episode_reward, "max_reward:", max(all_episode_reward), "all_action:", all_action)

        dt = datetime.now()  # 创建一个datetime类对象
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("开始时间：", self.startTime, "结束时间", self.endTime)
        print("均值：", np.mean(self.My_Game_Model.episodeTime), "方差：", np.var(self.My_Game_Model.episodeTime), "标准差:", np.std(self.My_Game_Model.episodeTime) )
        self.drawResult(self.My_Game_Model.episodeTime)
        # print(self.My_Game_Model.episodeTime)

    def saveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if self.My_Game_Model.episodeTime[len(self.My_Game_Model.episodeTime)-1] == min(self.My_Game_Model.episodeTime):
            savePath = "./plantPolicy_" + str(MyGameModel.XDim) + "_" + str(MyGameModel.YDim) + "_" + str(
            Train_episodes) + ".model"
            self.My_Train_NET.save_model(savePath)  # 保存模型

    def drawResult(self, all_episode_reward):
        # 画图
        print("drawResult")

        plt.figure()
        plt.title("AC_Mygame")
        plt.xlabel("episodes")
        plt.ylabel("score")
        plt.plot(all_episode_reward, color="g")

        # x = np.array(range(len(all_episode_reward)))
        meanList = [np.mean(all_episode_reward)] * len(all_episode_reward)
        plt.plot(meanList, linestyle='--', color="k")
        plt.text(1, np.mean(all_episode_reward)+1, "mean:" + str(np.mean(all_episode_reward)))
        plt.text(1, np.mean(all_episode_reward)+3, "min:" + str(min(all_episode_reward))+"__std:"+str(np.std(self.My_Game_Model.episodeTime)))

        # x = np.array(range(len(all_episode_reward)))
        # smooth_func = np.poly1d(np.polyfit(x, all_episode_reward, 3))
        # plt.plot(x, smooth_func(x), label='Mean', linestyle='--', color="k")

        plt.figure()
        # plt.hist(self.My_Train_NET.getLossValue())
        plt.plot(self.My_Train_NET.getLossValue())
        plt.show()


if __name__ == '__main__':
    xDim = 12
    yDim = 10
    MyGameModel = mathTreeGame(xDim, yDim)
    # MyGameModel = plantGame(xDim, yDim)

    Train_episodes = 3000  # 不建议该值超过5000
    TrainPlantModel = TrainGameModel(MyGameModel, Train_episodes)
    TrainPlantModel.runTraining(Train_episodes)
    print("training end")
