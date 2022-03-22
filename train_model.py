"""
train PlantSimulation model
@author: stone
"""
import sys
import random
import matplotlib.pyplot as plt
from policy_net_AC_pytorch import PolicyValueNet as trainNet
from newPlantGame import PlantGame_AVSRS as plantGame
from Warehouse import Warehouse as wh  # Sim replacement
import numpy as np
from datetime import datetime


class TrainGameModel():
    def __init__(self, warehouse_sim):
        # 创建神经网络
        self.warehouse_sim = warehouse_sim
        self.neural_network = trainNet(
            warehouse_sim.num_cols,
            warehouse_sim.num_rows * warehouse_sim.num_floors,
            3,  # number of states (?)
            warehouse_sim.num_locs,  # Number of storage locations.
            warehouse_sim.ActionNum)
        dt = datetime.now()  # 创建一个datetime类对象
        self.startTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        self.endTime = 0
        self.lr = 4.e-3
        self.lr_decay = 0.9
        self.epsiode_count = 0
        self.change_count = 1500

    def run_training(self, train_episodes):
        all_episode_reward = []
        for i_episode in range(train_episodes):
            my_state = self.warehouse_sim.get_init_state()
            available_pos = self.warehouse_sim.get_available_pos()
            episode_reward = 0
            all_action = []
            all_reward = []

            while True:
                # TODO: Increase the sim_time by some amount here.
                o_type = "infeed" if bool(random.getrandbits(1)) else "outfeed"
                self.warehouse_sim.order_system.GenerateNewOrder(
                    order_type=o_type,
                    item_type=1,
                    current_time=self.warehouse_sim.sim_time)

                self.warehouse_sim.CalcRTM()
                action = self.neural_network.select_action(np.array(my_state), available_pos)
                all_action.append(action)
                my_state, this_reward, is_end = self.warehouse_sim.do_action(action)
                available_pos = self.warehouse_sim.get_available_pos()
                self.neural_network.add_reward(this_reward)
                episode_reward += this_reward

                if is_end:
                    break

            # Adjust the learning rate.
            self.epsiode_count = self.epsiode_count + 1 # self.epsiode_count += 1
            if self.epsiode_count == self.change_count:
                self.lr = self.lr*self.lr_decay
                self.change_count = 500
                self.epsiode_count = 0

            # Perform a training step for the neural network.
            self.neural_network.train_step(self.lr)

            # Reset the warehouse instance for the next training episode.
            # self.warehouse_sim.dolastAction()
            all_episode_reward.append(episode_reward)

            # Print episode training meta info.
            # print(f"Finished episode {i_episode}/4000")
            # print("i_episode:", i_episode,
            #       "episode_reward", episode_reward,
            #       "max_reward:", max(all_episode_reward),
            #       "all_action:", all_action,
            #       "thisTime", self.warehouse_sim.episodeTimes[
            #           len(self.warehouse_sim.episodeTimes)-1],
            #       "minTime:", min(self.warehouse_sim.episodeTimes),
            #       "maxTime:", max(self.warehouse_sim.episodeTimes),
            #       "lr", self.lr)

        dt = datetime.now()  # 创建一个datetime类对象
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("Start time", self.startTime, "End time", self.endTime)
        # Print the average episode time. Shorter is better.
        # print("Average episode time:", np.mean(self.warehouse_sim.episodeTimes), "Variance episode times:", np.var(self.warehouse_sim.episodeTimes), "Standard deviation episode times:", np.std(self.warehouse_sim.episodeTimes) )
        # self.drawResult(self.warehouse_sim.episodeTimes)

    def saveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if self.warehouse_sim.episodeTimes[len(self.warehouse_sim.episodeTimes)-1] == min(self.warehouse_sim.episodeTimes):
            savePath = "./plantPolicy_" + str(self.warehouse_sim.XDim) + "_" + str(self.warehouse_sim.YDim) + "_" + str(
            self.train_episodes) + ".model"
            self.neural_network.save_model(savePath)  # 保存模型

    def drawResult(self, all_episode_reward):
        # 画图
        print("drawResult")

        plt.figure()
        plt.title("AC algorithm on SBS/RS")
        plt.xlabel("training rounds")
        plt.ylabel("spent time")
        plt.plot(all_episode_reward, color="g")

        mean_list = [np.mean(all_episode_reward)] * len(all_episode_reward)
        plt.figure()
        plt.title("Training loss (limited at 50 and -50)")
        loss_list = self.neural_network.getLossValue()
        for i in range(len(loss_list)):
            if loss_list[i] >= 50:
                loss_list[i] = 50
            elif loss_list[i] <= -50:
                loss_list[i] = -50

        plt.plot(loss_list)

        zero_list = [0] * len(all_episode_reward)
        plt.plot(zero_list, linestyle='--', color="k")
        plt.show()


def main():
    # x_dim = 8  # (column-1)
    # y_dim = 6 * 2  # row * 2
    num_rows = 2
    num_floors = 6
    num_cols = 8
    warehouse_sim = wh(num_rows, num_floors, num_cols)

    train_episodes = 1  # 不建议该值超过5000
    train_plant_model = TrainGameModel(warehouse_sim)
    train_plant_model.run_training(train_episodes)
    sys.exit("training end")


if __name__ == '__main__':
    main()
