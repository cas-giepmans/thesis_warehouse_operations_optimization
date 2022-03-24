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
    def __init__(self, wh_sim):
        # 创建神经网络
        self.wh_sim = wh()
        self.neural_network = trainNet(
            wh_sim.num_cols,
            wh_sim.num_rows * wh_sim.num_floors,
            wh_sim.num_hist_rtms,  # number of states (?)
            wh_sim.num_locs,  # Number of storage locations.
            wh_sim.ActionNum)
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
            wh_state = self.wh_sim.init_state()
            # Fill the RTM history registry
            for i in wh_state.num_hist_rtms:
                self.wh_sim.CalcRTM()

            episode_reward = 0
            all_action = []
            all_reward = []

            while True:
                # TODO: Increase the sim_time by some amount here.

                # Get the occupancy (and inverse occupancy).
                occupied_locs = self.wh_sim.shelf_occupied
                free_locs = ~occupied_locs

                # Generate a new order.
                self.wh_sim.order_system.GenerateNewOrder(order_type="random",
                                                          item_type=1,
                                                          current_time=self.wh_sim.sim_time)

                # Pick an order from one of the queues.
                next_order_id, next_order = self.wh_sim.order_system.GetNextOrder(False)

                # Calculate a new RTM.
                self.wh_sim.CalcRTM()

                # Prepare the state.
                wh_state = self.wh_sim.GetState()

                # Select an action with the NN based on the state, order type and occupancy.
                # TODO: Make sure the selected action is a usable shelf_id!
                if next_order["order_type"] == "infeed":
                    action = self.neural_network.select_action(np.array(wh_state), free_locs)
                elif next_order["order_type"] == "outfeed":
                    action = self.neural_network.select_action(np.array(wh_state), occupied_locs)
                else:
                    raise Exception(f"""The order type of order {next_order_id}
                                    ({next_order["order_type"]}) is wrong! Time:
                                    {self.wh_sim.sim_time}.""")

                all_action.append(action)
                # Have the selected action get executed by the warehouse sim.
                wh_state, this_reward, is_end = self.wh_sim.do_action(action)
                self.neural_network.add_reward(this_reward)
                episode_reward += this_reward

                if is_end:
                    break

            # Adjust the learning rate.
            self.epsiode_count = self.epsiode_count + 1  # self.epsiode_count += 1
            if self.epsiode_count == self.change_count:
                self.lr = self.lr*self.lr_decay
                self.change_count = 500
                self.epsiode_count = 0

            # Perform a training step for the neural network.
            self.neural_network.train_step(self.lr)

            # Reset the warehouse instance for the next training episode.
            # self.wh_sim.dolastAction()
            all_episode_reward.append(episode_reward)

            # Print episode training meta info.
            # print(f"Finished episode {i_episode}/4000")
            # print("i_episode:", i_episode,
            #       "episode_reward", episode_reward,
            #       "max_reward:", max(all_episode_reward),
            #       "all_action:", all_action,
            #       "thisTime", self.wh_sim.episodeTimes[
            #           len(self.wh_sim.episodeTimes)-1],
            #       "minTime:", min(self.wh_sim.episodeTimes),
            #       "maxTime:", max(self.wh_sim.episodeTimes),
            #       "lr", self.lr)

        dt = datetime.now()  # 创建一个datetime类对象
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("Start time", self.startTime, "End time", self.endTime)
        # Print the average episode time. Shorter is better.
        # print("Average episode time:", np.mean(self.wh_sim.episodeTimes), "Variance episode times:", np.var(self.wh_sim.episodeTimes), "Standard deviation episode times:", np.std(self.wh_sim.episodeTimes) )
        # self.drawResult(self.wh_sim.episodeTimes)

    def saveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if self.wh_sim.episodeTimes[len(self.wh_sim.episodeTimes)-1] == min(self.wh_sim.episodeTimes):
            savePath = "./plantPolicy_" + str(self.wh_sim.XDim) + "_" + str(self.wh_sim.YDim) + "_" + str(
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
    episode_length = 1000
    num_hist_rtms = 1
    wh_sim = wh(num_rows,
                num_floors,
                num_cols,
                episode_length,
                num_hist_rtms)

    train_episodes = 1  # 不建议该值超过5000
    train_plant_model = TrainGameModel(wh_sim)
    train_plant_model.run_training(train_episodes)
    sys.exit("training end")


if __name__ == '__main__':
    main()
