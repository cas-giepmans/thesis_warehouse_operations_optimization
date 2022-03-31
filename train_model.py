import sys
import random
import matplotlib.pyplot as plt
from policy_net_AC_pytorch import PolicyValueNet as trainNet
from Warehouse import Warehouse as wh  # Sim replacement
import numpy as np
from datetime import datetime


class TrainGameModel():
    def __init__(self, wh):
        self.wh_sim = wh
        self.neural_network = trainNet(
            wh.num_cols,
            wh.num_rows * wh.num_floors,
            wh.num_historical_rtms + 1,  # number of states (?)
            wh.num_locs,  # Number of storage locations.
            wh.num_locs)
        dt = datetime.now()  # 创建一个datetime类对象
        self.startTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        self.endTime = 0
        self.lr = 4.e-3
        self.lr_decay = 0.9
        self.epsiode_count = 0
        self.change_count = 1500

    def run_training(self, train_episodes):
        all_episode_reward = []
        dims = self.wh_sim.dims
        for i_episode in range(train_episodes):
            # wh_state = self.wh_sim.ResetState(random_fill_percentage=0.5)
            wh_state = self.wh_sim.ResetState(random_fill_percentage=0.0)

            # Reset local variables.
            episode_reward = 0
            all_action = []
            all_reward = []
            infeed_count = 0
            outfeed_count = 0
            occupied_locs = None
            free_locs = None

            while True:
                # Increase the sim_time by some amount here.
                # TODO: set it to be larger than num_floors * shortest response time.
                self.wh_sim.sim_time += 10.0

                # Get the occupancy (and inverse occupancy).
                # occupied_locs = np.reshape(self.wh_sim.shelf_occupied,
                #                            (12, 8)).flatten(order='C').tolist()
                # free_locs = np.reshape(~self.wh_sim.shelf_occupied, (12, 8)
                #                        ).flatten(order='C').tolist()
                # occupied_locs = None
                # free_locs = None
                # The original occupancy matrix needs to be transposed, reshaped, transposed again
                # and then flattened and cast to a list.
                occupied_locs = self.wh_sim.shelf_occupied.transpose((1, 0, 2))
                occupied_locs = occupied_locs.reshape((dims[0] * dims[1], dims[2]))
                occupied_locs = occupied_locs.transpose(1, 0).flatten().tolist()

                free_locs = ~self.wh_sim.shelf_occupied.transpose((1, 0, 2))
                free_locs = free_locs.reshape((dims[0] * dims[1], dims[2]))
                free_locs = free_locs.transpose(1, 0).flatten().tolist()

                # Store the number of free and occupied locations.
                free_and_occ = (len(free_locs), len(occupied_locs))

                # Generate a new order.
                # self.wh_sim.order_system.GenerateNewOrder(
                #     order_type="random", item_type=1, current_time=self.wh_sim.sim_time)
                self.wh_sim.order_system.GenerateNewOrder(
                    order_type="infeed", item_type=1, current_time=self.wh_sim.sim_time)

                # Pick an order from one of the queues. Also, check if order is possible given
                # warehouse occupancy (if order is infeed and warehouse is full, you can't infeed.)
                try:
                    next_order_id, next_order = self.wh_sim.order_system.GetNextOrder(
                        self.wh_sim.sim_time, free_and_occ, False)
                    if next_order["order_type"] == "infeed":
                        infeed = True
                    elif next_order["order_type"] == "outfeed":
                        infeed = False
                except RuntimeError:
                    # Training should continue, but this is unwanted behavior.
                    print(f"The episode terminated prematurely because of order {next_order_id}.")
                    episode_reward = -1000
                    self.neural_network.add_reward(-1000)
                    break

                # Calculate a new RTM.
                self.wh_sim.CalcRTM()

                # Prepare the state.
                wh_state = self.wh_sim.GetState(infeed)
                # print(wh_state)
                # self.wh_sim.PrintOccupancy()

                # Select an action with the NN based on the state, order type and occupancy.
                # TODO: Make sure the selected action is a usable shelf_id!
                if infeed:
                    action = self.neural_network.select_action(np.array(wh_state), free_locs)
                    infeed_count += 1
                elif not infeed:
                    action = self.neural_network.select_action(np.array(wh_state), occupied_locs)
                    outfeed_count += 1
                else:
                    raise Exception(f"""The order type of order {next_order_id}
                                    ({next_order["order_type"]}) is wrong! Time:
                                    {self.wh_sim.sim_time}.""")

                # Get a random shelf ID (for testing).
                # action = self.wh_sim.GetRandomShelfId(infeed=infeed)

                all_action.append(action)
                # print(action)
                # Have the selected action get executed by the warehouse sim.
                # TODO: fix the Categorical dist. not outputting an action.
                try:
                    action_reward, is_end = self.wh_sim.ProcessAction(infeed, action)
                except Exception:
                    print("Picking random action from available shelves.")
                    print(f"free IDs: {self.wh_sim.GetIds(True)}")
                    print(f"Free locs: {free_locs}")
                    # self.wh_sim.PrintOccupancy()
                    av_ids = self.wh_sim.GetIds(infeed)
                    action = random.choice(av_ids)
                    action_reward, is_end = self.wh_sim.ProcessAction(infeed, action)

                # Finish and log the executed order.
                finish_time = action_reward + self.wh_sim.sim_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Log the rewards.
                self.neural_network.add_reward(action_reward)
                episode_reward += action_reward

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

            # Append the episode's reward to the list of all rewards.
            all_episode_reward.append(episode_reward)

            # Print the order register for inspection.
            in_dens, out_dens = self.wh_sim.GetShelfAccessDensities(
                normalized=False, print_it=False)

            # Print occupancy matrix.
            # self.wh_sim.PrintOccupancy()

            # Print episode training meta info.
            print(f"Finished ep. {i_episode + 1}/{train_episodes}.")
            print(f"""Episode number: {i_episode}
                      \rEpisode time: {episode_reward}
                      \rMin. episode time so far: {min(all_episode_reward)}
                      \rLearning rate: {self.lr}
                      \rInfeed orders: {infeed_count}
                      \rOutfeed orders: {outfeed_count}
                      \r""")

        dt = datetime.now()  # 创建一个datetime类对象
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("Start time", self.startTime, "End time", self.endTime)

        # TODO: Write a function for exporting the order_register to a csv file. Would be good to
        # see shelf access density.

        # Print the average episode time. Shorter is better.
        # print("Average episode time:", np.mean(self.wh_sim.all_episode_reward), "Variance episode times:", np.var(self.wh_sim.all_episode_reward), "Standard deviation episode times:", np.std(self.wh_sim.all_episode_reward) )
        self.drawResult(all_episode_reward)
        self.DrawAccessDensity(in_dens)
        # self.DrawAccessDensity(out_dens)

    def saveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if all_episode_reward[len(all_episode_reward)-1] == min(all_episode_reward):
            savePath = "./plantPolicy_" + str(self.wh_sim.XDim) + "_" + str(self.wh_sim.YDim) + "_" + str(
                self.train_episodes) + ".model"
            self.neural_network.save_model(savePath)  # 保存模型

    def DrawAccessDensity(self, density_matrix):
        """Draw the access densities for shelves given infeed and outfeed order counts."""
        dims = self.wh_sim.dims
        density_matrix = np.reshape(density_matrix, (dims[0] * dims[1], dims[2]))

        plt.figure()
        plt.imshow(density_matrix, cmap="CMRmap")
        plt.colorbar()
        plt.show()
        # plt.imshow(outfeed_density, cmap="CMRmap")

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
    episode_length = 96
    num_hist_rtms = 3
    wh_sim = wh(num_rows,
                num_floors,
                num_cols,
                episode_length,
                num_hist_rtms=num_hist_rtms)

    train_episodes = 1  # 不建议该值超过5000
    train_plant_model = TrainGameModel(wh_sim)
    train_plant_model.run_training(train_episodes)
    sys.exit("training end")


if __name__ == '__main__':
    main()
