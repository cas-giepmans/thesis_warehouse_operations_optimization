import sys
import os
import random
import matplotlib.pyplot as plt
from policy_net_AC_pytorch import PolicyValueNet as trainNet
from Warehouse import Warehouse as wh  # Sim replacement
import numpy as np
from datetime import date, datetime
from pathlib import Path


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
        self.lr = 5.e-4
        self.lr_decay = 0.9
        self.epsiode_count = 0
        self.change_count = 200

    def RunTraining(self, train_episodes):
        all_episode_times = []
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
                # self.wh_sim.sim_time += 10.0

                # In order to recreate the Lei Luo paper results, new orders should start when the
                # vertical transporter becomes available.
                self.wh_sim.sim_time = self.wh_sim.agent_busy_till['vt']

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
                # Have the selected action get executed by the warehouse sim.
                try:
                    action_time, is_end = self.wh_sim.ProcessAction(infeed, action)
                except Exception:
                    # TODO: Make sure the system doesn't generate illegal orders.
                    print("Picking random action from available shelves.")
                    print(f"free IDs: {self.wh_sim.GetIds(True)}")
                    print(f"Free locs: {free_locs}")
                    print(next_order)
                    self.wh_sim.PrintOccupancy()
                    av_ids = self.wh_sim.GetIds(infeed)
                    action = random.choice(av_ids)
                    action_time, is_end = self.wh_sim.ProcessAction(infeed, action)

                # Finish and log the executed order.
                finish_time = action_time + self.wh_sim.sim_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Calculate and log the reward.
                reward = self.wh_sim.prev_action_time - action_time
                self.neural_network.add_reward(reward)
                # episode_reward += reward

                # Log the action time.
                self.wh_sim.prev_action_time = action_time

                if is_end:
                    break

            # Adjust the learning rate.
            self.epsiode_count = self.epsiode_count + 1  # self.epsiode_count += 1
            if self.epsiode_count % self.change_count == 0:
                self.lr = self.lr*self.lr_decay
                # self.change_count = 500
                self.epsiode_count = 0

            # Perform a training step for the neural network.
            self.neural_network.train_step(self.lr)

            # Append the episode's time to the list of all times.
            all_episode_times.append(self.wh_sim.sim_time)

            # Print the order register for inspection.
            in_dens, out_dens = self.wh_sim.GetShelfAccessDensities(
                normalized=False, print_it=False)

            # Print episode training meta info.
            print(f"Finished ep. {i_episode + 1}/{train_episodes}.")
            print(f"""Episode time: {self.wh_sim.sim_time}
                      \rMin. episode time so far: {min(all_episode_times)}
                      \rLearning rate: {self.lr}
                      \rInfeed orders: {infeed_count}
                      \rOutfeed orders: {outfeed_count}
                      \r""")

        # Training is done here, display the time taken.
        dt = datetime.now()
        self.endTime = dt.strftime('%y-%m-%d %I:%M:%S %p')
        print("Start time", self.startTime, "End time", self.endTime)

        # TODO: Write a function for exporting the order_register to a csv file. Would be good to
        # see shelf access density.

        # Print the average episode time. Shorter is better.
        # print("Average episode time:", np.mean(self.wh_sim.all_episode_times), "Variance episode times:", np.var(self.wh_sim.all_episode_times), "Standard deviation episode times:", np.std(self.wh_sim.all_episode_times) )
        self.DrawResults(all_episode_times)
        self.DrawAccessDensity(in_dens)
        self.DrawAccessDensity(out_dens)

    def SaveBestModel(self):
        # 根据总耗时最小的原则确定是否保存为新模型
        if self.all_episode_times[len(self.all_episode_times)-1] == min(self.all_episode_times):
            savePath = "./plantPolicy_" + str(self.wh_sim.XDim) + "_" + str(self.wh_sim.YDim) + "_" + str(
                self.train_episodes) + ".model"
            self.neural_network.save_model(savePath)  # 保存模型

    # TODO: Improve this function: subplots, titles, correctly oriented.
    def DrawAccessDensity(self, density_matrix):
        """Draw the access densities for shelves given infeed and outfeed order counts."""
        dims = self.wh_sim.dims
        density_matrix = np.reshape(density_matrix, (dims[0] * dims[1], dims[2]))

        plt.figure()
        plt.imshow(density_matrix, cmap="CMRmap")
        plt.colorbar()
        plt.show()

    def DrawResults(self, all_episode_times):
        # 画图
        print("DrawResults")

        plt.figure()
        plt.title("AC algorithm on SBS/RS")
        plt.xlabel("training rounds")
        plt.ylabel("spent time")
        plt.plot(all_episode_times, color="g")

        mean_list = [np.mean(all_episode_times)] * len(all_episode_times)
        plt.figure()
        plt.title("Training loss (limited at 50 and -50)")
        loss_list = self.neural_network.getLossValue()
        for i in range(len(loss_list)):
            if loss_list[i] >= 50:
                loss_list[i] = 50
            elif loss_list[i] <= -50:
                loss_list[i] = -50

        plt.plot(loss_list)

        zero_list = [0] * len(all_episode_times)
        plt.plot(zero_list, linestyle='--', color="k")
        plt.show()

    # TODO: Finish method for saving experiment results and metadata.
    def SaveExperimentResults(self):
        """Method for saving the parameters and results of an experiment to a folder."""
        # Navigate to the experiment folder, create the folder for this run.
        # Go up two levels to the 'Thesis' folder.
        os.chdir('../..')
        # Go down to experiments with the original model.
        os.chdir(r"Experiments\LeiLuo's model")
        # Set the name for the folder housing today's experiments.
        today_dir = f"{date.today().day}-{date.today().month}-{date.today().year}"
        try:
            os.mkdir(today_dir)
        except FileExistsError:
            # Folder already exists, use it.
            print(f"Using the folder '{today_dir}'")

        os.chdir(today_dir)  # Now we're inside today's folder.

        # Figure out how many experiments were stored before.
        nr_files = len(os.listdir())
        # fin_time = f"{datetime.now().hour}.{datetime.now().minute}.{datetime.now().second}"
        # Set the experiment's folder's name.
        ex_dir = f"Exp. {nr_files + 1}"  # ", t_fin = {fin_time}"
        try:
            os.mkdir(ex_dir)
        except FileExistsError:
            print(f"Something went wrong, folder '{ex_dir}' already exists...?")

        os.chdir(ex_dir)  # Now we're ready to store experiment parameters and results.

        # Begin writing experiment summary/metadata.
        with open(f"{ex_dir} summary.txt", 'w') as f:
            f.write(f"Description of experiment {nr_files + 1}.\n\n")
            f.write(
                f"Started: {self.startTime}. Finished: {self.endTime}. Run time: {self.endTime - self.startTime}\n")
            f.write(f"")

        # Save the figures.


def main():
    # x_dim = 8  # (column-1)
    # y_dim = 6 * 2  # row * 2
    num_rows = 2
    num_floors = 6
    num_cols = 6
    episode_length = 36
    num_hist_rtms = 5
    wh_sim = wh(num_rows,
                num_floors,
                num_cols,
                episode_length,
                num_hist_rtms=num_hist_rtms)

    train_episodes = 1000  # 不建议该值超过5000
    train_plant_model = TrainGameModel(wh_sim)
    train_plant_model.RunTraining(train_episodes)
    # sys.exit("training end")


if __name__ == '__main__':
    main()
