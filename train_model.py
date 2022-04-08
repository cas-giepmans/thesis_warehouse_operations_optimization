import sys
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
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

        self.endTime = 0
        self.lr = 2.e-4
        self.lr_decay = 0.9
        self.episode_reward = 0.0
        self.epsiode_count = 0
        self.change_count = 200

    def RunTraining(self, train_episodes):
        self.startTime = datetime.now()
        all_episode_times = []
        all_episode_rewards = []
        dims = self.wh_sim.dims
        for i_episode in range(train_episodes):
            # wh_state = self.wh_sim.ResetState(random_fill_percentage=0.5)
            wh_state = self.wh_sim.ResetState(random_fill_percentage=0.0)

            # Reset local variables.
            self.episode_reward = 0.0
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
                    # FIXME: this never occurs, see if you can disconnect it.
                    print(f"The episode terminated prematurely because of order {next_order_id}.")
                    self.episode_reward = -1000
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
                    # Process the action, returns the time at which the action is done.
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
                    # Process the action, returns the time at which the action is done.
                    action_time, is_end = self.wh_sim.ProcessAction(infeed, action)

                # Finish and log the executed order.
                finish_time = action_time  # + self.wh_sim.sim_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Calculate and log the reward.
                # TODO: find a more logical reward mechanism.
                # reward = self.wh_sim.prev_action_time - action_time
                # Try this: reward is relative action time (i.e. response time)
                reward = -action_time + self.wh_sim.sim_time
                self.neural_network.add_reward(reward)
                self.episode_reward += reward
                all_reward.append(reward)

                # Log the action time.
                self.wh_sim.prev_action_time = action_time

                if is_end:
                    break

            # Adjust the learning rate.
            if i_episode % self.change_count == 0:
                self.lr = self.lr * self.lr_decay

            # Perform a training step for the neural network.
            self.neural_network.train_step(self.lr)

            # Append the episode's time to the list of all times.
            all_episode_times.append(self.wh_sim.sim_time)
            all_episode_rewards.append(self.episode_reward)

            # Print the order register for inspection.
            # To use: slice access_densities with indices [0, :] and [1, :].
            access_densities = self.wh_sim.GetShelfAccessDensities(normalized=False, print_it=False)

            # Print episode training meta info.
            print(f"Finished ep. {i_episode + 1}/{train_episodes}.")
            print(f"Cumulative order fulfillment time: {-round(self.episode_reward, 2)}\n")
            print(f"""Episode time: {self.wh_sim.sim_time}
                      \rMin. episode time so far: {min(all_episode_times)}
                      \rEpisode reward sum: {self.episode_reward}
                      \rMax. episode reward: {max(all_reward)}
                      \rMin. episode reward: {min(all_reward)}
                      \rLearning rate: {self.lr}
                      \rInfeed orders: {infeed_count}
                      \rOutfeed orders: {outfeed_count}
                      \r""")
            # print(f"all rewards: {[round(reward, 1) for reward in all_reward]}")

        # Training is done here, display the time taken.
        self.endTime = datetime.now()  # dt.strftime('%y-%m-%d %I:%M:%S %p')
        t_seconds = (self.endTime - self.startTime).total_seconds()
        print(
            f"Time taken: {t_seconds} seconds. Episodes/second: {round(train_episodes / t_seconds, 2)}")

        # TODO: Write a function for exporting the order_register to a csv file. Would be good to
        # see shelf access density.

        # Print the average episode time. Shorter is better.
        # print("Average episode time:", np.mean(self.wh_sim.all_episode_times), "Variance episode times:", np.var(self.wh_sim.all_episode_times), "Standard deviation episode times:", np.std(self.wh_sim.all_episode_times) )
        # self.DrawResults(all_episode_rewards)
        # self.DrawAccessDensity(in_dens)
        # self.DrawAccessDensity(out_dens)

        # Print a matrix that shows the order in which shelves were filled. Infeed only!
        # self.DrawAccessOrder()
        self.SaveExperimentResults(train_episodes, all_episode_rewards)

    def RunBenchmark(self, n_iterations, benchmark_policy="random"):
        self.startTime = datetime.now()
        all_episode_times = []
        all_episode_rewards = []
        for iter_i in range(n_iterations):
            # wh_state = self.wh_sim.ResetState(random_fill_percentage=0.5)
            self.wh_sim.ResetState(random_fill_percentage=0.0)

            # Reset local variables.
            self.episode_reward = 0.0
            all_action = []
            all_reward = []
            infeed_count = 0
            outfeed_count = 0

            while True:
                # Increase the sim_time by some amount here.
                # In order to recreate the Lei Luo paper results, new orders should start when the
                # vertical transporter becomes available.
                self.wh_sim.sim_time = self.wh_sim.agent_busy_till['vt']

                # Store the number of free and occupied locations.
                free_and_occ = (np.count_nonzero(~self.wh_sim.shelf_occupied),
                                np.count_nonzero(self.wh_sim.shelf_occupied))

                # Generate a new order.
                self.wh_sim.order_system.GenerateNewOrder(
                    order_type="infeed", item_type=1, current_time=self.wh_sim.sim_time)

                # Pick an order from one of the queues. Also, check if order is possible given
                # warehouse occupancy (if order is infeed and warehouse is full, you can't infeed.)
                next_order_id, next_order = self.wh_sim.order_system.GetNextOrder(
                    self.wh_sim.sim_time, free_and_occ, False)
                # Assume we're benchmarking the infeed-only scenario.
                infeed = True

                # Calculate a new RTM.
                self.wh_sim.CalcRTM()

                # Get an action given the defined benchmark policy.
                action = self.wh_sim.GetNextBenchmarkPolicyShelfId(bench_pol=benchmark_policy)
                infeed_count += 1

                all_action.append(action)
                # Have the selected action get executed by the warehouse sim.
                try:
                    # Process the action, returns the time at which the action is done.
                    action_time, is_end = self.wh_sim.ProcessAction(infeed, action)
                except Exception:
                    # TODO: Make sure the system doesn't generate illegal orders.
                    print("Picking random action from available shelves.")
                    print(f"free IDs: {self.wh_sim.GetIds(True)}")
                    # print(f"Free locs: {free_locs}")
                    print(next_order)
                    self.wh_sim.PrintOccupancy()
                    av_ids = self.wh_sim.GetIds(infeed)
                    action = random.choice(av_ids)
                    # Process the action, returns the time at which the action is done.
                    action_time, is_end = self.wh_sim.ProcessAction(infeed, action)

                # Finish and log the executed order.
                finish_time = action_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Calculate and log the reward.
                reward = -action_time + self.wh_sim.sim_time
                self.episode_reward += reward
                all_reward.append(reward)

                # Log the action time.
                self.wh_sim.prev_action_time = action_time

                if is_end:
                    break

            # Append the iteration's time to the list of all times.
            all_episode_times.append(self.wh_sim.sim_time)
            all_episode_rewards.append(self.episode_reward)

            # Print iteration meta info.
            print(f"Finished it. {iter_i + 1}/{n_iterations}.")
            print(f"Cumulative order fulfillment time: {-round(self.episode_reward, 2)}\n")
            # print(f"""Episode time: {self.wh_sim.sim_time}
            #           \rMin. episode time so far: {min(all_episode_times)}
            #           \rEpisode reward sum: {self.episode_reward}
            #           \rMax. episode reward: {max(all_reward)}
            #           \rMin. episode reward: {min(all_reward)}
            #           \rInfeed orders: {infeed_count}
            #           \rOutfeed orders: {outfeed_count}
            #           \r""")

        # For the stochastic benchmarks, also output an average time.
        if benchmark_policy == ("random" or "eps_greedy"):
            avg_cumul_time = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)
            print(f"Average cumulative order fulfillment time: {avg_cumul_time}\n")

        self.endTime = datetime.now()
        t_seconds = (self.endTime - self.startTime).total_seconds()
        print(
            f"Time taken: {t_seconds} seconds. Episodes/second: {round(n_iterations / t_seconds, 2)}\n\n")

        self.SaveExperimentResults(n_iterations, all_episode_rewards,
                                   exp_name=benchmark_policy)

    def SaveBestModel(self):
        # Determine whether to save as a new model based on the principle of least total time consumption
        if self.all_episode_times[len(self.all_episode_times)-1] == min(self.all_episode_times):
            savePath = "./plantPolicy_" + str(self.wh_sim.XDim) + "_" + str(self.wh_sim.YDim) + "_" + str(
                self.train_episodes) + ".model"
            self.neural_network.save_model(savePath)  # 保存模型

    # TODO: Improve this function: subplots, titles, correctly oriented.
    def DrawAccessDensity(self, access_densities):
        """Draw the access densities for shelves given infeed and outfeed order counts."""
        dims = self.wh_sim.dims
        in_dens = np.reshape(access_densities[0], (dims[0] * dims[1], dims[2]))
        out_dens = np.reshape(access_densities[1], (dims[0] * dims[1], dims[2]))

        plt.figure()
        # plt.imshow(density_matrix, cmap="CMRmap")
        plt.colorbar()
        plt.show()

    def DrawAccessOrder(self, all_episode_rewards, exp_name=None):
        """Draw a warehouse-like matrix where each entry denotes when that shelf was filled."""
        dims = self.wh_sim.dims
        orders = self.wh_sim.order_system.order_register
        order_matrix = np.zeros((dims[0], dims[1], dims[2]), dtype=int)

        for (order_id, order_data) in orders.items():
            (r, f, c) = self.wh_sim.shelf_rfc[order_data['shelf_id']]
            order_matrix[r, f, c] = order_id

        order_matrix = np.reshape(order_matrix, (dims[0] * dims[1], dims[2]))

        halfway_point = int(dims[0] * dims[1] / 2)
        left_order_matrix = order_matrix[0:halfway_point, :]
        right_order_matrix = order_matrix[halfway_point:, :]

        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location='right',
                         cbar_mode='single',
                         cbar_size='7%',
                         cbar_pad=0.15)

        # Fill both subplots.
        matrices = [left_order_matrix, right_order_matrix]
        for idx, ax in enumerate(grid):
            im = ax.matshow(matrices[idx])
            ax.set_title(f"{'Left' if idx == 0 else 'Right'} side")
            ax.invert_yaxis()
            # Set the order number in its respective shelf.
            for j in range(halfway_point):
                for i in range(dims[2]):
                    val = matrices[idx][j, i]
                    ax.text(i, j, str(val), va='center', ha='center')

        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)

        avg_cumul_time = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)

        # Save before show. When show is called, the fig is removed from mem.
        if exp_name is not None:
            plt.suptitle("Shelf access order" + f" for {exp_name}. t = {avg_cumul_time}s")
        else:
            plt.suptitle("Shelf access order AC network")
        plt.savefig("storage order.jpg")
        plt.show()

    def DrawResults(self, all_episode_rewards, exp_name=None):
        # Drawing
        print("DrawResults")

        plt.figure()
        if exp_name is not None:
            plt.title("Storage performance for " + exp_name)
        else:
            plt.title("Storage performance AC network")
        plt.xlabel("Nr. of training episodes")
        plt.ylabel("Cumulative order fulfillment time")
        all_episode_rewards = [-reward for reward in all_episode_rewards]
        plt.plot(all_episode_rewards, color="g")
        plt.savefig("performance trajectory.jpg")

        # mean_list = [np.mean(all_episode_rewards)] * len(all_episode_rewards)
        if exp_name is None:
            plt.figure()
            plt.title("Training loss (limited at 50 and -50) AC network")
            loss_list = self.neural_network.getLossValue()
            for i in range(len(loss_list)):
                if loss_list[i] >= 50:
                    loss_list[i] = 50
                elif loss_list[i] <= -50:
                    loss_list[i] = -50

            plt.plot(loss_list)

            zero_list = [0] * len(all_episode_rewards)
            plt.plot(zero_list, linestyle='--', color="k")
            plt.savefig("training loss trajectory.jpg")
        else:
            pass
        plt.show()

    # TODO: Finish method for saving experiment results and metadata.
    def SaveExperimentResults(self, nr_of_episodes, all_episode_rewards, exp_name=None):
        """Method for saving the parameters and results of an experiment to a folder."""
        # Navigate to the experiment folder, create the folder for this run.
        # Go up two levels to the 'Thesis' folder.
        original_dir = os.getcwd()
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
        if exp_name is not None:
            ex_dir = "Benchmark " + str(nr_files + 1) + f" - {exp_name}"
        else:
            ex_dir = f"Exp. {nr_files + 1}"  # ", t_fin = {fin_time}"
        try:
            os.mkdir(ex_dir)
        except FileExistsError:
            print(f"Something went wrong, folder '{ex_dir}' already exists...?")

        os.chdir(ex_dir)  # Now we're ready to store experiment parameters and results.

        avg_cumul_time = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)

        # Begin writing experiment summary/metadata.
        run_time = self.endTime - self.startTime
        # run_time = run_time.strftime('%y-%m-%d %I:%M:%S %p')
        self.startTime = self.startTime.strftime('%y-%m-%d %I:%M:%S %p')
        self.endTime = self.endTime.strftime('%y-%m-%d %I:%M:%S %p')
        with open(f"{ex_dir} summary.txt", 'w') as f:
            f.write(f"Description of experiment {nr_files + 1}.\n\n")
            f.write(f"Start: {self.startTime}. Finish: {self.endTime}. Run time: {run_time}\n\n")
            f.write(f"Dimensions: {self.wh_sim.dims}\n")
            f.write(f"Episode length: {self.wh_sim.episode_length}\n")
            f.write(f"Number of historical RTMs: {self.wh_sim.num_historical_rtms}\n")
            f.write(f"Number of episodes: {nr_of_episodes}\n")
            f.write(f"Learning rate: {self.lr}\n")
            f.write(f"Learning rate decay: {self.lr_decay}, every {self.change_count} episodes\n\n")

            f.write(f"Average cumulative order fulfillment time: {avg_cumul_time}")
        f.close()

        # Save the figures for stochastic benchmarks
        if exp_name in ('random', 'eps_greedy'):
            self.DrawResults(all_episode_rewards, exp_name=exp_name)
        elif exp_name is None:
            self.DrawResults(all_episode_rewards)
        else:
            pass
        self.DrawAccessOrder(all_episode_rewards, exp_name=exp_name)

        # Return to the original directory in case we're executing multiple benchmarks in sequence.
        os.chdir(original_dir)


def main():
    # x_dim = 8  # (column-1)
    # y_dim = 6 * 2  # row * 2
    num_rows = 2
    num_floors = 6
    num_cols = 6
    episode_length = 72
    num_hist_rtms = 5
    wh_sim = wh(num_rows,
                num_floors,
                num_cols,
                episode_length,
                num_hist_rtms=num_hist_rtms)

    train_episodes = 1000  # This value exceeding 5000 is not recommended
    train_plant_model = TrainGameModel(wh_sim)
    # Train the network; regular operation.
    # train_plant_model.RunTraining(train_episodes)

    # Benchmarks: run one at a time, still a work in progress. Note: be aware of the fact that plots
    # are saved locally! See SaveExperimentResults()
    train_plant_model.RunBenchmark(50, benchmark_policy='random')
    train_plant_model.RunBenchmark(1, benchmark_policy='greedy')
    train_plant_model.RunBenchmark(50, benchmark_policy='eps_greedy')
    train_plant_model.RunBenchmark(1, benchmark_policy='col_by_col')
    train_plant_model.RunBenchmark(1, benchmark_policy='col_by_col_alt')
    train_plant_model.RunBenchmark(1, benchmark_policy='floor_by_floor')
    train_plant_model.RunBenchmark(1, benchmark_policy='floor_by_floor_alt')
    train_plant_model.RunBenchmark(1, benchmark_policy='rfc_policy')
    train_plant_model.RunBenchmark(1, benchmark_policy='crf_policy')
    # sys.exit("training end")


if __name__ == '__main__':
    main()
