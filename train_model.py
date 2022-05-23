import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from policy_net_AC_pytorch import PolicyValueNet as trainNet
import torch
# from torchsummary import summary
from Warehouse import Warehouse as wh  # Sim replacement
import numpy as np
from datetime import date, datetime
import copy


class TrainGameModel():
    def __init__(self, wh, lr=1.e-4, lr_decay=0.9, lr_decay_interval=200, discount_factor=1.0, reward_type="makespan_full_spectrum"):
        """
        Initialize the object that governs the training of the network and the running of the
        simulation.

        Parameters
        ----------
        wh : Warehouse
            The warehouse object.
        lr : float, optional
            The initial learning rate. The default is 1.e-4.
        lr_decay : float, optional
            Scalar that specifies the decrease of the learning rate at each interval.
        lr_decay_interval : int, optional
            Specifies after how many episodes the learning rate is decreased.
        discount_factor : float, optional
            Gamma, used for calculating the discounted reward over time.
        reward_type : str, optional
            Which reward mechanism to use. The default is "makespan_full_spectrum".

        Returns
        -------
        None.

        """
        self.wh_sim = wh
        self.neural_network = trainNet(
            wh.num_cols,
            wh.num_rows * wh.num_floors,
            wh.num_historical_rtms + 1,  # number of states (?)
            wh.num_locs,  # Number of storage locations.
            wh.num_locs)

        # summary(self.neural_network.policy_value_net, input_size=(6, 12, 6))
        self.endTime = 0
        self.lr_init = lr
        self.lr = lr
        self.lr_decay = lr_decay
        self.discount_factor = discount_factor
        self.episode_reward = 0.0
        self.epsiode_count = 0
        self.change_count = lr_decay_interval

        self.scenarios = ["infeed", "outfeed", "both"]
        self.all_reward_types = ["rel_action_time",
                                 "makespan_half_spectrum",
                                 "makespan_full_spectrum"]
        self.reward_type = reward_type
        self.benchmark_policies = ["random",
                                   "greedy",
                                   "eps_greedy",
                                   "rcf_policy",
                                   "cfr_policy",
                                   "frc_policy",
                                   "fcr_policy",
                                   "rfc_policy",
                                   "crf_policy"]
        self.baselines = {}

    def RunTraining(self, train_episodes, scenario="infeed", baselines=["random", "greedy"]):
        """
        Runs a number of training episodes using the specified warehouse and training settings. Can
        also run benchmarks to include in the generated plots. Saves experiment (meta)data in a text
        file.

        Parameters
        ----------
        train_episodes : int
            The number of episodes for which to train the network.
        scenario : str, optional
            Whether the warehouse is processing infeed, outfeed or both types of orders.
        baselines : list(str), optional
            Which benchmark policies to run in order to calculate baselines (for comparison).

        Raises
        ------
        ValueError
            If the input parameters are defined wrongly.
        RuntimeError
            When an incompatible training setting is detected.
        Exception
            If the order-to-process is not of a valid type.

        Returns
        -------
        None.

        """

        # First perform some checks.
        if scenario not in self.scenarios:
            raise ValueError(
                f"There is no scenario called '{scenario}'! Specify 'infeed', 'outfeed' or 'both'.")

        if self.reward_type not in self.all_reward_types:
            raise ValueError(f"Reward type {self.reward_type} is not a valid reward type!")

        if baselines is not None:
            for benchmark in baselines:
                if benchmark not in self.benchmark_policies:
                    raise ValueError(f"There is no benchmark policy named {benchmark}.")

            if scenario != "infeed" and baselines != ["random"]:
                raise RuntimeError(
                    "Can only run infeed-only simulations when using benchmarks other than random!")

        self.startTime = datetime.now()
        all_episode_times = []
        all_episode_rewards = []
        last_order_registers = []
        last_access_densities = []
        dims = self.wh_sim.dims

        # Run benchmarks to establish baselines that are included in the training performance graph.
        print(f"Running the following benchmarks: {baselines}\n\n")
        for bench_pol in baselines:
            print(f"Running {bench_pol} benchmark...")
            self.baselines[bench_pol] = self.RunBenchmark(100, scenario, bench_pol, False)
            print(f"Finished. Average makespan: {round(self.baselines[bench_pol], 2)} seconds.")

        # Specify the exploration strategy here.
        init_eps = 1.0
        fin_eps = 0.04
        epsilon = 1.0
        eps_trajectory = [0.1, 0.5, 1.0]

        for i_episode in range(train_episodes):
            # wh_state = self.wh_sim.ResetState(random_fill_percentage=0.5)
            wh_state = self.wh_sim.ResetState(random_fill_percentage=self.wh_sim.init_fill_perc)

            # Set epsilon here.
            point_in_training = i_episode / train_episodes
            if point_in_training < eps_trajectory[0]:
                epsilon = init_eps
            elif point_in_training < eps_trajectory[1]:
                epsilon = max((1.0 - 2 * point_in_training), fin_eps)
            else:
                epsilon = fin_eps

            # Reset local variables.
            self.episode_reward = 0.0
            all_action = []
            all_reward = []
            infeed_count = 0
            outfeed_count = 0
            occupied_locs = None
            free_locs = None
            prev_max_busy_till = 0.0
            max_busy_till = 0.0

            while True:
                # Increase the sim_time by some amount here.
                # TODO: Create a mechanism for stochasticall (?) increasing simulation time.
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
                if scenario == "both":  # If we're doing both infeed and outfeed.
                    self.wh_sim.order_system.GenerateNewOrder(current_time=self.wh_sim.sim_time,
                                                              order_type="random",
                                                              free_and_occ=free_and_occ,
                                                              item_type=1)
                else:  # If we're doing only infeed or only outfeed.
                    self.wh_sim.order_system.GenerateNewOrder(current_time=self.wh_sim.sim_time,
                                                              order_type=scenario,
                                                              free_and_occ=free_and_occ,
                                                              item_type=1)

                # Pick an order from one of the queues. Also, check if order is possible given
                # warehouse occupancy (if order is infeed and warehouse is full, you can't infeed.)
                next_order_id, next_order = self.wh_sim.order_system.GetNextOrder(
                    self.wh_sim.sim_time, free_and_occ, init_fill_perc=self.wh_sim.init_fill_perc)
                self.wh_sim.order_system.order_register[next_order_id]['time_start'] = float(
                    self.wh_sim.sim_time)

                infeed = True if next_order["order_type"] == "infeed" else False

                # Calculate a new RTM.
                self.wh_sim.CalcRTM()

                # Prepare the state.
                wh_state = self.wh_sim.BuildState(infeed)
                # print(infeed_count)

                # Select an action with the NN based on the state, order type and occupancy.
                # TODO: Make sure the selected action is a usable shelf_id!
                if infeed:
                    action = self.neural_network.select_action(
                        np.array(wh_state), free_locs, epsilon)
                    infeed_count += 1
                elif not infeed:
                    action = self.neural_network.select_action(
                        np.array(wh_state), occupied_locs, epsilon)
                    outfeed_count += 1
                else:
                    raise Exception(f"""The order type of order {next_order_id}
                                    ({next_order["order_type"]}) is wrong! Time:
                                    {self.wh_sim.sim_time}.""")

                # Have the selected action get executed by the warehouse sim.
                action_time, is_end = self.wh_sim.ProcessAction(infeed, action)
                all_action.append(action)

                # Update maximum busy time.
                prev_max_busy_till = max_busy_till
                max_busy_till = max(self.wh_sim.agent_busy_till.values())

                # Calculate the reward according to the specified reward type.
                if self.reward_type == "rel_action_time":
                    # Reward is negative proportional to response time for action.
                    candidate_reward = -action_time + self.wh_sim.sim_time
                elif self.reward_type == "makespan_half_spectrum":
                    # Reward is negative if makespan increased, else zero.
                    candidate_reward = prev_max_busy_till - max_busy_till
                elif self.reward_type == "makespan_full_spectrum":
                    # Reward is negative if makespan increased, else positive.
                    candidate_reward = prev_max_busy_till - \
                        max_busy_till + (max_busy_till - action_time)

                # Finish and log the executed order.
                finish_time = action_time  # + self.wh_sim.sim_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Add and log the reward.
                reward = candidate_reward
                self.neural_network.add_reward(reward)
                self.episode_reward += reward
                all_reward.append(reward)

                # Log the action time.
                self.wh_sim.prev_action_time = action_time

                if is_end:
                    break

            # Set the sim time to when the last agent finishes its scheduled task.
            if self.wh_sim.sim_time < max_busy_till:
                self.wh_sim.sim_time = max_busy_till

            # Adjust the learning rate.
            if i_episode != 0 and i_episode % self.change_count == 0:
                self.lr = self.lr * self.lr_decay

            # Perform a training step for the neural network.
            self.neural_network.train_step(self.lr, self.discount_factor)

            # Append the episode's time to the list of all times.
            all_episode_times.append(self.wh_sim.sim_time)
            all_episode_rewards.append(self.episode_reward)

            # To use: slice access_densities with indices [0] and [1].
            access_densities = self.wh_sim.GetShelfAccessDensities(normalized=False, print_it=False)

            # If the last 20 episodes are taking place, save order regs and access densities.
            if i_episode >= train_episodes - 100:
                last_order_registers.append(copy.deepcopy(self.wh_sim.order_system.order_register))
                last_access_densities.append(access_densities)

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
        print(f"Time taken: {t_seconds}s. Episodes/second: {round(train_episodes / t_seconds, 2)}")

        # Save all the generated plots. Based on infeed/outfeed order generation, check before run!
        self.SaveExperimentResults(train_episodes,
                                   all_episode_times,
                                   last_order_registers=last_order_registers,
                                   last_access_densities=last_access_densities,
                                   save_plots=True)

    def RunBenchmark(self,
                     n_iterations,
                     scenario="infeed",
                     benchmark_policy="random",
                     save_results=True):
        """
        Run one of the benchmarks. Can generate plots and save them, or return an average makespan.

        Parameters
        ----------
        n_iterations : int
            How many iterations of a benchmark to run to establish an average.
        scenario : str, optional
            Whether the warehouse is processing infeed, outfeed or both types of orders.
        benchmark_policy : str, optional
            Which benchmark policy to run. The default is "random".
        save_results : bool, optional
            Whether to generate and save plots, or return an average. The default is True.

        Raises
        ------
        ValueError
            If any of the input parameters are wrongly defined.

        Returns
        -------
        float
            The average makespan of the run benchmark policy.

        """

        # Perform some checks.
        if scenario not in self.scenarios:
            raise ValueError(
                f"There is no scenario called '{scenario}'! Please specify 'infeed', 'outfeed' or 'both'.")
        if benchmark_policy not in self.benchmark_policies:
            raise ValueError(
                f"There is no benchmark called '{benchmark_policy}'! Check the arguments of your function call.")
        if self.reward_type not in self.all_reward_types:
            raise ValueError(f"Reward type {self.reward_type} is not a valid reward type!")

        self.startTime = datetime.now()
        all_episode_times = []
        all_episode_rewards = []
        all_order_registers = []
        all_access_densities = []
        for iter_i in range(n_iterations):
            self.wh_sim.ResetState(random_fill_percentage=self.wh_sim.init_fill_perc)

            # Reset local variables.
            self.episode_reward = 0.0
            all_action = []
            all_reward = []
            infeed_count = 0
            outfeed_count = 0
            prev_max_busy_till = 0.0
            max_busy_till = 0.0

            while True:
                # Increase the sim_time by some amount here.
                # In order to recreate the Lei Luo paper results, new orders should start when the
                # vertical transporter becomes available.
                self.wh_sim.sim_time = self.wh_sim.agent_busy_till['vt']

                # Store the number of free and occupied locations.
                free_and_occ = (np.count_nonzero(~self.wh_sim.shelf_occupied),
                                np.count_nonzero(self.wh_sim.shelf_occupied))

                # Generate a new order.
                if scenario == "both":  # If we're doing both infeed and outfeed.
                    self.wh_sim.order_system.GenerateNewOrder(current_time=self.wh_sim.sim_time,
                                                              order_type="random",
                                                              free_and_occ=free_and_occ,
                                                              item_type=1)
                else:  # If we're doing only infeed or only outfeed.
                    self.wh_sim.order_system.GenerateNewOrder(current_time=self.wh_sim.sim_time,
                                                              order_type=scenario,
                                                              free_and_occ=free_and_occ,
                                                              item_type=1)

                # Pick an order from one of the queues. Also, check if order is possible given
                # warehouse occupancy (if order is infeed and warehouse is full, you can't infeed.)
                next_order_id, next_order = self.wh_sim.order_system.GetNextOrder(
                    self.wh_sim.sim_time, free_and_occ, self.wh_sim.init_fill_perc)
                self.wh_sim.order_system.order_register[next_order_id]['time_start'] = float(
                    self.wh_sim.sim_time)

                # See if we're executing an infeed or outfeed order.
                infeed = True if next_order['order_type'] == "infeed" else False

                # Calculate a new RTM.
                self.wh_sim.CalcRTM()
                # self.wh_sim.PrintRTM()

                # Get an action given the defined benchmark policy.
                action = self.wh_sim.GetNextBenchmarkPolicyShelfId(
                    bench_pol=benchmark_policy, infeed=infeed)
                infeed_count += 1
                # print(f"picked an action: {action}.")
                prev_max_busy_till = max_busy_till

                all_action.append(action)
                # Have the selected action get executed by the warehouse sim.
                # Process the action, returns the time at which the action is done.
                action_time, is_end = self.wh_sim.ProcessAction(infeed, action)

                # Update maximum busy time.
                prev_max_busy_till = max_busy_till
                max_busy_till = max(self.wh_sim.agent_busy_till.values())

                # Calculate the reward according to the specified reward type.
                if self.reward_type == "rel_action_time":
                    # Reward is negative proportional to response time for action.
                    candidate_reward = -action_time + self.wh_sim.sim_time
                elif self.reward_type == "makespan_half_spectrum":
                    # Reward is negative if makespan increased, else zero.
                    candidate_reward = prev_max_busy_till - max_busy_till
                elif self.reward_type == "makespan_full_spectrum":
                    # Reward is negative if makespan increased, else positive.
                    candidate_reward = prev_max_busy_till - \
                        max_busy_till + (max_busy_till - action_time)

                # Finish and log the executed order.
                finish_time = action_time
                self.wh_sim.order_system.FinishOrder(next_order_id, action, finish_time)

                # Calculate and log the reward.
                # reward = -action_time + self.wh_sim.sim_time
                reward = candidate_reward
                self.episode_reward += reward
                all_reward.append(reward)

                # Log the action time.
                self.wh_sim.prev_action_time = action_time

                if is_end:
                    break

            # Set the sim time to when the last agent finishes its scheduled task.
            if self.wh_sim.sim_time < max_busy_till:
                self.wh_sim.sim_time = max_busy_till

            # To use: slice access_densities with indices [0] and [1].
            access_densities = self.wh_sim.GetShelfAccessDensities(normalized=False, print_it=False)

            # Append the iteration's time to the list of all times.
            all_episode_times.append(self.wh_sim.sim_time)
            all_episode_rewards.append(self.episode_reward)
            all_order_registers.append(copy.deepcopy(self.wh_sim.order_system.order_register))
            all_access_densities.append(access_densities)

            # Print iteration meta info.
            # print(f"Finished it. {iter_i + 1}/{n_iterations}.")
            # print(f"Cumulative order fulfillment time: {-round(self.episode_reward, 2)}\n")
            # print(f"Simulation time: {self.wh_sim.sim_time}")
            # print(f"Max busy time: {max_busy_till}")
            # print(f"""Episode time: {self.wh_sim.sim_time}
            #           \rMin. episode time so far: {min(all_episode_times)}
            #           \rEpisode reward sum: {self.episode_reward}
            #           \rMax. episode reward: {max(all_reward)}
            #           \rMin. episode reward: {min(all_reward)}
            #           \rInfeed orders: {infeed_count}
            #           \rOutfeed orders: {outfeed_count}
            #           \r""")

        # For the stochastic benchmarks, also output an average time.
        if benchmark_policy in ["random", "greedy", "eps_greedy"]:
            avg_cumul_time = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)
            # print(f"Average summed reward: {avg_cumul_time}\n")

        self.endTime = datetime.now()
        t_seconds = (self.endTime - self.startTime).total_seconds()
        print(
            f"Time taken: {t_seconds}s. Episodes/second: {round(n_iterations / t_seconds, 2)}\n\n")

        if save_results is True:
            self.SaveExperimentResults(n_iterations,
                                       all_episode_times,
                                       last_order_registers=all_order_registers,
                                       last_access_densities=all_access_densities,
                                       exp_name=benchmark_policy,
                                       save_plots=True)
        else:
            # Return the average makespan
            return sum(all_episode_times)/len(all_episode_times)

    def SaveBestModel(self):
        # Determine whether to save as a new model based on the principle of least total time consumption
        if self.all_episode_times[len(self.all_episode_times)-1] == min(self.all_episode_times):
            savePath = "./plantPolicy_" + str(self.wh_sim.XDim) + "_" + str(self.wh_sim.YDim) + "_" + str(
                self.train_episodes) + ".model"
            self.neural_network.save_model(savePath)  # 保存模型

    def DrawAccessDensity(self, access_densities, exp_name=None, save_fig=True):
        """Draw the access densities for shelves given infeed and outfeed order counts."""
        dims = self.wh_sim.dims
        in_dens = np.reshape(access_densities[0], (dims[0] * dims[1], dims[2]))
        out_dens = np.reshape(access_densities[1], (dims[0] * dims[1], dims[2]))

        halfway_point = int(dims[0] * dims[1] / 2)
        left_in_dens_matrix = in_dens[0:halfway_point, :]
        right_in_dens_matrix = in_dens[halfway_point:, :]
        left_out_dens_matrix = out_dens[0:halfway_point, :]
        right_out_dens_matrix = out_dens[halfway_point:, :]

        fig = plt.figure(figsize=(7, 6))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2),
                         axes_pad=0.30,
                         share_all=True,
                         cbar_location='right',
                         cbar_mode='single',
                         cbar_size='7%',
                         cbar_pad=0.15)

        # Fill the subplots.
        matrices = [left_in_dens_matrix, right_in_dens_matrix,
                    left_out_dens_matrix, right_out_dens_matrix]
        for idx, ax in enumerate(grid):
            im = ax.matshow(matrices[idx], vmin=0, vmax=np.max(access_densities))
            if idx in [0, 1]:
                ax.set_title('Left side' if idx in [0, 2] else 'Right side')
            if idx in [0, 2]:
                ax.set_ylabel('Infeed' if idx == 0 else 'Outfeed')
            ax.invert_yaxis()  # TODO: Check if this is necessary!
            # Set the access count for its respective shelf.
            for j in range(halfway_point):
                for i in range(dims[2]):
                    val = matrices[idx][j, i]
                    ax.text(i, j, str(val), va='center', ha='center', fontsize='small')

        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)

        # Save before show. When show is called, the fig is removed from memory.
        n_ord = np.sum(access_densities)
        if exp_name is not None:
            plt.suptitle(f"Mean shelf access densities for {exp_name}. n_orders = {n_ord}")
        else:
            plt.suptitle(f"Mean shelf access densities for AC network. n_orders = {n_ord}")

        if save_fig:
            plt.savefig("Access densities.jpg")
        plt.show()

    def DrawAccessOrder(self, all_episode_rewards, order_matrix, exp_name=None, save_fig=True):
        """Draw a warehouse-like matrix where each entry denotes when that shelf was filled. This
           plot is only informative when either completely filling an empty warehouse, or emptying a
           full warehouse."""
        dims = self.wh_sim.dims

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
            im = ax.matshow(matrices[idx], vmin=1, vmax=self.wh_sim.num_locs)
            ax.set_title(f"{'Left' if idx == 0 else 'Right'} side")
            ax.invert_yaxis()
            # Set the order number in its respective shelf.
            for j in range(halfway_point):
                for i in range(dims[2]):
                    val = matrices[idx][j, i]
                    ax.text(i, j, str(val), va='center', ha='center')

        # grid.cbar_axes.
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        # cb = grid.cbar_axes[0].colorbar(im)

        avg_cumul_time = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)

        # Save before show. When show is called, the fig is removed from mem.
        if exp_name is not None:
            plt.suptitle("Mean shelf access order" + f" for {exp_name}. t = {avg_cumul_time}s")
        else:
            plt.suptitle("Mean shelf access order AC network")

        if save_fig:
            plt.savefig("storage order.jpg")
        plt.show()

    def DrawResults(self, all_episode_rewards, exp_name=None, save_fig=True):
        """Draws the progression of the makespan over the course of the training/benchmark, as well
           as the progression of the loss values."""
        print("Drawing plots...")

        n = len(self.baselines.keys()) + 1
        color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))

        plt.figure()
        if exp_name is not None:
            plt.title("Storage performance for " + exp_name)
        else:
            plt.title("Storage performance AC network")
            for bench_pol in self.baselines.keys():
                plt.hlines(self.baselines[bench_pol], 0, len(
                    all_episode_rewards), label=bench_pol, color=next(color))
        plt.xlabel("Nr. of training episodes")
        plt.ylabel("Simulation time/makespan")
        all_episode_rewards = [reward for reward in all_episode_rewards]
        # y_min = int(min([min(all_episode_rewards), min(self.baselines.values())]) / 1.1)
        # y_max = int(max([max(all_episode_rewards), max(self.baselines.values())]) * 1.1)
        plt.ylim(1 * self.wh_sim.num_locs, 5 * self.wh_sim.num_locs)
        plt.plot(all_episode_rewards, color=next(color),
                 label="AC network" if exp_name is None else exp_name)
        plt.legend(loc="best")
        if save_fig:
            plt.savefig("performance trajectory.jpg")

        # mean_list = [np.mean(all_episode_rewards)] * len(all_episode_rewards)
        if exp_name is None:
            plt.figure()
            plt.title("Actor losses")
            plt.grid(True)
            loss_list = self.neural_network.getActorLosses()

            plt.plot(loss_list)

            zero_list = [0] * len(all_episode_rewards)
            plt.plot(zero_list, linestyle='--', color="k")
            if save_fig:
                plt.savefig("actor loss trajectory.jpg")
            plt.show()

            plt.figure()
            plt.title("Critic losses")
            plt.grid(True)
            loss_list = self.neural_network.getCriticLosses()

            plt.plot(loss_list)

            zero_list = [0] * len(all_episode_rewards)
            plt.plot(zero_list, linestyle='--', color="k")
            if save_fig:
                plt.savefig("critic loss trajectory.jpg")
        else:
            pass
        plt.show()

    def CalcMeanOrderMatrix(self, last_order_registers):
        """Calculates the mean shelf access order."""
        dims = self.wh_sim.dims
        nr_regs = len(last_order_registers)
        mean_order_matrix = np.zeros((dims[0] * dims[1], dims[2]), dtype=int)

        for order_reg in last_order_registers:
            order_matrix = np.zeros((dims[0], dims[1], dims[2]), dtype=int)

            if len(order_reg.keys()) > self.wh_sim.num_locs:
                raise Exception("Too many processed orders to draw the shelf access order!")

            for (order_id, order_data) in order_reg.items():
                (r, f, c) = self.wh_sim.shelf_rfc[order_data['shelf_id']]
                order_matrix[r, f, c] = order_id

            order_matrix = np.reshape(order_matrix, (dims[0] * dims[1], dims[2]))
            mean_order_matrix = mean_order_matrix + order_matrix

        mean_order_matrix = mean_order_matrix / nr_regs
        return mean_order_matrix.round(1)

    def CalcMeanAccessDensity(self, last_access_densities):
        """Calculates the mean shelf access density."""
        dims = self.wh_sim.dims
        nr_mats = len(last_access_densities)
        mean_access_density_matrix = np.zeros((2, dims[0], dims[1], dims[2]), dtype=np.float64)

        for dens_mat in last_access_densities:
            if mean_access_density_matrix.shape != dens_mat.shape:
                print(mean_access_density_matrix.shape, dens_mat.shape)
                raise Exception("Tried to calculate mean access density matrix, but shapes differ!")

            mean_access_density_matrix += dens_mat

        mean_access_density_matrix = mean_access_density_matrix / nr_mats
        return mean_access_density_matrix.round(2)

    def SaveExperimentResults(self, nr_of_episodes,
                              all_episode_rewards,
                              last_order_registers=None,
                              last_access_densities=None,
                              exp_name=None,
                              save_plots=True):
        """This method should be customized by anyone using it, in order to save experiment
        results in the right folder."""
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

        avg_episode_reward = -round(sum(all_episode_rewards) / len(all_episode_rewards), 2)
        min_episode_reward = min(all_episode_rewards)
        min_episode_number = np.argmin(all_episode_rewards)

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
            f.write(f"V_vt: {self.wh_sim.v_vt}")
            f.write(f"V_sh: {self.wh_sim.v_sh}")
            f.write(f"Number of historical RTMs: {self.wh_sim.num_historical_rtms}\n")
            f.write(f"Number of episodes: {nr_of_episodes}\n")
            f.write(f"Learning rate: {self.lr_init}\n")
            f.write(f"Learning rate decay: {self.lr_decay}, every {self.change_count} episodes\n\n")
            f.write(f"Discount factor: {self.discount_factor}")

            f.write(f"Reward type: {self.reward_type}\n")

            f.write(f"Average episode reward sum: {avg_episode_reward}\n")
            f.write(f"Best episode: {min_episode_number}\n")
            f.write(f"Best episode time: {round(min_episode_reward, 2)} seconds\n\n")

            f.write("Average makespans for benchmark policies:")
            for bench_pol in self.baselines.keys():
                f.write(f"{bench_pol}: {round(self.baselines[bench_pol], 2)} seconds\n")
            # f.write(f"")
            # f.write(f"")
        f.close()

        # Save the figures for stochastic benchmarks
        if exp_name in ('random', 'eps_greedy'):
            self.DrawResults(all_episode_rewards, exp_name=exp_name, save_fig=save_plots)
        elif exp_name is None:
            self.DrawResults(all_episode_rewards, save_fig=save_plots)
        else:
            pass

        if last_order_registers is not None:
            if exp_name in ('random', 'eps_greedy') or exp_name is None:
                mean_order_matrix = self.CalcMeanOrderMatrix(last_order_registers)
                self.DrawAccessOrder(all_episode_rewards, mean_order_matrix,
                                     exp_name=exp_name if exp_name is not None else "AC net",
                                     save_fig=save_plots)
            # Save a separate one as well for demonstration purposes.
            order_matrix = self.CalcMeanOrderMatrix([last_order_registers[-1]])
            self.DrawAccessOrder(all_episode_rewards, order_matrix,
                                 exp_name=str(exp_name if exp_name is not None else "AC net"
                                              + " (last episode)"),
                                 save_fig=save_plots)
        if last_access_densities is not None and self.wh_sim.episode_length > self.wh_sim.num_locs:
            mean_access_density_matrix = self.CalcMeanAccessDensity(last_access_densities)
            self.DrawAccessDensity(mean_access_density_matrix,
                                   exp_name=exp_name, save_fig=save_plots)

        # Return to the original directory in case we're executing multiple benchmarks in sequence.
        os.chdir(original_dir)


def main():
    # Train the network; regular operation.
    baselines = ["random", "greedy", "crf_policy"]
    architectures = [(2, 4, 4), (2, 6, 6), (2, 10, 6)]
    opt_speeds = [12., 30., 90.]
    gammas = [0.8, 0.86, 0.91]
    alphas = [4.e-4, 1.78e-4, 1.07e-4]
    alpha_deltas = [160, 360, 600]
    alpha_decay = 0.8
    num_eps = [1600, 3600, 6000]
    # num_eps = [10, 10, 10]

    experiment_settings = zip(
        architectures,
        opt_speeds,
        gammas,
        alphas,
        alpha_deltas,
        num_eps)

    speed_scalars = [1., 0.5, 0.25]

    for (dims, v_opt, gamma, alpha, alpha_delta, num_eps) in experiment_settings:
        for scale in speed_scalars:
            wh_sim = wh(dims[0],
                        dims[1],
                        dims[2],
                        dims[0] * dims[1] * dims[2],
                        5,
                        0,
                        v_opt * scale,
                        1.,
                        0.0)

            model = TrainGameModel(wh_sim,
                                   alpha,
                                   alpha_decay,
                                   alpha_delta,
                                   gamma,
                                   "makespan_full_spectrum")

            model.RunTraining(num_eps, "infeed", baselines)


if __name__ == '__main__':
    main()
