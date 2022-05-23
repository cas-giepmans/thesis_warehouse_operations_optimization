# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:00:49 2022

@author: casgi
"""
import numpy as np
import random

"""_____________________________________Warehouse.py methods.____________________________________"""


def CalcRTM(self):
    """
    REDUNDANT. Use new CalcRTM method in the warehouse class in Warehouse.py.
    IMPORTANT: Make sure that between the last update of Warehouse.sim_time and the execution
    of this function, no actions are scheduled and each agent's schedule (busy times) isn't
    altered.

    Calculate a new Response Time Matrix.  for each possible request given the current warehouse
    state. The prepositioning time for each location is included in the calculation. Based on
    whether a location (r, f, c) is occupied or not, the response time is calculated as a
    retrieval or a storage operation.

    Return None.
    """

    # Archive the previous RTM in a shift registry.
    self.rtm_history.insert(0, self.rtm)

    # Pop the oldest stored RTM if the shift registry's length exceeds the specified length.
    if len(self.rtm_history) >= self.num_historical_rtms:
        self.rtm_history.pop(-1)

    # Reset the current RTM.
    self.rtm = np.zeros_like(self.rtm)

    # For each shelf in the warehouse, calculate the response time.
    for i in range(0, self.num_locs):
        # Get the row, floor and column coordinates for shelf i.
        (r, f, c) = self.shelf_rfc[i]

        # Get the shelf access sequence for shelf i, e.g. the agents involved in accessing this
        # shelf. Remember, the sequence must be reversed for outfeed response time calculation.
        # (reversing does not really matter but it keeps you sane if you do it consistently)
        sequence = self.shelf_access_sequence[i]

        # Keep track of the longest prepositioning time for this sequence's agents. Only the
        # longest will be added to the final RTM, since every agent with a shorter
        # prepositioning time will already be done with prepositioning itself.
        max_prep_time = 0.0

        # In case the considered location (r, f, c) is not occupied, i.e. a possible infeed
        # location, we calculate the time required for infeeding an item to this shelf,
        # including the prepositioning of the agents to their default infeed location (0, 0, 0).
        if ~self.shelf_occupied[r, f, c]:
            for agent in list(sequence.keys()):
                busy_till = self.agent_busy_till[agent]
                agent_loc = self.agent_location[agent][busy_till]

                # Find out for how long the agent is busy, or if it's available immediately.
                agent_time_remaining = np.maximum(0, (busy_till - self.sim_time))

                # Calculate how long the agent would need for prepositioning from the location
                # it finished its last task at to the default infeed location for this agent.
                # Differentiate between 'vt' and 'sh_'. Note that the (r, f, c) of shelf_id[0]
                # is (0, 0, 0), it is not important that shuttles occupy different floors here.
                if agent == 'vt':
                    # Find out if this agent needs more prepositioning time than previous ones.
                    max_prep_time = np.maximum(max_prep_time,
                                               self.CalcAgentTravelTime(
                                                   agent,
                                                   self.shelf_id[0, agent_loc, 0],
                                                   0))
                else:
                    # Same as above, but for shuttles (column movement) instead of vertical
                    # transporters (floor movement).
                    max_prep_time = np.maximum(max_prep_time,
                                               self.CalcAgentTravelTime(
                                                   agent,
                                                   self.shelf_id[0, 0, agent_loc],
                                                   0))

                # Add the contribution of this agent to the response time for location (r, f, c)
                self.rtm[r, f, c] += (agent_time_remaining + sequence[agent])

        # In case the considered location (r, f, c) is occupied, we calculate the time required
        # for outfeeding the item occupying this shelf, which includes the prepositioning to
        # this shelf.
        else:
            # For each agent in the (reversed) sequence (since this is outfeed), calculate its
            # contribution to the response time for location (r, f, c) and add it to the
            # RTM(r, f, c).
            for agent in reversed(list(sequence.keys())):
                busy_till = self.agent_busy_till[agent]
                agent_loc = self.agent_location[agent][busy_till]

                # Find out for how long the agent is busy, or if it's available immediately.
                agent_time_remaining = np.maximum(0, (busy_till - self.sim_time))

                # Calculate how long the agent would need for prepositioning from the location
                # it finished its last task at to floor or column component of the considered
                # (r, f, c) where an item is stored. Differentiate between 'vt' and 'sh_'.
                if agent == 'vt':
                    # Find out if this agent needs more prepositioning time than previous ones.
                    max_prep_time = np.maximum(max_prep_time,
                                               self.CalcAgentTravelTime(
                                                   agent,
                                                   self.shelf_id[0, agent_loc, 0],
                                                   self.shelf_id[r, f, c]))
                else:
                    # Same as above, but for shuttles (column movement) instead of vertical
                    # transporters (floor movement).
                    max_prep_time = np.maximum(max_prep_time,
                                               self.CalcAgentTravelTime(
                                                   agent,
                                                   self.shelf_id[0, 0, agent_loc],
                                                   self.shelf_id[r, f, c]))

                # Add the contribution of this agent to the response time for location (r, f, c)
                self.rtm[r, f, c] += (agent_time_remaining + sequence[agent])

        # Finally, after each agent's contribution has been added, also add the maximum
        # prepositioning time to the response time for this location. This can be done
        # irrespective of whether infeed or outfeed response time was calculated.
        self.rtm[r, f, c] += max_prep_time


def Infeed(self, selected_shelf_id=None, rfc=None):
    """
    REDUNDANT: use Warehouse.ProcessAction
    Was located in the warehouse class in Warehouse.py.


    Return.

    -------
    None.

    """
    if selected_shelf_id is None and rfc is None:
        raise Exception("""No storage location designated! Specify either
                        a selected_shelf_id or an (r, f, c), or both.""")
    elif selected_shelf_id is not None and rfc is not None:
        if selected_shelf_id != self.shelf_id[rfc[0], rfc[1], rfc[2]]:
            raise Exception("""Two different locations were specified!""")
    else:
        if selected_shelf_id is None:
            selected_shelf_id = self.shelf_id[rfc[0], rfc[1], rfc[2]]
        else:
            (r, f, c) = self.shelf_rfc[selected_shelf_id]

    # Update agents' schedule.
    sequence = self.shelf_access_sequence[selected_shelf_id]

    # Start at the current simulation time.
    # TODO: Check if each transporter is in the required starting location.
    action_time = self.sim_time
    for agent in sequence.keys():
        # Calculate start time for transporter,
        # immediately increase for the next transporter in the sequence.
        action_time = np.maximum(
            action_time, self.agent_busy_till[agent]) + sequence[agent]

        # Update agent busy-untill time.
        self.agent_busy_till[agent] = action_time
        # Schedule agent future location, make distinction between vt/sh
        self.agent_location[agent][action_time] = c if agent != 'vt' else f

    # Label the selected shelf as occupied.
    (r, f, c) = self.shelf_rfc[selected_shelf_id]
    self.shelf_occupied[r, f, c] = True


def Outfeed(self, selected_shelf_id=None, rfc=None):
    """
    REDUNDANT: use Warehouse.ProcessAction
    Was located in the warehouse class in Warehouse.py.

    Return.

    -------
    None.

    """
    # TODO: write this function, use function above bot loop over
    # sequence.keys() in the opposite direction.
    if selected_shelf_id is None and rfc is None:
        raise Exception("""No retrieval location designated! Specify either
                        a selected_shelf_id or an (r, f, c), or both.""")
    elif selected_shelf_id is not None and rfc is not None:
        if selected_shelf_id != self.shelf_id[rfc[0], rfc[1], rfc[2]]:
            raise Exception("""Two different locations were specified!""")
    else:
        if selected_shelf_id is None:
            selected_shelf_id = self.shelf_id[rfc[0], rfc[1], rfc[2]]
        else:
            (r, f, c) = self.shelf_rfc[selected_shelf_id]

    # Check if the selected shelf isn't empty.
    if self.shelf_occupied[r, f, c] is False:
        raise Exception("""The selected location contains no item!""")

    # Update agents' schedule.
    sequence = self.shelf_access_sequence[selected_shelf_id]

    # Start at the current simulation time.
    # TODO: check if each transporter is in the right starting location.
    action_time = self.sim_time

    # Move through the sequence backwards
    for agent in reversed(sequence.keys()):
        # Calculate start time for transporter,
        # immediately increase for the next transporter in the sequence.
        action_time = np.maximum(
            action_time, self.agent_busy_till[agent]) + sequence[agent]

        # Update agent busy-untill time.
        self.agent_busy_till[agent] = action_time
        # Schedule agent future location.
        self.agent_location[agent][action_time] = 0

    # Label the previously occupied shelf as free again.
    (r, f, c) = self.shelf_rfc[selected_shelf_id]
    self.shelf_occupied[r, f, c] = False


"""_____________________________________Order system methods.____________________________________"""


def ExecuteNextOrder(self, current_time, prioritize_outfeed=True):
    """

    Summary.
    -------
    This method schedules a new order for execution. It takes the next
    item from either the infeed or outfeed order queue. It also logs the
    time spend waiting in the queue.


    Parameters
    ----------
    current_time : float
        The current simulation time.
    prioritize_outfeed : bool, optional
        Whether outfeed orders have priority over infeed orders. The
        default is True.

    Raises
    ------
    Exception
        When both order queues are empty.

    Returns
    -------
    None.

    """
    # Before executing an order, update the warehouse's sim_time!

    # Check if queues are empty.
    infeed_queue_empty = False if self.infeed_queue.qsize() > 0 else True
    outfeed_queue_empty = False if self.outfeed_queue.qsize() > 0 else True

    # Catch the case where both queues are empty.
    if infeed_queue_empty and outfeed_queue_empty:
        raise Exception(f"""Both queues are empty, there's no need to call
                        'ExecuteNextOrder' now at time {current_time}.""")

    # If there's an outfeed prioritization (to prevent saturated warehouse)
    if prioritize_outfeed:
        if not outfeed_queue_empty:
            next_order_id = self.outfeed_queue.get()
        else:
            if not infeed_queue_empty:
                next_order_id = self.infeed_queue.get()
    # No prioritization is given, 50/50 chance for both (if both non-empty)
    else:
        if not outfeed_queue_empty and not infeed_queue_empty:
            # Get random boolean.
            decider = bool(random.getrandbits(1))
            if decider is True:
                next_order_id = self.outfeed_queue.get()
            else:
                next_order_id = self.infeed_queue.get()
        elif not outfeed_queue_empty:
            next_order_id = self.outfeed_queue.get()
        elif not infeed_queue_empty:
            next_order_id = self.infeed_queue.get()

    self.order_register[next_order_id]["time_in_queue"] = current_time - \
        self.order_register[next_order_id]["time_created"]
    self.order_register[next_order_id]["in_queue"] = False
    self.order_register[next_order_id]["time_start"] = current_time

    return next_order_id


def GetNextOrder(self, current_time, free_and_occ, init_fill_perc=0.5, prioritize_outfeed=True):
    """
    Gets the next order from either the infeed or outfeed queue.

    Parameters
    ----------
    current_time : float
        The time at which the next order is requested. Should be current simulation time.
    free_and_occ : tuple.
        Contains the count of free and occupied shelves in the current warehouse state.
    prioritize_outfeed : bool, optional
        Whether outfeed should be prioritized over infeed. The default is True.

    Raises
    ------
    RuntimeError
        When the queues are empty or the warehouse is empty or full.

    Returns
    -------
    next_order_id : int
        The ID of the next order to be executed.
    order : dict
        The entry in the order_register that contains all the values of the order.

    """
    # Check if queues are empty.
    infeed_Q_empty = False if self.infeed_queue.qsize() > 0 else True
    outfeed_Q_empty = False if self.outfeed_queue.qsize() > 0 else True

    # Catch the case where both queues are empty.
    if infeed_Q_empty and outfeed_Q_empty:
        pass
        # Catch the cases where you want to infeed into a full warehouse or outfeed from an
        # empty warehouse.
    if infeed_Q_empty and free_and_occ[1] == 0:
        raise RuntimeError(f"Trying to outfeed from an empty warehouse! Time: {current_time}")
    elif outfeed_Q_empty and free_and_occ[0] == 0:
        raise RuntimeError(f"Trying to infeed to a full warehouse! Time: {current_time}")

    # If there's an outfeed prioritization (to prevent saturated warehouse)
    if prioritize_outfeed:
        if not outfeed_Q_empty:
            next_order_id = self.outfeed_queue.get()
        else:
            # Don't need to check if infeed queue is empty, that's already done above.
            next_order_id = self.infeed_queue.get()
    # No prioritization is given, 50/50 chance for both (if both non-empty)
    else:
        if not outfeed_Q_empty and not infeed_Q_empty:
            # Get random boolean.
            decider = bool(random.getrandbits(1))
            if decider is True:
                next_order_id = self.outfeed_queue.get()
            else:
                next_order_id = self.infeed_queue.get()
        elif not outfeed_Q_empty:
            next_order_id = self.outfeed_queue.get()
        elif not infeed_Q_empty:
            next_order_id = self.infeed_queue.get()

    # Calculate in-queue time, set start time.
    self.order_register[next_order_id]["time_in_queue"] = (
        current_time - self.order_register[next_order_id]["time_created"])
    self.order_register[next_order_id]["in_queue"] = False
    self.order_register[next_order_id]["time_start"] = current_time

    # Return order ID, order.
    return next_order_id, self.order_register[next_order_id]


"""_____________________________________train_model methods._____________________________________"""


def main():
    # x_dim = 8  # (column-1)
    # y_dim = 6 * 2  # row * 2
    torch.autograd.set_detect_anomaly(False)
    # num_rows = 2
    # num_floors = 6
    # num_cols = 6
    # episode_length = 72
    # num_hist_rtms = 5
    # num_hist_occs = 0  # Currently not in use!
    # vt_speed = 30.0
    # sh_speed = 1.0
    # percentage_filled = 0.0
    # wh_sim = wh(num_rows,
    #             num_floors,
    #             num_cols,
    #             episode_length,
    #             num_hist_rtms,
    #             num_hist_occs,
    #             vt_speed,
    #             sh_speed,
    #             percentage_filled)
    # scenario = "infeed"

    # train_episodes = 10  # This value exceeding 5000 is not recommended
    # # learning_rate = 1.78e-4
    # learning_rate = 1.78e-4
    # learning_rate_decay_factor = 0.8
    # learning_rate_decay_interval = 360
    # reward_discount_factor = 0.86
    # reward_type = "makespan_half_spectrum"
    # train_plant_model = TrainGameModel(wh_sim,
    #                                    lr=learning_rate,
    #                                    lr_decay=learning_rate_decay_factor,
    #                                    lr_decay_interval=learning_rate_decay_interval,
    #                                    discount_factor=reward_discount_factor,
    #                                    reward_type=reward_type)
    # Train the network; regular operation.
    # train_plant_model.RunTraining(train_episodes, scenario, baselines)

    # Benchmarks: Can run multiple in order. Note: be aware of the fact that plots are saved
    # locally! See SaveExperimentResults()
    # TODO: change col_by_col to rfc or something
    # train_plant_model.RunBenchmark(100, scenario=scenario, benchmark_policy='random')
    # train_plant_model.RunBenchmark(100, scenario=scenario, benchmark_policy='greedy')
    # train_plant_model.RunBenchmark(100, scenario=scenario, benchmark_policy='eps_greedy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='rcf_policy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='cfr_policy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='frc_policy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='fcr_policy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='rfc_policy')
    # train_plant_model.RunBenchmark(1, scenario=scenario, benchmark_policy='crf_policy')
    # sys.exit("training end")


"""________________________________policy_net_AC_pytorch methods.________________________________"""


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
