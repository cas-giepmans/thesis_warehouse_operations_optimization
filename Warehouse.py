# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:44 2022.

@author: casgi
"""
import numpy as np
from numba import jit
import random
from orders import OrderSystem


class Warehouse():
    """
    The warehouse class.

    Return.

    -------
    None.

    """

    def __init__(
            self,
            num_rows=2,
            num_floors=6,
            num_cols=6,
            episode_length=1000,
            num_hist_rtms=5,
            num_hist_occs=0,
            vt_speed=1.0,
            sh_speed=1.0,
            fill_perc=0.5,
            product_frequencies=[]):
        """


        Parameters
        ----------
        num_rows : int, optional
            Number of rows in the warehouse. The default is 2.
        num_floors : int, optional
            Number of floors in the warehouse. The default is 6.
        num_cols : int, optional
            Number of columns in the warehouse. The default is 6.
        episode_length : int, optional
            The number of orders that are processed before a simulation episode is over. The default is 1000.
        num_hist_rtms : int, optional
            The number of RTMs from previous simulation steps to include as input for the agent. The default is 5.
        num_hist_occs : int, optional
            Not in use. Number of previous availability matrices. The default is 0.
        vt_speed : float, optional
            Speed (m/s) of the vertical transporter. The default is 1.0.
        sh_speed : float, optional
            Speed (m/s) of the shuttle (horizontal transporter). The default is 1.0.
        fill_perc : float, optional
            The desired fulness of the warehouse. The default is 0.5.
        product_frequencies : list, optional
            List of the pass-through frequencies for each product. The default is [].

        Returns
        -------
        None.

        """
        # Warehouse dimensions
        self.num_rows = num_rows
        self.num_floors = num_floors
        self.num_cols = num_cols

        # 3-tuple housing the warehouse dimensions. For shorter notation
        self.dims = (self.num_rows, self.num_floors, self.num_cols)

        # Total number of storage locations in warehouse
        self.num_locs = self.num_rows * self.num_floors * self.num_cols

        # Numpy arrays to keep track of response times and occupancy.
        self.rtm = np.zeros(self.dims, np.float32)
        self.shelf_occupied = np.zeros(self.dims, dtype=bool)
        self.rtm_history = []
        self.occ_history = []

        # dictionary to store shelf coordinates, accessed through shelf_id.
        self.shelf_rfc = {}  # Key: shelf_id[r, c, f], val: (r, c, f)

        # Numpy array to keep track of shelf IDs
        self.shelf_id = np.zeros(self.dims, np.int16)

        # Variables about time
        self.episode_length = episode_length  # Nr. of orders per episode.
        self.num_historical_rtms = num_hist_rtms
        self.num_historical_occs = num_hist_occs
        self.sim_time = 0.0
        self.prev_action_time = 0.0
        self.v_vt = vt_speed
        self.v_sh = sh_speed

        # The physical distance between two adjacent storage locations or floors is 1.2 meters.
        self.floor_travel_time = 1.2 / 1.0 if vt_speed is None else 1.2 / vt_speed
        self.column_travel_time = 1.2 / 1.0 if sh_speed is None else 1.2 / sh_speed

        # Dictionary to store agent schedule.
        # There is a single vertical transporter, it is available after t = 0.
        # Also add a shuttle for each floor, each one is available after t = 0.
        self.agent_busy_till = {'vt': 0.0}
        for f in range(self.num_floors):
            self.agent_busy_till['sh'+str(f)] = 0.0

        # Dictionary of dictionaries to keep track of agents locations in time.
        # For each agent's location dict, the key is the simulation time, and
        # the value is the location index (should I use meters here?).
        self.agent_location = {'vt': {0.0: 0}}
        for f in range(self.num_floors):
            self.agent_location['sh'+str(f)] = {0.0: 0}

        # Count the number of storage/retrieval actions/orders.
        self.action_counter = 0

        # Specify how full the warehouse should be at the start of an episode.
        self.init_fill_perc = fill_perc

        # Initiate the order system.
        self.order_system = OrderSystem()

        # Initiate random number generator.
        self.rng = np.random.default_rng()

        # sequence of agents to acess to each shelf, with access time
        self.shelf_access_sequence = {}

        # Compute time needed to reach the shelf from infeed point
        # (assumes agents are at their default position)
        i = 0  # Initiate shelf ID counter.

        for c in range(self.num_cols):
            for f in range(self.num_floors):
                for r in range(self.num_rows):
                    self.shelf_access_sequence[i] = {
                        'vt': f * self.floor_travel_time,
                        'sh'+str(f): (1 + c) * self.column_travel_time}

                    self.shelf_id[r, f, c] = i
                    self.shelf_rfc[i] = (r, f, c)

                    i += 1

        # Variables related to multiple product types
        self.shelf_contents = np.zeros(self.dims, dtype=int)  # 0 means empty, 1 means type 1 etc.
        # Check to see if they sum up to 1.
        sum_frequencies = sum(product_frequencies)
        if sum_frequencies != 1.0:
            product_frequencies = [frequency /
                                   sum_frequencies for frequency in product_frequencies]
        self.product_frequencies = dict(list(enumerate(product_frequencies, start=1)))
        self.product_counts = dict(list(enumerate([0 for i in product_frequencies], start=1)))
        self.num_product_types = len(product_frequencies)

        # Variables used in benchmarking
        # TODO: this can be done neater, with recursion. Specify policy type, adjust recursion order.
        # Order in which shelves are accessed for col_by_col policy.
        # !!!: Assumption: we access floors in order, not randomly.
        self.rcf_sequence = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                for f in range(self.num_floors):
                    self.rcf_sequence.append(self.shelf_id[r, f, c])

        # Order in which shelves are accessed for col_by_col_alt policy.
        # !!!: Assumption: we access floors in order, not randomly.
        self.cfr_sequence = []
        for c in range(self.num_cols):
            for f in range(self.num_floors):
                for r in range(self.num_rows):
                    self.cfr_sequence.append(self.shelf_id[r, f, c])

        # Order in which shelves are accessed for tier_by_tier policy.
        # !!!: Assumption: we access columns in order, not randomly.
        self.frc_sequence = []
        for f in range(self.num_floors):
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    self.frc_sequence.append(self.shelf_id[r, f, c])

        # Order in which shelves are accessed for tier_by_tier_alt policy.
        # !!!: Assumption: we access floors in order, not randomly.
        self.fcr_sequence = []
        for f in range(self.num_floors):
            for c in range(self.num_cols):
                for r in range(self.num_rows):
                    self.fcr_sequence.append(self.shelf_id[r, f, c])

        # Given that there are 6 combinations of r, f and c, here are the last two:
        self.rfc_sequence = []
        for r in range(self.num_rows):
            for f in range(self.num_floors):
                for c in range(self.num_cols):
                    self.rfc_sequence.append(self.shelf_id[r, f, c])

        self.crf_sequence = []
        for c in range(self.num_cols):
            for r in range(self.num_rows):
                for f in range(self.num_floors):
                    self.crf_sequence.append(self.shelf_id[r, f, c])

    def ResetState(self, random_fill_percentage=None, dfg=[1, 1, 1]) -> list:
        """
        Resets the warehouse instance's variables, the order system's instance and returns a fresh
        state array. The state array's availability matrix (bottom matrix) specifies occupied
        shelves.

        Parameters
        ----------
        random_fill_percentage : float, optional
            If set, the warehouse is randomly filled for the given percentage. The default is None.
        dfg : list, optional
            Which extra input matrices to give the agent. These are the AMPT (d), FPM (f) and FPMPT (g).
            See thesis for more information on these.

        Returns
        -------
        fresh_wh_state : list
            A list of numpy matrices, containing historical RTMs, current RTM and availability
            matrix.

        """
        self.shelf_occupied = np.zeros(self.dims, dtype=bool)
        self.shelf_contents = np.zeros(self.dims, dtype=int)
        for key in self.product_counts.keys():
            self.product_counts[key] = 0

        # If requested, fill the shelves randomly up to a certain percentage.
        if random_fill_percentage is not None:
            self.SetRandomOccupancy(random_fill_percentage)
        else:
            self.SetRandomOccupancy(self.init_fill_perc)

        self.sim_time = 0.0
        self.prev_action_time = 0.0
        self.action_counter = 0
        self.agent_busy_till.clear()
        self.agent_location.clear()

        self.agent_busy_till = {'vt': 0.0}

        for f in range(self.num_floors):
            self.agent_busy_till['sh'+str(f)] = 0.0

        self.agent_location = {'vt': {0.0: 0}}
        for f in range(self.num_floors):
            self.agent_location['sh'+str(f)] = {0.0: 0}

        # Reset the current RTM. Calculate a couple of times to fill RTM history.
        self.rtm = np.zeros_like(self.rtm, np.float32)
        self.rtm_history.clear()
        for i in range(self.num_historical_rtms):
            self.CalcRTM()

        self.order_system.Reset()

        fresh_wh_state = self.BuildState(dfg=dfg)

        return fresh_wh_state

    def BuildState(self, infeed=True, product_type=1, dfg=[1, 1, 1]):
        """Build the new warehouse state, consisting of n historical RTMs, the
        current RTM and the binary matrix representing pickable locations,
        either for storage or for retrieval ops."""
        wh_state = []
        # Add historical RTMs, reshape them to fit Lei Luo's network.
        # TODO: deal with the reshaping for different warehouse architectures.
        for hrtm in self.rtm_history:
            rtm = np.reshape(hrtm, (self.dims[0] * self.dims[1], self.dims[2])).tolist()
            wh_state.append(rtm)
        # Add the current RTM.
        wh_state.append(np.reshape(self.rtm, (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        # Add the occupancy matrix.
        if infeed is True:
            wh_state.append(np.reshape(~self.shelf_occupied,
                            (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        else:
            wh_state.append(np.reshape(self.shelf_occupied,
                            (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        # Multi-product-related matrices.
        # One availability matrix for each product type.

        freq_matrix = np.zeros(self.dims)
        for p_type, p_freq in enumerate(self.product_frequencies, 1):
            av_matrix = np.where(self.shelf_contents == p_type, 1, 0)

            av_matrix = np.reshape(av_matrix,
                                   (self.dims[0] * self.dims[1], self.dims[2])).tolist()

            if dfg[0] == 1:
                wh_state.append(av_matrix)

            # Fill the frequency matrix with each product's respective frequency.
            freq_matrix = np.where(self.shelf_contents == p_type, p_freq, freq_matrix)

        chosen_product_matrix = np.where(self.shelf_contents == product_type, 1, 0)

        # Add the extra input matrices.
        if dfg[0] == 1:
            wh_state.append(np.reshape(chosen_product_matrix,
                                       (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        if dfg[1] == 1:
            wh_state.append(np.reshape(chosen_product_matrix * self.product_frequencies[product_type],
                                       (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        if dfg[2] == 1:
            wh_state.append(np.reshape(freq_matrix,
                                       (self.dims[0] * self.dims[1], self.dims[2])).tolist())

        return wh_state

    def GenerateNewOrder(self, scenario, free_and_occ):
        # Can only generate outfeed order for specific type if it's present in warehouse.
        unavailable_types = []
        if scenario == "infeed":
            pass
        else:
            for pair in self.product_counts.items():
                if pair[1] == 0:
                    unavailable_types.append(pair[0])
        # print(f"unavailable types: {unavailable_types}")

        if scenario == "both":
            infeed_prob = 0.0

            cur_f = round(free_and_occ[1] / self.num_locs, 2)  # Current fullness
            des_f = self.init_fill_perc  # Desired fullness

            # Calculate the probability for generating an infeed order.
            if cur_f == des_f:
                infeed_prob = 0.5
            elif cur_f > des_f:
                infeed_prob = 1 - (0.5 / (1.0 - des_f) * cur_f - (0.5 / (1.0 - des_f) - 1))
            elif cur_f < des_f:
                infeed_prob = 1 - 0.5 / des_f * cur_f

            # Draw the order type randomly according to the fulness.
            order_type = "infeed" if self.rng.uniform() < infeed_prob else "outfeed"
        else:
            if scenario == "infeed":
                order_type = "infeed"
            else:
                order_type = "outfeed"

        # Draw a product type based on given probabilities and availability.
        if order_type == "infeed":
            product_type = self.DrawProductType()
        else:
            product_type = self.DrawProductType(unavailable_types)

        # Finally, generate a new order.
        self.order_system.GenerateNewOrder(self.sim_time, order_type, free_and_occ, product_type)

    def GetNextOrder(self, free_and_occ):
        """"Pass-through for the method in the order system of the same name."""
        return self.order_system.GetNextOrder(self.sim_time, free_and_occ)

    def CalcShelfAccessTime(self, shelf_id, infeed=True):
        """Calculates the time needed for accessing a shelf, either for infeed or outfeed."""

        # Get the shelf access sequence for given shelf.
        sequence = self.shelf_access_sequence[shelf_id]

        # The sequence must be reversed for outfeed response time calculation.
        agents = list(sequence.keys())
        if not infeed:
            agents.reverse()
        pth = (0, shelf_id) if infeed else (shelf_id, 0)

        # initiate time
        curr_time = self.sim_time

        # move agents one by one to access to the shelf
        for idx, agent in enumerate(agents):
            # Agent will be available at this position at this time
            busy_till = self.agent_busy_till[agent]
            agent_loc = self.agent_location[agent][busy_till]

            # Wait till agent is ready (assuming that we don't have a prepositioning policy),
            # but agent can start moving immediately for prepositioning if it is not busy.
            busy_till = np.maximum(busy_till, self.sim_time)

            # Perform prepositioning, no need to wait for other agents
            if agent == 'vt':
                busy_till += self.CalcAgentTravelTime(agent,
                                                      self.shelf_id[0, agent_loc, 0], pth[0], False)
            else:
                busy_till += self.CalcAgentTravelTime(agent,
                                                      self.shelf_id[0, 0, agent_loc], pth[0], False)

            # Pick-up item and go to required location, the agents who will get the item will wait.
            curr_time = np.maximum(busy_till, curr_time) + self.CalcAgentTravelTime(agent,
                                                                                    pth[0],
                                                                                    pth[1])

            # We need to account for the first agent in a sequence potentially waiting until the
            # second agent is at the hand over point.
            if idx == 0:
                curr_time = np.maximum(curr_time, self.agent_busy_till[agents[idx + 1]])

        # Return the relative time
        return curr_time - self.sim_time

    def CalcRTM(self):
        """
        Important: make sure to update Warehouse.sim_time before you calculate the RTM, and that you
        calculate the RTM before you change the warehouse state (by performing an action).

        Calculate the Response Time Matrix (RTM) at the current sim time.

        Returns
        -------
        None.

        """
        # Archive the previous RTM in a shift registry.
        self.rtm_history.insert(0, self.rtm)

        # Pop the oldest stored RTM if the shift registry's length exceeds the specified length.
        if len(self.rtm_history) >= self.num_historical_rtms:
            self.rtm_history.pop(-1)

        # Reset the current RTM.
        self.rtm = np.zeros_like(self.rtm)

        # For each shelf in the warehouse, calculate the access time.
        for shelf_id in range(self.num_locs):
            # Get the row, floor and column coordinates for this shelf.
            (r, f, c) = self.shelf_rfc[shelf_id]

            # If the shelf is occupied, we can infeed to this location.
            if ~self.shelf_occupied[r, f, c]:
                self.rtm[r, f, c] = self.CalcShelfAccessTime(shelf_id, infeed=True)
            # Else, the shelf isn't occupied, so we can outfeed from this location.
            else:
                self.rtm[r, f, c] = self.CalcShelfAccessTime(shelf_id, infeed=False)

        # Normalize the RTM to a value between 0 and 1.
        _max = self.rtm.max()
        _min = self.rtm.min()
        self.rtm = (self.rtm - _min) / (_max - _min)

    def ReadyTransporters(self, target_shelf_id, infeed=True):
        """
        Given the ID of a shelf to be accessed and the type of operation, move
        every involved agent to the required starting location and update their
        busy times and locations accordingly.

        Parameters.

        ----------
        shelf_id : int
            ID of the shelf to be accessed.
        infeed : bool, optional
            Whether the coming order is infeed or outfeed. The default is True.

        """
        # Get all the agents involved in accessing the selected shelf.
        sequence = self.shelf_access_sequence[target_shelf_id]
        (r, f, c) = self.shelf_rfc[target_shelf_id]

        # If infeed: move all involved agents to position 0.
        if infeed:
            for agent in sequence.keys():
                # Calculate time necessary for prepositioning agent
                agent_loc = self.agent_location[agent][self.agent_busy_till[agent]]
                if agent == 'vt':
                    from_id = self.shelf_id[r, agent_loc, c]
                else:
                    from_id = self.shelf_id[r, f, agent_loc]
                agent_preposit_time = self.CalcAgentTravelTime(agent, from_id, 0, False)

                # Calculate when agent is done with prepositioning
                new_busy_time = np.maximum(
                    self.sim_time, self.agent_busy_till[agent]) + agent_preposit_time

                # Update agent's busy time and location.
                self.agent_busy_till[agent] = new_busy_time
                self.agent_location[agent][new_busy_time] = 0
        # If outfeed: move all involved agents to the selected location
        else:
            for agent in sequence.keys():
                # Calculate time necessary for prepositioning agent
                agent_loc = self.agent_location[agent][self.agent_busy_till[agent]]
                if agent == 'vt':
                    from_id = self.shelf_id[r, agent_loc, c]
                else:
                    from_id = self.shelf_id[r, f, agent_loc]
                agent_preposit_time = self.CalcAgentTravelTime(
                    agent, from_id, target_shelf_id, False)

                new_busy_time = np.maximum(
                    self.sim_time, self.agent_busy_till[agent]) + agent_preposit_time

                # Update agent's busy time and location.
                self.agent_busy_till[agent] = new_busy_time
                if agent == 'vt':
                    self.agent_location[agent][new_busy_time] = f
                else:
                    self.agent_location[agent][new_busy_time] = c

    def CalcAgentTravelTime(self, agent, from_id, to_id, for_access=True) -> float:
        """
        Calculate the time required by an agent from traveling from point A to point B. Shuttles
        always take at least 1 * V_sh extra, otherwise some storage actions would take 0 seconds.
        This would mess up the simulation.

        Parameters:

        ----------
        agent : str
            Name of the agent we're calculating the travel time for.
        from_id : int
            The ID of the origin
        to_id : int
            The ID of the target location
        for_access : bool
            Whether to add + 1 for shuttle shelf access operation

        ----------
        Returns travel time.

        """
        if agent != 'vt':
            _, _, c_origin = self.shelf_rfc[from_id]
            _, _, c_target = self.shelf_rfc[to_id]
            if for_access is True:
                return (abs(c_origin - c_target) + 1) * self.column_travel_time
            else:
                return abs(c_origin - c_target) * self.column_travel_time
        else:
            _, f_origin, _ = self.shelf_rfc[from_id]
            _, f_target, _ = self.shelf_rfc[to_id]
            return abs(f_origin - f_target) * self.floor_travel_time

    def ProcessAction(self, infeed, product_type, selected_shelf_id=None, rfc=None):
        """
        Perform an infeed/outfeed action. The returned action_time includes the time needed for
        prepositioning agents. Used to be two functions 'Infeed' and 'Outfeed' but this seemed neater.

        Parameters:

        ----------
        infeed : bool
            Whether the order is infeed, if not then it's outfeed.
        selected_shelf_id : int
            The ID of the shelf from which to retrieve or to store at.
        rfc : 3-tuple
            Tuple containing the row, floor and column of the selected shelf.

        ----------
        Returns action_time, is_end.

        """
        if selected_shelf_id is None and rfc is None:
            raise Exception("""No location designated! Specify either a
                            selected_shelf_id or an (r, f, c), or both.""")
        elif selected_shelf_id is not None and rfc is not None:
            if selected_shelf_id != self.shelf_id[rfc[0], rfc[1], rfc[2]]:
                raise Exception("""Two different locations were specified!""")
        else:
            if selected_shelf_id is None:
                selected_shelf_id = self.shelf_id[rfc[0], rfc[1], rfc[2]]
            else:
                (r, f, c) = self.shelf_rfc[selected_shelf_id]

        # Preposition the involved transporters.
        self.ReadyTransporters(selected_shelf_id, infeed)

        # Get the shelf access sequence for given shelf.
        sequence = self.shelf_access_sequence[selected_shelf_id]

        # The sequence must be reversed for outfeed response time calculation.
        agents = list(sequence.keys())
        if not infeed:
            agents.reverse()
        pth = (0, selected_shelf_id) if infeed else (selected_shelf_id, 0)

        # Initially set the action time (when it finishes) to be the same as the simulation time.
        action_time = self.sim_time

        # Add each agent's contribution to the total action time.
        for idx, agent in enumerate(agents):
            # Calculate the agent travel time.
            agent_travel_time = self.CalcAgentTravelTime(agent, pth[0], pth[1])

            # Update the time needed for executing the complete storage/retrieval action
            action_time = np.maximum(
                action_time, self.agent_busy_till[agent]) + agent_travel_time

            # Set the new busy time of this agent.
            # If it is the first agent in the sequence, then it is either busy until it reaches its
            # destination, or until the second agent arrives at the handover point, whichever one is
            # larger.
            if idx == 0:
                self.agent_busy_till[agent] = np.maximum(
                    action_time, self.agent_busy_till[agents[idx + 1]])
            # If it is the second agent in the sequence, there is no next agent it could potentially
            # wait for, so it will only be busy until it is done with the action.
            else:
                self.agent_busy_till[agent] = action_time

            # Set the agent's location at the end of this action. Note that action_time here relates
            # to each agent individually, and not to the total action time. Otherwise the involved
            # agents would all be "busy" doing nothing after they finish their part in the action.
            if infeed:
                if agent == 'vt':
                    self.agent_location[agent][self.agent_busy_till[agent]] = f
                else:
                    self.agent_location[agent][self.agent_busy_till[agent]] = c
            else:
                self.agent_location[agent][self.agent_busy_till[agent]] = 0

        self.shelf_occupied[r, f, c] = True if infeed else False
        self.shelf_contents[r, f, c] = product_type if infeed else 0

        # Increase/decrease product count.
        self.product_counts[product_type] += 1 if infeed else -1

        # Determine if this the episode length has been reached.
        self.action_counter += 1
        is_end = True if self.action_counter >= self.episode_length else False

        return action_time, is_end

    """
    Below are some utility functions and pass-throughs.
    |-------------------------------------------------------------------------|
    """

    def Get_RTM(self, rfc=None):
        """Return the Response Time Matrix (RTM), or an entry from the RTM."""
        return self.rtm if rfc is None else self.rtm[rfc[0], rfc[1], rfc[2]]

    def PrintRTM(self):
        """Print the correctly oriented response time matrix."""
        print(np.flip(self.rtm.round(2), axis=1))

    def PrintIdMatrix(self, print_it=True):
        """Prints the warehouse with each shelf containing its shelf_id, correctly oriented."""
        id_matrix = np.zeros(self.dims, dtype=int)
        for shelf_id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[shelf_id]
            id_matrix[r, f, c] = shelf_id
        if print_it is True:
            print(np.flip(id_matrix, axis=1))
        else:
            return id_matrix

    def PrintFinishTimeMatrix(self, print_it=True):
        """Prints the warehouse with each shelf containing its finish_time, correctly oriented."""
        fin_matrix = np.zeros(self.dims, dtype=float)
        for order_id in self.order_system.order_register.keys():
            (r, f, c) = self.shelf_rfc[self.order_system.order_register[order_id]['shelf_id']]
            fin_matrix[r, f, c] = round(
                self.order_system.order_register[order_id]['time_finish'], 2)
        if print_it is True:
            print(np.flip(fin_matrix, axis=1))
        else:
            return fin_matrix

    def PrintStartTimeMatrix(self, print_it=True):
        """Prints the warehouse with each shelf containing its start time, correctly oriented."""
        start_matrix = np.zeros(self.dims, dtype=float)
        for order_id in self.order_system.order_register.keys():
            (r, f, c) = self.shelf_rfc[self.order_system.order_register[order_id]['shelf_id']]
            start_matrix[r, f, c] = round(
                self.order_system.order_register[order_id]['time_start'], 2)
        if print_it is True:
            print(np.flip(start_matrix, axis=1))
        else:
            return start_matrix

    def PrintOccupancy(self):
        """Print the correctly oriented occupancy matrix."""
        print(np.flip(self.shelf_occupied, axis=1))

    def GetNextBenchmarkPolicyShelfId(self, infeed, bench_pol="random", product_type=1, eps=0.1) -> np.int16:
        """Get the next shelf ID according to a benchmark policy. Possible benchmark policies are
        'random', 'greedy', 'eps_greedy', 'rcf_policy', 'cfr_policy', 'frc_policy' and
        'fcr_policy'. Only meant for benchmarking infeed-only scenarios."""

        if infeed:
            product_type = 0

        # Stochastic policies.
        if bench_pol == 'random':
            return self.GetRandomShelfId(product_type=product_type)
        elif bench_pol == 'greedy':
            return self.GetGreedyShelfId(product_type=product_type)
        elif bench_pol == 'eps_greedy':
            return self.GetEpsGreedyShelfId(eps=eps, product_type=product_type)
        else:
            # Deterministic (rule-based) policies.
            if bench_pol == 'rcf_policy':
                return self.FindNextShelfIdInSequence(self.rcf_sequence, product_type)
            elif bench_pol == 'cfr_policy':
                return self.FindNextShelfIdInSequence(self.cfr_sequence, product_type)
            elif bench_pol == 'frc_policy':
                return self.FindNextShelfIdInSequence(self.frc_sequence, product_type)
            elif bench_pol == 'fcr_policy':
                return self.FindNextShelfIdInSequence(self.fcr_sequence, product_type)
            elif bench_pol == 'rfc_policy':
                return self.FindNextShelfIdInSequence(self.rfc_sequence, product_type)
            elif bench_pol == 'crf_policy':
                return self.FindNextShelfIdInSequence(self.crf_sequence, product_type)
            else:
                ValueError(f"There is no benchmark policy called '{bench_pol}'.")

    def FindNextShelfIdInSequence(self, sequence, product_type=1) -> np.int16:
        """Find the next available shelf ID using a sequence in the warehouse given the product type
           (infeed uses product_type=0)."""
        for shelf_id in sequence:
            (r, f, c) = self.shelf_rfc[shelf_id]
            if self.shelf_contents[r, f, c] == product_type:
                return shelf_id
            else:
                continue

    def GetRandomShelfId(self, product_type=1) -> np.int16:
        """Return random shelf ID that contains a product of the specified type or is empty (0)."""
        rfc = self.rng.choice(np.argwhere(self.shelf_contents == product_type).tolist(), axis=0)
        return int(self.shelf_id[rfc[0], rfc[1], rfc[2]])

    def GetGreedyShelfId(self, product_type=1) -> np.int16:
        """Return a shelf ID that has the lowest response time, given an infeed/outfeed request."""
        candidate_ids = self.GetIds(product_type)

        min_rt = float('inf')
        min_rt_shelf_ids = []

        for c_id in candidate_ids:
            (r, f, c) = self.shelf_rfc[c_id]
            if self.rtm[r, f, c] > min_rt:  # If response time is greater
                continue
            elif self.rtm[r, f, c] == min_rt:  # If response time is equal to smallest-encountered
                min_rt_shelf_ids.append(c_id)
            else:  # If response time is better than previously encountered
                min_rt = self.rtm[r, f, c]
                min_rt_shelf_ids.clear()
                min_rt_shelf_ids.append(c_id)
        return self.rng.choice(min_rt_shelf_ids)

    def GetEpsGreedyShelfId(self, eps=0.1, product_type=1) -> np.int16:
        """Return a shelf ID that has the lowest response time with prob. (1-eps), else return a
        random ID."""
        if self.rng.uniform() <= eps:
            return self.GetRandomShelfId(product_type)
        else:
            return self.GetGreedyShelfId(product_type)

    def GetIds(self, product_type=1) -> list:
        """Get the IDs of shelves, occupied or free, given a product type (0 for infeed)."""
        ids = []

        for _id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[_id]
            if self.shelf_contents[r, f, c] == product_type:
                ids.append(np.int16(_id))
            else:
                continue
        return ids

    def SetRandomOccupancy(self, fill_percentage=0.5):
        """Randomly sets a percentage of shelves to either occupied or not."""
        locs = [*range(self.num_locs - 1)]
        init_fills = np.random.choice(locs, size=int(
            self.num_locs * fill_percentage), replace=False)
        for shelf_id in init_fills:
            r, f, c = self.shelf_rfc[shelf_id]
            self.shelf_occupied[r, f, c] = True

            # Randomly assign a product type to this shelf.
            product_type = self.DrawProductType()
            self.shelf_contents[r, f, c] = product_type
            self.product_counts[product_type] += 1

    def GetShelfAccessDensities(self, normalized=True, print_it=False):
        """Calculate and return the access density matrix for an episode."""
        infeed_counts, outfeed_counts = self.order_system.GetShelfAccessCounts()
        infeed_density = np.zeros_like(self.shelf_occupied, dtype=float)
        outfeed_density = np.zeros_like(self.shelf_occupied, dtype=float)

        # For each shelf, assign the infeed and outfeed count to their representation in the
        # density matrices. Catch exception where a shelf was never accessed.
        for shelf_id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[shelf_id]
            try:
                infeed_density[r, f, c] = infeed_counts[shelf_id]
            except KeyError:
                infeed_density[r, f, c] = 0

            try:
                outfeed_density[r, f, c] = outfeed_counts[shelf_id]
            except KeyError:
                outfeed_density[r, f, c] = 0

        if normalized is True:
            # Normalize the matrices: get min and max, subtract min from matrix and divide that by
            # the difference between max and min. Yields values between 0 and 1 (inclusive).
            matrix_max = np.max(infeed_density)
            matrix_min = np.min(infeed_density)
            infeed_density = (infeed_density - matrix_min) / (matrix_max - matrix_min)

            matrix_max = np.max(outfeed_density)
            matrix_min = np.min(outfeed_density)
            outfeed_density = (outfeed_density - matrix_min) / (matrix_max - matrix_min)

        # If it has to be printed.
        if print_it is True:
            print(np.flip(infeed_density.round(2), axis=1))
            print(np.flip(outfeed_density.round(2), axis=1))

        # Return as a single numpy.ndarray object for convenience.
        access_densities = np.stack([infeed_density, outfeed_density], axis=0)
        return access_densities

    # TODO: calculate the ... time in the warehouse for each shelf, each product

    def GetMostCommonProductTypes(self, normalize=True, print_it=False):
        """For each shelf, for each product type, count how many orders involved that shelf. Then
           return the most occuring product type for each shelf."""
        bad_orders = 0
        most_common_type_matrix = np.zeros(self.dims, dtype=int)

        # Create a 2D counter
        type_counter = {}
        for shelf in range(self.num_locs):
            type_counter[shelf] = {}
            for product_type in range(1, self.num_product_types + 1, 1):
                type_counter[shelf][product_type] = 0.0

        # Iterate over orders, count the product type occurences.
        for order in self.order_system.order_register.values():
            if order["time_finish"] is None:
                # This order never finished for some reason.
                bad_orders += 1
            else:
                type_counter[order["shelf_id"]][order["product_type"]] += 1.0

        # Iterate over shelves, normalize (divide count by type frequency).
        if normalize:
            for shelf_id in range(self.num_locs):
                for product_type in range(1, self.num_product_types + 1, 1):
                    type_counter[shelf_id][product_type] /= self.product_frequencies[product_type]

        # Find the most common product type for each shelf.
        for shelf_id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[shelf_id]
            most_common_type = 1  # Initialize to 1 in order to avoid key error.
            for product_type in type_counter[shelf_id].items():
                if product_type[1] > type_counter[shelf_id][most_common_type]:
                    most_common_type = product_type[0]
                else:
                    continue
            most_common_type_matrix[r, f, c] = most_common_type

        if bad_orders != 0:
            print(f"There were {bad_orders} bad orders that didn't finish.")
        else:
            # print("There were no bad orders that didn't finish.")
            pass

        if print_it:
            print("Most occuring product for each shelf type:")
            print(np.flip(most_common_type_matrix, axis=1))

        return most_common_type_matrix

    def GetProductTypeDoS(self, print_it=False):
        """For each shelf, get a list of counters that denote how often each
           product type was stored there."""
        # Counter, just in case.
        bad_orders = 0
        # Counter to keep track of number of orders per product type per shelf.
        counter_matrix = np.zeros(
            (self.dims[0], self.dims[1], self.dims[2], self.num_product_types + 1), dtype=float)
        # Counter to track cumulative storage time per type per shelf.
        time_matrix = np.zeros(
            (self.dims[0], self.dims[1], self.dims[2], self.num_product_types + 1), dtype=float)
        # List to keep track of when each shelf was last filled, so you can calc storage duration.
        shelf_drop_off_times = [0.0 for i in range(self.num_locs)]

        # Iterate over orders.
        for order in self.order_system.order_register.values():
            if order["time_finish"] is None:
                # This order never finished for some reason.
                bad_orders += 1
            else:
                # Get the location.
                r, f, c = self.shelf_rfc[order["shelf_id"]]
                # Increase the count
                counter_matrix[r, f, c, order["product_type"]] += 1

                if order["order_type"] == "outfeed":
                    # If order was outfeed, add duration of stay to total.
                    time_matrix[r, f, c, order["product_type"]
                                ] += (order["time_finish"] - shelf_drop_off_times[order["shelf_id"]])
                    # Start counting for empty-shelf-duration
                    shelf_drop_off_times[order["shelf_id"]] = order["time_finish"]
                    counter_matrix[r, f, c, 0] += 2  # 2 because later it gets divided by 2
                if order["order_type"] == "infeed":
                    # Increase the empty-shelf-duration.
                    time_matrix[r, f, c, 0] += (order["time_finish"] -
                                                shelf_drop_off_times[order["shelf_id"]])
                    # If order was infeed, set the time at which the product was stored.
                    shelf_drop_off_times[order["shelf_id"]] = order["time_finish"]

        # Get actual number of products stored there, not number of orders going here.
        counter_matrix += 0.5
        counter_matrix /= 2.0
        # Divide each cumulative duration of stay by the number of products there.
        time_matrix = np.divide(time_matrix, counter_matrix)

        if print_it:
            for i in range(self.num_locs):
                r, f, c = self.shelf_rfc[i]
                print(f"shelf RFC: {r, f, c}, DoS: {time_matrix[r, f, c, :]}")

        return time_matrix

    def DrawProductType(self, unavailable_types=[]) -> int:
        """Draw a product type according to predefined probabilities and availability."""

        dict_items = list(zip(*self.product_frequencies.items()))

        types = list(dict_items[0])
        freqs = list(dict_items[1])

        if unavailable_types:  # boolean interpretation of non-empty list equates to True.
            # Iterate in reversed order, otherwise you change the indices as you pop from the lists.
            for _type in reversed(unavailable_types):
                types.pop(_type - 1)  # -1, because type 1 has index 0.
                freqs.pop(_type - 1)  # same here.
            # Don't have to renormalize freqs, random.choices does this automatically.
        else:  # No unavailable product types.
            pass

        product_type = random.choices(types, freqs, k=1)
        return product_type[0]


# Legacy Warehouse class tester.
# def main():
#     test_wh = Warehouse()
    # item_id_a = 0
    # item_id_b = 12

    # test_wh.CalcRTM()
    # test_wh.PrintRTM()
    # print(test_wh.agent_location)

    # # Infeed an item
    # test_wh.ProcessAction(infeed=True, selected_shelf_id=item_id_a)
    # # Calculate the new response time matrix given the new infeed order
    # test_wh.CalcRTM()
    # # Print it
    # test_wh.PrintRTM()
    # # Progress simulation time
    # # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_a])
    # test_wh.sim_time += 1
    # print(test_wh.agent_location)

    # test_wh.ProcessAction(infeed=True, selected_shelf_id=item_id_b)
    # test_wh.CalcRTM()
    # test_wh.PrintRTM()
    # # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_b])
    # # print(test_wh.shelf_rfc[item_id_b])
    # test_wh.sim_time += 5.4
    # test_wh.CalcRTM()
    # test_wh.PrintRTM()
    # print(test_wh.agent_location)
    # print(test_wh.shelf_occupied)
    # # print(test_wh.GetGreedyShelfId(infeed=False))
    # print(test_wh.GetRandomShelfId())

    # test_wh.CalcRTM()
    # test_wh.PrintRTM()

    # for i in range(10000):
    #     test_wh.CalcRTM()


# if __name__ == '__main__':
#     main()
