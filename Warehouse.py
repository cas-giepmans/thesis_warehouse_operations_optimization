# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:44 2022.

@author: casgi
"""
import numpy as np
import random
from orders import OrderSystem


class Warehouse():
    """


    Return.

    -------
    None.

    """

    def __init__(
            self,
            num_rows=2,
            num_floors=6,
            num_cols=8,
            episode_length=1000,
            num_hist_rtms=2,
            num_hist_occs=2):
        """


        Return.

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
        self.rtm = np.zeros(self.dims)
        self.shelf_occupied = np.zeros(self.dims, dtype=bool)
        self.rtm_history = []
        self.occ_history = []

        # dictionary to store shelf coordinates, accessed through shelf_id.
        self.shelf_rfc = {}  # Key: shelf_id[r, c, f], val: (r, c, f)

        # Numpy array to keep track of shelf IDs
        self.shelf_id = np.zeros(self.dims)

        # Variables about time
        self.episode_length = episode_length  # Nr. of orders per episode.
        self.num_historical_rtms = num_hist_rtms
        self.num_historical_occs = num_hist_occs
        self.sim_time = 0.0
        self.last_episode_time = 0.0
        self.vt_floor_travel_time = 2.0 / 5.0  # 2m floor height, 5m/s vt speed
        self.column_travel_time = 1.8 / 1  # 1.8m col width, 1m/s shuttle speed

        # Dictionary to store agent schedule.
        # There is a single vertical transporter, it is available after t = 0.
        # Also add a shuttle for each floor, each one is available after t = 0.
        self.agent_busy_till = {'vt': 0}
        for f in range(self.num_floors):
            self.agent_busy_till['sh'+str(f)] = 0

        # Dictionary of dictionaries to keep track of agents locations in time.
        # For each agent's location dict, the key is the simulation time, and
        # the value is the location index (should I use meters here?).
        self.agent_location = {'vt': {0.0: 0}}
        for f in range(self.num_floors):
            self.agent_location['sh'+str(f)] = {0.0: 0}

        # sequence of agents to acess to each shelf, with access time
        self.shelf_access_sequence = {}

        # Count the number of storage/retrieval actions.
        self.action_counter = 0

        # Initiate the order system.
        self.order_system = OrderSystem()

        # Initiate random number generator.
        self.rng = np.random.default_rng()

        # Compute time needed to reach the shelf from infeed point
        # (assumes agents are at their default position)
        i = 0  # Initiate shelf ID counter.

        for c in range(self.num_cols):
            for f in range(self.num_floors):
                for r in range(self.num_rows):
                    self.shelf_access_sequence[i] = {
                        'vt': f * self.vt_floor_travel_time,
                        'sh'+str(f): (1 + c) * self.column_travel_time}

                    self.shelf_id[r, f, c] = i
                    self.shelf_rfc[i] = (r, f, c)

                    i += 1

    def ResetState(self, random_fill_percentage=None):
        """Initiate the state for a new training episode."""
        self.shelf_occupied = np.zeros(self.dims, dtype=bool)

        # If requested, fill the shelves randomly up to a certain percentage.
        if random_fill_percentage is not None:
            self.SetRandomOccupancy(fill_percentage=random_fill_percentage)
        self.last_episode_time = self.sim_time
        self.sim_time = 0.0
        self.action_counter = 0
        self.agent_busy_till.clear()
        self.agent_location.clear()

        self.agent_busy_till = {'vt': 0}

        for f in range(self.num_floors):
            self.agent_busy_till['sh'+str(f)] = 0

        self.agent_location = {'vt': {0.0: 0}}
        for f in range(self.num_floors):
            self.agent_location['sh'+str(f)] = {0.0: 0}

        # Reset the current RTM. Calculate a couple of times to fill RTM history.
        self.rtm = np.zeros_like(self.rtm)
        self.rtm_history.clear()
        for i in range(self.num_historical_rtms):
            self.CalcRTM()

        self.order_system.Reset()

        fresh_wh_state = self.GetState()

        return fresh_wh_state

    # CHECK: is this one really necessary?
    def GetState(self, infeed=True):
        """Return the newly built warehouse state."""
        return self.BuildState(infeed)

    # CHECK: don't you need occupancy history?
    def BuildState(self, infeed=True):
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
            wh_state.append(np.reshape(self.shelf_occupied,
                            (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        else:
            wh_state.append(np.reshape(~self.shelf_occupied,
                            (self.dims[0] * self.dims[1], self.dims[2])).tolist())
        return wh_state

    def CalcShelfAccessTime(self, shelf_id, infeed=True):
        """Calculates the time needed for accessing a shelf, either for infeed or outfeed."""

        # Get the shelf access sequence for given shelf.
        sequence = self.shelf_access_sequence[shelf_id]

        # The sequence must be reversed for outfeed response time calculation.
        agents = sequence.keys() if infeed else reversed(list(sequence.keys()))
        pth = (0, shelf_id) if infeed else (shelf_id, 0)

        # initiate time
        curr_time = self.sim_time

        # move agents one by one to access to the shelf
        for agent in agents:
            # Agent will be available at this position at this time
            busy_till = self.agent_busy_till[agent]
            agent_loc = self.agent_location[agent][busy_till]

            # Wait till agent is ready (assuming that we don't have a prepositioning policy),
            # but agent can start moving immediately for prepositioning if it is not busy.
            busy_till = np.maximum(busy_till, self.sim_time)

            # Perform prepositioning, no need to wait for other agents
            # print(f"agent location: {agent_loc}, agent: {agent}")
            if agent == 'vt':
                busy_till += self.CalcAgentTravelTime(agent, self.shelf_id[0, agent_loc, 0], pth[0])
            else:
                busy_till += self.CalcAgentTravelTime(agent, self.shelf_id[0, 0, agent_loc], pth[0])
            # try:
            #     busy_till += self.CalcAgentTravelTime(agent, self.shelf_id[0, agent_loc, 0], pth[0])
            # except IndexError:
            #     print(f"""Sim time: {self.sim_time},\nAgent: {agent},\nbusy_till: {busy_till},
            #           \rShelf ID: {shelf_id},\nPath_0: {pth[0]},\nPath_1: {pth[1]},\nInfeed: {infeed}.\n""")
            #     print(self.order_system.order_register[self.action_counter].values())

            # Pick-up item and go to required location, the agents who will get the item will wait.
            curr_time = np.maximum(busy_till, curr_time) + self.CalcAgentTravelTime(agent,
                                                                                    pth[0],
                                                                                    pth[1])

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

    def ReadyTransporters(self, shelf_id, infeed=True):
        """

        Given the ID of a shelf to be accessed and the type of operation, move
        every involved agent to the required starting location and update their
        busy times accordingly.

        Parameters.

        ----------
        shelf_id : int
            ID of the shelf to be accessed.
        infeed : bool, optional
            Whether the coming order is infeed or outfeed. The default is True.

        """
        # Get all the agents involved in accessing the selected shelf.
        sequence = self.shelf_access_sequence[shelf_id]

        # If infeed: move all involved agents to position 0.
        if infeed:
            for agent in sequence.keys():
                # Calculate time necessary for prepositioning agent
                agent_preposit_time = self.CalcAgentTravelTime(
                    agent, self.agent_location[agent][self.agent_busy_till[agent]], 0)

                # Calculate when agent is done with prepositioning
                new_busy_time = np.maximum(
                    self.sim_time, self.agent_busy_till[agent]) + agent_preposit_time

                # Update agent's busy time
                self.agent_busy_till[agent] = new_busy_time
        # If outfeed: move all involved agents to the selected location
        else:
            for agent in sequence.keys():
                # Calculate time necessary for prepositioning agent
                agent_preposit_time = self.CalcAgentTravelTime(
                    agent, self.agent_location[agent][self.agent_busy_till[agent]], shelf_id)

                new_busy_time = np.maximum(
                    self.sim_time, self.agent_busy_till[agent]) + agent_preposit_time

                self.agent_busy_till[agent] = new_busy_time

    def CalcAgentTravelTime(self, agent, from_id, to_id) -> float:
        """
        Calculate the time required by an agent from traveling from point A to
        point B.

        Parameters:

        ----------
        agent : str
            Name of the agent we're calculating the travel time for.
        from_id : int
            The ID of the origin
        to_id : int
            The ID of the target location

        ----------
        Returns travel time.

        """
        if agent != 'vt':
            _, _, c_origin = self.shelf_rfc[from_id]
            _, _, c_target = self.shelf_rfc[to_id]
            return abs(c_origin - c_target) * self.column_travel_time
        else:
            _, f_origin, _ = self.shelf_rfc[from_id]
            _, f_target, _ = self.shelf_rfc[to_id]
            return abs(f_origin - f_target) * self.vt_floor_travel_time

    def ProcessAction(self, infeed, selected_shelf_id=None, rfc=None):
        """
        Perform an infeed/outfeed action. Used to be two functions 'Infeed' and
        'Outfeed' but this seemed neater.

        Parameters:

        ----------
        infeed : bool
            Whether the order is infeed, if not then it's outfeed.
        selected_shelf_id : int
            The ID of the shelf from which to retrieve or to store at.
        rfc : 3-tuple
            Tuple containing the row, floor and column of the selected shelf.

        ----------
        Returns None.

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
        agents = sequence.keys() if infeed else reversed(list(sequence.keys()))
        pth = (0, selected_shelf_id) if infeed else (selected_shelf_id, 0)

        # Initially set the action time (when it finishes) to be the same as the simulation time.
        action_time = self.sim_time

        # Add each agent's contribution to the total action time.
        for agent in agents:
            # Calculate the agent travel time.
            agent_travel_time = self.CalcAgentTravelTime(agent, pth[0], pth[1])

            # Update the time needed for executing the complete storage/retrieval action
            action_time = np.maximum(
                action_time, self.agent_busy_till[agent]) + agent_travel_time

            # Set the new busy time of this agent.
            self.agent_busy_till[agent] = action_time

            # Set the agent's location at the end of this action. Note that action_time here relates
            # to each agent individually, and not to the total action time. Otherwise the involved
            # agents would all be "busy" doing nothing after they finish their part in the action.
            if infeed:
                if agent == 'vt':
                    self.agent_location[agent][action_time] = f
                else:
                    self.agent_location[agent][action_time] = c
            else:
                self.agent_location[agent][action_time] = 0

        self.shelf_occupied[r, f, c] = True if infeed else False

        # Calculate the reward:
        reward = action_time - self.sim_time

        # Determine if this the episode length has been reached.
        self.action_counter += 1
        is_end = True if self.action_counter >= self.episode_length else False

        return reward, is_end

    """
    Below are some utility functions and pass-throughs.
    |-------------------------------------------------------------------------|
    """

    def Get_RTM(self, rfc=None):
        """Return the Response Time Matrix (RTM), or an entry from the RTM."""
        return self.rtm if rfc is None else self.rtm[rfc[0], rfc[1], rfc[2]]

    def PrintRTM(self):
        """Print the correctly oriented response time matrix."""
        print(np.flip(self.rtm, axis=1))

    def PrintIdMatrix(self, print_it=True):
        """Prints the warehouse with each shelf containing its shelf_id."""
        id_matrix = np.zeros(self.dims, dtype=int)
        for shelf_id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[shelf_id]
            id_matrix[r, f, c] = shelf_id
        if print_it is True:
            print(id_matrix)
        else:
            return id_matrix

    def PrintOccupancy(self):
        """Print the correctly oriented occupancy matrix."""

        print(np.flip(self.shelf_occupied, axis=1))

    def GetRandomShelfId(self, infeed=True) -> int:
        """Return a shelf ID randomly picked from the available IDs."""
        # Pick a random location you can infeed to.
        if infeed:
            rfc = self.rng.choice(np.argwhere(self.shelf_occupied == 0).tolist(), axis=0)
        # Pick a random location you can outfeed from.
        else:
            rfc = self.rng.choice(np.argwhere(self.shelf_occupied == 1).tolist(), axis=0)
        return int(self.shelf_id[rfc[0], rfc[1], rfc[2]])

    def GetGreedyShelfId(self, infeed=True) -> int:
        """Return a shelf ID that has the lowest response time."""
        candidate_ids = self.GetIds(want_free_ids=infeed)

        # print(candidate_ids)
        min_rt = float('inf')
        shelf_id = None

        for c_id in candidate_ids:
            (r, f, c) = self.shelf_rfc[c_id]
            if self.rtm[r, f, c] >= min_rt:
                continue
            else:
                min_rt = self.rtm[r, f, c]
                shelf_id = c_id
        return shelf_id

    def GetEpsGreedyShelfId(self, infeed=True, eps=0.1) -> int:
        """Return a shelf ID that has the lowest response time with prob. (1-eps), else return a
        random ID."""
        candidate_ids = self.GetIds(want_free_ids=infeed)

        if np.random.uniform() <= eps:
            return np.random.choice(candidate_ids)
        else:
            return self.GetGreedyShelfId(infeed)

    def GetIds(self, want_free_ids=True) -> list:
        """Get the IDs of shelves, occupied or free."""
        occ_ids = []
        free_ids = []
        for _id in range(self.num_locs):
            (r, f, c) = self.shelf_rfc[_id]
            if self.shelf_occupied[r, f, c]:
                occ_ids.append(_id)
            else:
                free_ids.append(_id)
        return free_ids if want_free_ids else occ_ids

    def SetRandomOccupancy(self, fill_percentage=0.5):
        """Randomly sets a percentage of shelves to either occupied or not."""
        random_bools = (np.random.rand(self.num_locs) < fill_percentage).astype(bool)
        random_array = np.reshape(np.asarray(random_bools, dtype=bool),
                                  (self.dims[0], self.dims[1], self.dims[2]))
        self.shelf_occupied = random_array

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

        return infeed_density, outfeed_density


def main():
    test_wh = Warehouse()
    item_id_a = 0
    item_id_b = 12

    test_wh.CalcRTM()
    test_wh.PrintRTM()
    print(test_wh.agent_location)

    # Infeed an item
    test_wh.ProcessAction(infeed=True, selected_shelf_id=item_id_a)
    # Calculate the new response time matrix given the new infeed order
    test_wh.CalcRTM()
    # Print it
    test_wh.PrintRTM()
    # Progress simulation time
    # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_a])
    test_wh.sim_time += 1
    print(test_wh.agent_location)

    test_wh.ProcessAction(infeed=True, selected_shelf_id=item_id_b)
    test_wh.CalcRTM()
    test_wh.PrintRTM()
    # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_b])
    # print(test_wh.shelf_rfc[item_id_b])
    test_wh.sim_time += 5.4
    test_wh.CalcRTM()
    test_wh.PrintRTM()
    print(test_wh.agent_location)
    print(test_wh.shelf_occupied)
    # print(test_wh.GetGreedyShelfId(infeed=False))
    print(test_wh.GetRandomShelfId())


if __name__ == '__main__':
    main()
    main()
