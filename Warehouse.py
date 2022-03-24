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
            num_hist_rtms=1):
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

        # dictionary to store shelf coordinates, accessed through shelf_id.
        self.shelf_rfc = {}  # Key: shelf_id[r, c, f], val: (r, c, f)

        # Numpy array to keep track of shelf IDs
        self.shelf_id = np.zeros(self.dims)

        # Variables about time
        self.episode_length = episode_length # Nr. of orders per episode.
        self.num_historical_rtms = num_hist_rtms
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

        # Compute time needed to reach the shelf from infeed point 
        # (assumes agents are at their default position)
        i = 0 # Initiate shelf ID counter.

        for c in range(self.num_cols):
            for f in range(self.num_floors):
                for r in range(self.num_rows):
                    self.shelf_access_sequence[i] = {
                        'vt': f * self.vt_floor_travel_time,
                        'sh'+str(f): (1 + c) * self.column_travel_time}

                    self.shelf_id[r, f, c] = i
                    self.shelf_rfc[i] = (r, f, c)

                    i += 1

    def ResetState(self):
        """Initiate the state for a new training episode."""
        self.shelf_occupied = np.zeros(self.dims, dtype=bool)
        # self.SetRandomOccupancy()  # Randomize occupancy(?)
        self.last_episode_time = self.sim_time
        self.sim_time = 0.0
        self.action_counter = 0

        self.agent_busy_till = {'vt': 0}

        for f in range(self.num_floors):
            self.agent_busy_till['sh'+str(f)] = 0

        self.agent_location = {'vt': {0.0: 0}}
        for f in range(self.num_floors):
            self.agent_location['sh'+str(f)] = {0.0: 0}

        self.order_system.reset()

        fresh_wh_state = self.GetState()

        return fresh_wh_state

    # CHECK: is this one really necessary?
    def GetState(self, infeed=True):
        """Return the newly built warehouse state."""
        return self.BuildState()

    # CHECK: don't you need occupancy history?
    def BuildState(self, infeed=True):
        """Build the new warehouse state, consisting of n historical RTMs, the
        current RTM and the binary matrix representing pickable locations,
        either for storage or for retrieval ops."""
        wh_state = []
        # Add historical RTMs, reshape them to fit Lei Luo's network.
        # TODO: deal with the reshaping for different warehouse architectures.
        for hrtm in self.rtm_history:
            rtm = np.reshape(hrtm, (self.dims[0] * self.dims[1], self.dims[2]))
            wh_state.append(rtm)
        # Add the current RTM.
        wh_state.append(self.rtm)
        # Add the occupancy matrix.
        if infeed is True:
            wh_state.append(self.shelf_occupied)
        else:
            wh_state.append(~self.shelf_occupied)
        return wh_state

    def do_action(self, action):
        # CHECK comments pls
        self.action_counter += 1
        is_end = True if self.action_counter == self.episode_length else False
        
        return self.rtm(action), reward, is_end
        
    def CalcShelfAccessTime(self, shelf_id, infeed=True):

            # Get the row, floor and column coordinates for shelf i.
            (r, f, c) = self.shelf_rfc[shelf_id]

            # Get the shelf access sequence for given shelf, e.g. the agents involved in accessing this shelf. 
            sequence = self.shelf_access_sequence[shelf_id]

            # The sequence must be reversed for outfeed response time calculation.
            agents = sequence.keys() if infeed else reversed(sequence.keys())
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
                busy_till += self.CalcAgentTravelTime(agent,self.shelf_id[0, agent_loc, 0],pth[0])
                
                # Pick-up item and go to srequired location, the agents who will get the item will wait for you
                curr_time = np.maximum(busy_till, curr_time) + self.CalcAgentTravelTime(agent,pth[0],pth[1])

            return curr_time - self.sim_time

    def CalcRTM(self):
        """
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
            # prepositioning time will be already be done with prepositioning itself.
            max_prep_time = 0.0

            # In case the considered location (r, f, c) is not occupied, i.e. a possible infeed
            # location, we calculate the time required for infeeding an item to this shelf,
            # including the prepositioning of the agents to their default infeed location (0, 0, 0).
            if ~self.shelf_occupied[r, f, c]:
                for agent in sequence.keys():
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
                for agent in reversed(sequence.keys()):
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

    def Get_RTM(self, rfc=None):
        """Return the Response Time Matrix (RTM), or an entry from the RTM."""
        return self.rtm if rfc is None else self.rtm[rfc[0], rfc[1], rfc[2]]

    def PrintRTM(self):
        """Print the correctly oriented response time matrix."""
        print(np.flip(self.rtm, axis=1))

    def ReadyTransporters(self, selected_shelf_id, infeed=True):
        """

        Given the ID of a shelf to be accessed and the type of operation, move
        every involved agent to the required starting location and update their
        busy times accordingly.

        Parameters.

        ----------
        selected_shelf_id : int
            ID of the shelf to be accessed.
        infeed : bool, optional
            Whether the coming order is infeed or outfeed. The default is True.


        Return None.

        -------
        

        """
        # Get all the agents involved in accessing the selected shelf.
        sequence = self.shelf_access_sequence[selected_shelf_id]

        # If infeed: move all involved agents to position 0.
        if infeed:
            for agent in sequence.keys():
                # Calculate time necessary for prepositioning agent
                agent_preposit_time = self.CalcAgentTravelTime(
                    agent,
                    self.agent_location[agent][self.agent_busy_till[agent]],
                    0)

                # Calculate when agent is done with prepositioning
                new_busy_time = np.maximum(
                    self.sim_time,
                    self.agent_busy_till[agent]) + agent_preposit_time

                # Update agent's busy time
                self.agent_busy_till[agent] = new_busy_time
        # If outfeed: move all involved agents to the selected location
        else:
            for agent in sequence.keys():
                agent_preposit_time = self.CalcAgentTravelTime(
                    agent,
                    self.agent_location[agent][self.agent_busy_till[agent]],
                    selected_shelf_id)

                new_busy_time = np.maximum(
                    self.sim_time,
                    self.agent_busy_till[agent]) + agent_preposit_time

                self.agent_busy_till[agent] = new_busy_time

    def CalcAgentTravelTime(self, agent, from_id, to_id) -> float:
        """

        Parameters.

        ----------
        agent : str
            Name of the agent we're calculating the travel time for.
        from_id : int
            The ID of the origin
        to_id : int
            The ID of the target location


        Returns travel time.

        -------
        Calculate the time required by an agent from traveling from point A to
        point B.

        """
        if agent != 'vt':
            _, _, c_origin = self.shelf_rfc[from_id]
            _, _, c_target = self.shelf_rfc[to_id]
            return abs(c_origin - c_target) * self.column_travel_time
        else:
            _, f_origin, _ = self.shelf_rfc[from_id]
            _, f_target, _ = self.shelf_rfc[to_id]
            return abs(f_origin - f_target) * self.vt_floor_travel_time

    def ProcessOrder(self, infeed, selected_shelf_id=None, rfc=None):
        """

        Parameters.

        ----------
        infeed : bool
            Whether the order is infeed, if not then it's outfeed.
        selected_shelf_id : int
            The ID of the shelf from which to retrieve or to store at.
        rfc : 3-tuple
            Tuple containing the row, floor and column of the selected shelf.


        Returns None.

        -------
        Perform an infeed/outfeed order. Used to be two functions 'Infeed' and
        'Outfeed' but this seemed neater.

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

        # Check whether there are any available spots
        if infeed and not np.any(~self.shelf_occupied):
            raise Exception("The warehouse is fully filled, can't infeed!")
        if not infeed and not self.shelf_occupied.any():
            raise Exception("Warehouse is empty, no items to outfeed!")
            

        self.ReadyTransporters(selected_shelf_id, infeed)

        if infeed:
            sequence = self.shelf_access_sequence[selected_shelf_id]
        else:
            sequence = reversed(self.shelf_access_sequence[selected_shelf_id])

        # TODO: figure out how to update the sim_time in the meantime...
        # TODO: update the sim time once a new order comes in.
        action_time = self.sim_time
        for agent in sequence.keys():
            if infeed:
                agent_travel_time = self.CalcAgentTravelTime(
                    agent,
                    0,  # Can use 0 here, since its rfc is (0, 0, 0)
                    selected_shelf_id)
            else:
                agent_travel_time = self.CalcAgentTravelTime(
                    agent,
                    selected_shelf_id,
                    0)
            action_time = np.maximum(
                action_time, self.agent_busy_till[agent]) + agent_travel_time

            self.agent_busy_till[agent] = action_time
            if infeed:
                if agent != 'vt':
                    self.agent_location[agent][action_time] = c
                else:
                    self.agent_location[agent][action_time] = f
            else:
                self.agent_location[agent][action_time] = 0

        self.shelf_occupied[r, f, c] = True if infeed else False

    def Infeed(self, selected_shelf_id=None, rfc=None):
        """


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

    """
    Below are some utility functions and pass-throughs.
    |-------------------------------------------------------------------------|
    """

    def GetRandomShelfId(self, occupied=True):
        """Return a shelf ID randomly picked from the available IDs."""
        # Pick an occupied shelf's ID.
        if occupied:
            [r, f, c] = np.random.choice(
                np.where(self.shelf_occupied == 1)).tolist()
        # Pick an unoccupied shelf's ID.
        else:
            [r, f, c] = np.random.choice(
                np.where(self.shelf_occupied == 0)).tolist()
        return self.shelf_id[r, f, c]

    def SetRandomOccupancy(self):
        """Randomly sets every shelf to either True or False."""
        random_bools = (np.random.rand(self.num_locs) > 0.5).astype(bool)
        random_array = np.reshape(
            np.asarray(random_bools, dtype=bool),
            (self.dims[0], self.dims[1], self.dims[2]))
        self.shelf_occupied = random_array

    # def get_next_order_id(self, prioritize_outfeed=True):
    #     """Get an order from the order system."""
    #     return self.order_system.get_next_order(prioritize_outfeed)


def main():
    test_wh = Warehouse()
    item_id_a = 0
    item_id_b = 12

    test_wh.CalcRTM()
    test_wh.PrintRTM()
    print(test_wh.agent_location)

    # Infeed an item
    test_wh.ProcessOrder(infeed=True, selected_shelf_id=item_id_a)
    # Calculate the new response time matrix given the new infeed order
    test_wh.CalcRTM()
    # Print it
    test_wh.PrintRTM()
    # Progress simulation time
    # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_a])
    test_wh.sim_time += 1
    print(test_wh.agent_location)

    test_wh.ProcessOrder(infeed=True, selected_shelf_id=item_id_b)
    test_wh.CalcRTM()
    test_wh.PrintRTM()
    # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_b])
    # print(test_wh.shelf_rfc[item_id_b])
    test_wh.sim_time += 5.4
    test_wh.CalcRTM()
    test_wh.PrintRTM()
    print(test_wh.agent_location)


if __name__ == '__main__':
    main()
