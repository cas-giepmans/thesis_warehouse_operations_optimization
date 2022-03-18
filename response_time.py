# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:44 2022.

@author: casgi
"""
import numpy as np
import random


class Warehouse():
    """


    Return.

    -------
    None.

    """

    def __init__(self, num_rows=2, num_floors=6, num_cols=8):
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

        # dictionary to store shelf coordinates, accessed through shelf_id.
        self.shelf_rfc = {}  # Key: shelf_id[r, c, f], val: (r, c, f)

        # Numpy array to keep track of shelf IDs
        self.shelf_id = np.zeros(self.dims)

        # Variables about time
        self.T = 4.0
        self.sim_time = 0.0
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
        i = 0  # shelf id

        for c in range(self.num_cols):
            for f in range(self.num_floors):
                for r in range(self.num_rows):
                    self.shelf_access_sequence[i] = {
                        'vt': f * self.vt_floor_travel_time,
                        'sh'+str(f): (1 + c) * self.column_travel_time}

                    self.shelf_id[r, f, c] = i
                    self.shelf_rfc[i] = (r, f, c)

                    i += 1

        # print(self.shelf_access_sequence)

    # def CalcRTM(self, current_time):
    def CalcRTM(self):
        """


        Return None.

        -------
        None.

        """
        # TODO: fix this function, see if Kutalmis' stuff is useful here or somewhere else.
        for i in range(0, self.num_locs):
            (r, f, c) = self.shelf_rfc[i]
            sequence = self.shelf_access_sequence[i]

            self.rtm[r, f, c] = 0

            if ~self.shelf_occupied[r, f, c]:
                # now = current_time
                for agent in sequence.keys():
                #     start_time = np.maximum(now, self.agent_busy_till[s])
                #     end_time = start_time + sequence[s]
                #     # if we take this action: self.agent_usy_till[s] = end_time
                #     now = end_time
                # rtm = now - current_time
                    agent_time_remaining = np.maximum(
                        0,
                        (self.agent_busy_till[agent] - self.sim_time))
                    self.rtm[r, f, c] += (agent_time_remaining +
                                          sequence[agent])
                    # self.rtm[r, c, f] += np.maximum(
                    #     self.T,
                    #     self.agent_busy_till[s]) + sequence[s]

        # Here rtm is ready

    # def Take_Action(self, shelf_id, curent_time):
    #     (r, f, c) = self.shelf_rfc[shelf_id]
    #         sequence = self.shelf_access_sequence[i]

    #             now = current_time
    #             for s in sequence.keys():
    #                 start_time = np.maximum(now, self.agent_busy_till[s])
    #                 end_time = start_time + sequence[s]
    #                  self.agent_usy_till[s] = end_time

    def Get_RTM(self, rfc=None):
        """Return the response time matrix."""
        return self.rtm if rfc is None else self.rtm[rfc[0], rfc[1], rfc[2]]

    def PrintRTM(self):
        """Print the correctly oriented response time matrix."""
        print(np.flip(self.rtm, axis=1))

    def ReadyTransporters(self, selected_shelf_id, infeed=True):
        """

        Parameters.

        ----------
        selected_shelf_id : int
            ID of the shelf to be accessed.
        infeed : bool, optional
            Whether the coming order is infeed or outfeed. The default is True.


        Return None.

        -------
        Given the ID of a shelf to be accessed and the type of operation, move
        every involved agent to the required starting location and update their
        busy times accordingly.

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


        Return travel time.

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

    def ExecuteOrder(self, infeed, selected_shelf_id=None, rfc=None):
        """

        Parameters.

        ----------
        infeed : bool
            Whether the order is infeed, if not then it's outfeed.
        selected_shelf_id : int
            The ID of the shelf from which to retrieve or to store at.
        rfc : 3-tuple
            Tuple containing the row, floor and column of the selected shelf.


        Return None.

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


def main():
    test_wh = Warehouse()
    item_id_a = 0
    item_id_b = 12

    test_wh.CalcRTM()
    test_wh.PrintRTM()
    print(test_wh.agent_location)

    # Infeed an item
    test_wh.Infeed(item_id_a)
    # Calculate the new response time matrix given the new infeed order
    test_wh.CalcRTM()
    # Print it
    test_wh.PrintRTM()
    # Progress simulation time
    # test_wh.sim_time += test_wh.Get_RTM(test_wh.shelf_rfc[item_id_a])
    test_wh.sim_time += 1
    print(test_wh.agent_location)

    test_wh.Infeed(item_id_b)
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
