# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:24:20 2022

@author: casgi
"""

import numpy as np
import random
from queue import Queue


class OrderSystem():
    """

    """

    def __init__(self):

        # Order queues (FIFO).
        self.infeed_queue = Queue()
        self.outfeed_queue = Queue()

        # Dictionary for logging orders. Key: 'order_id' (int), val: dict
        self.order_register = {}
        self.order_count = 0

    def Reset(self):
        """Reset the order system for the next training episode."""
        self.infeed_queue.empty()
        self.outfeed_queue.empty()
        self.order_register.clear()
        self.order_count = 0

    # def generate_new_order_schedule(self, episode_length, max_rt=15.0):

    def GenerateNewOrder(self, **kwargs):
        # Before generating a new order, update the warehouse's sim_time!

        # Create preliminary order_id for potential troubleshooting.
        order_id = self.order_count + 1

        # Perform some checks.
        if "order_type" not in kwargs:
            raise Exception(f"No order_type specified for order {order_id}")
        if kwargs["order_type"] not in ["infeed", "outfeed", "random"]:
            raise Exception(f"""The specified order_type {kwargs["order_type"]}
                            for order {order_id} is invalid""")
        if "item_type" not in kwargs:
            raise Exception(f"No item_type specified for order {order_id}")

        # Green light: order will be created.
        self.order_count += 1

        # Get keyword arguments.
        if kwargs["order_type"] == "random":
            order_type = "infeed" if bool(random.getrandbits(1)) else "outfeed"
        else:
            order_type = kwargs["order_type"]
        item_type = kwargs["item_type"]
        current_time = kwargs["current_time"]

        # Create the order
        order = {"order_type": order_type,
                 "item_type": item_type,
                 "time_created": current_time,
                 "in_queue": False,
                 "time_in_queue": None,
                 "time_start": None,
                 "time_finish": None,
                 "shelf_id": None}

        # Add the newly created order to the register.
        self.order_register[order_id] = order

        # Queue the order.
        self.QueueOrder(order_id)

        # Finish.
        # print(f"Order {order_id} ({order_type}, item type {item_type}) generated and queued.")

    def QueueOrder(self, order_id):
        """Queue an order that has just been generated."""
        order = self.order_register[order_id]

        if order["order_type"] == "infeed":
            self.infeed_queue.put(order_id)
        elif order["order_type"] == "outfeed":
            self.outfeed_queue.put(order_id)

        order["in_queue"] = True
        order["time_in_queue"] = 0.0

    def GetNextOrder(self, current_time, free_and_occ, prioritize_outfeed=True):
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
        infeed_queue_empty = False if self.infeed_queue.qsize() > 0 else True
        outfeed_queue_empty = False if self.outfeed_queue.qsize() > 0 else True

        # Catch the case where both queues are empty.
        if infeed_queue_empty and outfeed_queue_empty:
            raise RuntimeError("""Both queues are empty, can't get next order.""")

        # Catch the cases where you want to infeed into a full warehouse or outfeed from an
        # empty warehouse.
        if infeed_queue_empty and free_and_occ[1] == 0:
            raise RuntimeError(f"Trying to outfeed from an empty warehouse! Time: {current_time}")
        elif outfeed_queue_empty and free_and_occ[0] == 0:
            raise RuntimeError(f"Trying to infeed to a full warehouse! Time: {current_time}")

        # If there's an outfeed prioritization (to prevent saturated warehouse)
        if prioritize_outfeed:
            if not outfeed_queue_empty:
                next_order_id = self.outfeed_queue.get()
            else:
                # Don't need to check if infeed queue is empty, that's already done above.
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

        self.order_register[next_order_id]["time_in_queue"] = (
            current_time - self.order_register[next_order_id]["time_created"])
        self.order_register[next_order_id]["in_queue"] = False
        self.order_register[next_order_id]["time_start"] = current_time

        return next_order_id, self.order_register[next_order_id]

    def FinishOrder(self, order_id, shelf_id, finish_time):
        """Complete order, log time and set shelf_id (set None if outfeed)."""
        order = self.order_register[order_id]
        order["time_finish"] = finish_time
        order["shelf_id"] = shelf_id

    def GetShelfAccessCounts(self) -> dict:
        """Calculates and returns a dictionary filled with access counts for each shelf_id."""
        infeed_counts = {}
        outfeed_counts = {}

        # For each order in the registry
        for order_id in self.order_register.keys():
            # Get the shelf ID that was accessed (either for infeed or outfeed).
            shelf_id = self.order_register[order_id]['shelf_id']
            if self.order_register[order_id]['order_type'] == 'infeed':
                # Try to increase the counter of that shelf.
                try:
                    infeed_counts[shelf_id] += 1
                # If we haven't seen this shelf yet, add it, set it's value to 1.
                except KeyError:
                    infeed_counts[shelf_id] = 1
            elif self.order_register[order_id]['order_type'] == 'outfeed':
                try:
                    outfeed_counts[shelf_id] += 1
                except KeyError:
                    outfeed_counts[shelf_id] = 1
            # This shouldn't happen, only other order_type is 'random' but that can't occur.
            else:
                raise Exception(f"""Order type ({self.order_register[order_id]['order_type']})
                                \rwasn't expected!""")

        return infeed_counts, outfeed_counts

    def PrintOrderQueues(self):
        for i in list(self.infeed_queue.queue):
            print(self.order_register[i])
        for i in list(self.outfeed_queue.queue):
            print(self.order_register[i])

    def PrintOrderRegister(self):
        for key, values in self.order_register.items():
            print(f"Order ID: {key}")
            for value in values.keys():
                print(f"    {value}: {values[value]}")
            print("\n")


def main():
    """Just some tests to see if the above methods are working."""
    sim_time = 0.0
    test_os = OrderSystem()
    sim_time += 3
    test_os.GenerateNewOrder(current_time=sim_time, order_type="infeed", item_type=1)
    for i in range(1, 6, 1):
        sim_time += 3
        test_os.GenerateNewOrder(current_time=sim_time, order_type="infeed", item_type=1)
    test_os.PrintOrderQueues()
    next_executed_order = test_os.ExecuteNextOrder(current_time=sim_time)
    print("\n")
    print(test_os.order_register[next_executed_order])
    print("\n")
    test_os.PrintOrderQueues()
    test_os.FinishOrder(order_id=next_executed_order, shelf_id=12, current_time=sim_time)
    test_os.PrintOrderRegister()


if __name__ == '__main__':
    main()
