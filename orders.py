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

    def reset(self):
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
        if kwargs["order_type"] != ("infeed" or "outfeed" or "random"):
            raise Exception(f"""The specified order_type {kwargs["order_type"]}
                            for order {order_id} is invalid""")
        if "item_type" not in kwargs:
            raise Exception(f"No item_type specified for order {order_id}")

        # Green light: order will be created.
        self.order_count += 1

        # Get keyword arguments.
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
        print(f"Order {order_id} ({order_type}, item type {item_type}) generated and queued.")

    def QueueOrder(self, order_id):
        order = self.order_register[order_id]

        if order["order_type"] == "infeed":
            self.infeed_queue.put(order_id)
        elif order["order_type"] == "outfeed":
            self.outfeed_queue.put(order_id)

        order["in_queue"] = True
        order["time_in_queue"] = 0.0

    def get_next_order(self, prioritize_outfeed=True):
        """Get the next order from one of the queues"""
        # Check if queues are empty.
        infeed_queue_empty = False if self.infeed_queue.qsize() > 0 else True
        outfeed_queue_empty = False if self.outfeed_queue.qsize() > 0 else True

        # Catch the case where both queues are empty.
        if infeed_queue_empty and outfeed_queue_empty:
            raise Exception("""Both queues are empty.""")

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

        
        # self.order_register[next_order_id]["time_in_queue"] = current_time - self.order_register[next_order_id]["time_created"]
        # self.order_register[next_order_id]["in_queue"] = False
        # self.order_register[next_order_id]["time_start"] = current_time

        return next_order_id

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

        
        self.order_register[next_order_id]["time_in_queue"] = current_time - self.order_register[next_order_id]["time_created"]
        self.order_register[next_order_id]["in_queue"] = False
        self.order_register[next_order_id]["time_start"] = current_time

        return next_order_id

    def FinishOrder(self, order_id, shelf_id, current_time):
        """Complete order, log time and set shelf_id (set None if outfeed)."""
        order = self.order_register[order_id]
        order["time_finish"] = current_time
        order["shelf_id"] = shelf_id

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








