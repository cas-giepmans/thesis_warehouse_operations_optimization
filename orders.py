# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:24:20 2022

@author: Cas Giepmans

Copyright (C) <year>  <name of author>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import random
from queue import Queue


class OrderSystem():
    """

    """

    def __init__(self):

        # Order queues (FIFO)". They get filled with order IDs.
        self.infeed_queue = Queue()
        self.outfeed_queue = Queue()

        # Dictionary for logging orders. Key: 'order_id' (int), val: dict
        self.order_register = {}
        self.order_count = 0

        # RNG for generating orders.
        self.rng = np.random.default_rng()

    def Reset(self):
        """Reset the order system for the next training episode."""
        self.infeed_queue.empty()
        self.outfeed_queue.empty()
        self.order_register.clear()
        self.order_count = 0

    def GenerateNewOrder(self, current_time, order_type, free_and_occ, product_type=1):

        # Create preliminary order_id for potential troubleshooting.
        order_id = self.order_count + 1

        # if order_type == "infeed" and free_and_occ[0] == 0:
        #     raise Exception("Can't infeed to a full warehouse!")
        # if order_type == "outfeed" and free_and_occ[1] == 0:
        #     raise Exception("Can't outfeed from an empty warehouse!")

        # Green light: order will be created.
        self.order_count += 1

        # Check whether the current warehouse state requires a specific action
        if order_type == "random":
            if free_and_occ[0] == 0:
                order_type = "outfeed"
            elif free_and_occ[1] == 0:
                order_type = "infeed"
            else:
                order_type = "infeed" if bool(random.getrandbits(1)) else "outfeed"

        # Create the order.
        order = {"order_type": order_type,
                 "product_type": product_type,
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
        # print(f"Order {order_id} ({order_type}, item type {product_type}) generated and queued.")

    def QueueOrder(self, order_id):
        """Queue an order that has just been generated."""
        order = self.order_register[order_id]

        if order["order_type"] == "infeed":
            self.infeed_queue.put(order_id)
        elif order["order_type"] == "outfeed":
            self.outfeed_queue.put(order_id)

        self.order_register[order_id]["in_queue"] = True
        self.order_register[order_id]["time_in_queue"] = 0.0

    def GetNextOrder(self, current_time, free_and_occ, init_fill_perc=0.5):
        """Rewrite of the original method of the same name. IMPORTANT: if init_fill_perc is 1.0 or
           0.0, immediately get an outfeed or infeed order, respectively. This is the only legal
           behavior, as you can't reliably simulate around a full or empty warehouse!"""
        n_free = free_and_occ[0]  # Number of free shelves
        n_occ = free_and_occ[1]  # Number of occupied shelves
        curr_fill_perc = float(n_occ / (n_free + n_occ))  # How full the warehouse is at this time
        in_Q_empty = False if self.infeed_queue.qsize() > 0 else True  # Check for infeed orders
        out_Q_empty = False if self.outfeed_queue.qsize() > 0 else True  # Check for outfeed orders
        # print(f"We stop at 1")
        if in_Q_empty and out_Q_empty:
            raise RuntimeError("""Both queues are empty, can't get next order.""")
        elif in_Q_empty and not out_Q_empty:
            next_order_id = self.outfeed_queue.get()
            return next_order_id, self.order_register[next_order_id]
        elif out_Q_empty and not in_Q_empty:
            next_order_id = self.infeed_queue.get()
            return next_order_id, self.order_register[next_order_id]

        # Handle the case where we start from a full or empty warehouse.
        if init_fill_perc in [0.0, 1.0]:
            # Interpret the float value of init_fill_perc as a boolean.
            next_order_id = self.outfeed_queue.get() if init_fill_perc else self.infeed_queue.get()
            return next_order_id, self.order_register[next_order_id]
        # print(f"We stop at 2")
        if (n_free == 0 and out_Q_empty) or (n_occ == 0 and in_Q_empty):
            raise RuntimeError("Warehouse is full/empty but there is no outfeed/infeed order!")

        # print(f"We stop at 3")
        # Here we know for sure that we can outfeed in case the warehouse is full, or infeed in case
        # the warehouse is empty. So do that to prevent illegal situations.
        if n_free == 0:
            next_order_id = self.outfeed_queue.get()
            return next_order_id, self.order_register[next_order_id]
        elif n_occ == 0:
            next_order_id = self.infeed_queue.get()
            return next_order_id, self.order_register[next_order_id]
        else:
            # Green light to draw from a random variable, without risking illegal behavior.
            # FIXME: I think something is going wrong here, RunBenchmark gets stuck somewhere.
            # decider = self.rng.uniform()
            # print(f"Get random order from queues.")
            # print(
            #     f"Queue sizes (in and out): {self.infeed_queue.qsize()}, {self.outfeed_queue.qsize()}")
            decider = bool(random.getrandbits(1))
            # prob_outf = curr_fill_perc**(init_fill_perc / (1 - init_fill_perc))
            # if decider <= prob_outf:
            if decider:
                # Get an outfeed order from the queue.
                next_order_id = self.outfeed_queue.get()
                return next_order_id, self.order_register[next_order_id]
            else:
                # Get an infeed order from the queue.
                next_order_id = self.infeed_queue.get()
                return next_order_id, self.order_register[next_order_id]

    def FinishOrder(self, order_id, shelf_id, finish_time):
        """Complete order, log time and set shelf_id (either where the item was stored or from where
           it got retrieved)."""
        order = self.order_register[order_id]
        order["time_finish"] = finish_time
        order["shelf_id"] = shelf_id

    def GetShelfAccessCounts(self) -> dict:
        """Calculates and returns two dictionaries (one for infeed, one for outfeed) filled with
           access counts for each shelf_id."""
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
                # If we haven't seen this shelf yet, add it, then set it's value to 1.
                except KeyError:
                    infeed_counts[shelf_id] = 1
            elif self.order_register[order_id]['order_type'] == 'outfeed':
                # Do the same thing as above, but then for outfeed orders.
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
        """Print all the orders that are currently queued."""
        for i in list(self.infeed_queue.queue):
            print(self.order_register[i])
        for i in list(self.outfeed_queue.queue):
            print(self.order_register[i])

    def PrintOrderRegister(self):
        """Print the order register, i.e. all orders generated since simulation start."""
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
    test_os.GenerateNewOrder(current_time=sim_time, order_type="infeed", product_type=1)
    for i in range(1, 6, 1):
        sim_time += 3
        test_os.GenerateNewOrder(current_time=sim_time, order_type="infeed", product_type=1)
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
