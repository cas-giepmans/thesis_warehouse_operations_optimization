# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:24:20 2022

@author: casgi
"""

import numpy as np
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

        


    def GenerateNewOrder(self, **kwargs):
        # Before generating a new order, update the warehouse's sim_time!

        # Create preliminary order_id for potential troubleshooting.
        order_id = self.order_count + 1

        # Perform some checks.
        if "order_type" not in kwargs:
            raise Exception(f"No order_type specified for order {order_id}")
        if kwargs["order_type"] != ("infeed" or "outfeed"):
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
                 "queued": False,
                 "time_in_queue": None,
                 "time_start": None,
                 "time_finish": None,
                 "shelf_id": None}

        # Add the newly created order to the register.
        self.order_register[order_id] = order

        # Queue the order.
        self.QueueOrder(order_id)

    def QueueOrder(self, order_id):
        order = self.order_register[order_id]

        if order["order_type"] == "infeed":
            self.infeed_queue.put(order_id)
        elif order["order_type"] == "outfeed":
            self.outfeed_queue.put(order_id)

        order["queued"] = True
        order["time_in_queue"] = 0.0

    def ExecuteNextOrder(self, current_time, prioritize_outfeed=True):
        # Before executing an order, update the warehouse's sim_time!

        infeed_queue_empty = False if self.infeed_queue.qsize() > 0 else True
        outfeed_queue_empty = False if self.outfeed_queue.qsize() > 0 else True

        if prioritize_outfeed and not outfeed_queue_empty:
            next_order_id = self.outfeed_queue.get()
        elif prioritize_outfeed and outfeed_queue_empty:
            if not infeed_queue_empty:
                next_order_id = self.infeed_queue.get()
            else:
                # TODO: no orders to execute, wat do?
        else:
            
            next_order_id
        next_order = self.order_register[self.]
        
        
        












