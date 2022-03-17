# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:22:13 2022

@author: casgi
"""
class Warehouse:
  """
    
  """
  def __init__(self):
    self.shelves = []                     # list(n, Shelf): list of shelves in WH
    self.shuttles = []                    # list(n, Shuttle): list of shuttles in WH
    self.satellites = []                  # list(n, Satellite): list of satellites in WH
    self.storedPallets = []               # list(n, Pallet): list of pallets stored in WH
    self.inTransitPallets = []            # list(n, Pallet): list of pallets in transit


    self.nrShelves = len(self.shelves)                    # int: shelf count
    self.nrShuttles = len(self.shuttles)                  # int: shuttle count
    self.nrSatellites = self.nrShuttles                   # int: satellite count
    self.nrStoredPallets = len(self.storedPallets)        # int: stored pallet count
    self.nrInTransitPallets = len(self.inTransitPallets)  # int: in transit pallet count

class Floor:
  """
    
  """
  def __init__(self, floorId):
    self.floorId = floorId

class Shelf:
  """
    
  """
  def __init__(self, shelfId, shelfDepth, shelfLocation):
    # Initiate a shelf of depth x
    self.Id = shelfId
    self.depth = shelfDepth
    self.storage = [None] * self.depth
    self.location = shelfLocation

class Pallet:
  """
    
  """
  def __init__(self, palletId):
    # Initiate a pallet
    self.Id = palletId                  # int: pallet Id
    self.inTransit = False              # bool: is pallet being transported?
    self.isStored = not self.inTransit  # bool: opposite of being in transit
    self.onTransporter = None           # int: Transporter Id
    self.storedInShelf = None           # int: Shelf Id
    self.spotInShelf = None             # int: the spot in the shelf

    self.location = [0.0, 0.0, 0.0]       # list(3, float): pallet location in WH
    self.X = self.location[0]           # float: pallet X coordinate
    self.Y = self.location[1]           # float: pallet Y coordinate
    self.Z = self.location[2]           # float: pallet Z coordinate (height)

    self.contains = None                # str: textual description of load
    self.weight = 0.0                   # float: pallet's weight
    self.dimensions = [0.0, 0.0, 0.0]     # list(3, float): pallet's dimensions
    self.length = self.dimensions[0]    # float: pallet length
    self.width = self.dimensions[1]     # float: pallet width
    self.height = self.dimensions[2]    # float: pallet height

class Transporter:
  """

  """
  def __init__(self, transporterId):
    self.Id = transporterId
    self.transporterType = None
    
    self.location = [0.0, 0.0, 0.0]
    self.X = self.location[0]           # float: 
    self.Y = self.location[1]           # float: 
    self.Z = self.location[2]

    self.carryingId = None
    self.busy = False
    self.busyTime = 0.0

    self.currentTaskId = None
    self.currentActionId = None

class Shuttle(Transporter):
  """
    Description
  """
  def __init__(self, transporterId):
    super().__init__(transporterId)
    return

class Satellite(Transporter):
  """
    Description
  """
  def __init__(self, transporterId):
    super().__init__(transporterId)
    return

class VerticalTransport(Transporter):
  """
    Description
  """
  def __init__(self, transporterId):
    super().__init__(transporterId)
    return

class WarehouseOrder:
  """
    Single or multiple ProductOrders; batch order.
  """
  def __init__(self, warehouseOrderId) -> None:
    self.Id = warehouseOrderId
    self.description = ""
    self.productOrderIds = []
    self.involvedPalletIds = []

class ProductOrder:
  """ 
    Sequence of actions with a start time and projected end time
  """
  def __init__(self, productOrderId) -> None:
    self.Id = productOrderId
    self.description = ""
    self.involvedTransporterIds = []
    self.involvedPalletIds = []
    self.involvedShelfIds = []

    self.actions = []                 # list(n, Action): list of actions
    self.startingAction = self.actions[0]
    self.finishingAction = self.actions[len(self.actions)]
    self.startingTime = 0.0
    self.finishingTime = 0.0
    self.timeTaken = 0.0

class Action:
  """
    Description
  """
  def __init__(self, actionId) -> None:
    self.Id = actionId
    self.description = ""
    self.startingTime = 0.0
    self.finishingTime = 0.0
    self.timeTaken = 0.0

  

  
  def execute(self):
    # Execute this action
    return