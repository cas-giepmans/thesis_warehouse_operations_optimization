import socket
from TrainingGame import TrainingGame
from Warehouse import Warehouse
from orders import OrderSystem
import copy
import numpy as np

"""
    This plantGame is based on PlantGame_AVSRS.py from the original files 
    provided by Lei Luo. In this version, I've attempted to disconnect the 
    Siemens Plant Simulation elements (server elements) and build a warehouse 
    state representation directly here. This should make the code usable for 
    my own research, as I do not possess Siemens Plant Simulation.
    
    The main difference is that in the case of Lei Luo, the response times are
    aqcuired through the SPS. I, however, intend to use the data from one of 
    Viscon's warehouses to estimate the response time. This should be a simple 
    matter of supervised machine learning: in go details about the job, out 
    comes a prediction of the time necessary, which is then checked against the
    time it actually took.
"""


def index_str(str1, str2):
    """Finds the specified string str1 containing the full location of the specified substr2, Returns as a list"""
    length_2 = len(str2)
    length_1 = len(str1)
    index_str2 = []
    i = 0
    while str2 in str1[i:]:
        index_tmp = str1.index(str2, i, length_1)
        index_str2.append(index_tmp)
        i = (index_tmp + length_2)
    return index_str2


class PlantGame_AVSRS(TrainingGame):
    def __init__(self, xDim, yDim):
        super(PlantGame_AVSRS, self).__init__(xDim, yDim)
        self.StateNum = 4  # stateNum represents the number of states passed in, which varies from game to game

        self.last_episode_time = 0
        self.episodeTimes = []
        self.last_response_time_matrix = []
        self.this_episodes_time = 0.0
        self.local_episode_counter = 0

        print("start plant simulation")  # create object here

        # self.server = socket.socket()
        # self.server.bind(('127.0.0.5', 8520))
        # self.server.listen(5)
        # self.socketObj, self.socketAddress = self.server.accept()
        self.init_para()

    def init_para(self):
        self.AvailablePos = [1.] * self.X_dim * self.Y_dim
        self.last_episode_time = 0
        self.chooseTime = 1  # How many times the game has made a choice
        self.allAction = []
        self.local_episode_counter = 0
        self.this_episodes_time = 0.0

        self.last_response_time_matrix = [
            [0. for col in range(self.X_dim)] for row in range(self.Y_dim)]
        # self.last_response_time_matrix_list.append(self.last_response_time_matrix)
        self.last_response_time_matrix_list = []

    def get_state_from_socket(self):  # TODO: replace with non-server thingemajig
        # original_date = self.socketObj.recv(1024)
        # original_date_str = format(original_date.decode())
        # return original_date_str
        return

    def get_init_state(self):
        self.init_para()

        # original_date_str = self.get_state_from_socket()
        original_date_str = '_' + '_'.join([str(x) for x in np.round(np.random.uniform(
            3.0, 32.0, 96), 1).tolist()]) + '_' + f'{self.local_episode_counter}+1.00'
        state_time_matrix, is_end, episode_time = self.get_response_time(original_date_str)

        # self.last_episode_time = 0.0
        self.last_episode_time = episode_time
        return state_time_matrix
        # Contains three attempts: 1. Response time view, 2. Selectable location view, and 3. Last few selection views
        # Added a new view: Response time view at last selection (first view)

    def do_action(self, action):
        # send action
        # self.socketObj.send(str(action).encode())
        self.AvailablePos[action] = 0

        self.allAction.append(action)
        self.chooseTime += 1

        # # Generate list of randoms
        # listOfRandoms = np.round(np.random.uniform(3.0, 32.0, 96), 1).tolist()

        # # Add the time of the chosen action to this episode's time
        # self.this_episodes_time += listOfRandoms[action]

        # # Element-wise multiplication of randoms and available locations.
        # # This gives 0 for every occupied location, just like the simulation.
        # listOfRandomsWithoutFilledSpaces = [
        #     listOfRandoms[i] *
        #     self.AvailablePos[i] for i in range(len(listOfRandoms))]

        # """ Check for last action, increase local episode counter if satisfied
        #     this was wrong, episode counting was already being done. This
        #     actually just functions as a finish indicator."""

        # if len(self.allAction) == 96:
        #     self.local_episode_counter = 1

        # """ Create the 'tail counter', this keeps track of nr of episodes and
        #     this episode's run time."""

        # tail_counter = f'_{self.local_episode_counter}+{round(self.this_episodes_time, 2)}'

        # # Parse randoms to one string, separated by '_' and add tail counter
        # stringOfRandoms = '_' + '_'.join([
        #     str(x) for x in listOfRandomsWithoutFilledSpaces]) + tail_counter

        # # Rename
        # original_date_str = stringOfRandoms

        # recieve socket message
        # original_date_str = self.get_state_from_socket()

        # get state_matrix
        state_time_matrix, is_end, episode_time = self.get_response_time(original_date_str)

        """The partial reward is equal to the action time. Continue in
        train_step in policy_net_AC_pytorch.py."""
        # calculate reward
        reward = self.last_episode_time - episode_time  # 望大
        # reward = episode_time - self.last_episode_time # 望小
        self.last_episode_time = episode_time

        return state_time_matrix, reward, is_end

    def dolastAction(self):
        # self.socketObj.send(str(-1.).encode())
        return

    def get_response_time(self, response_time_matrix):
        """This replaces an earlier implementation by Lei Luo. It takes in an
        ndarray, maintained by the Warehouse instance."""

        return state, is_end, episode_time

    def get_response_time(self, original_date_str):
        # print("A new step")
        # print("original data from simulation:", original_date_str)
        """Converts a string to an array"""

        """Initialize the variable"""
        # print("original_date_str:", original_date_str)
        data_num = index_str(original_date_str, '_')
        response_time_matrix = []  # Original "Full Position Response Time Matrix"
        permit_place_matrix = []  # Optional location view
        last_action_matrix = []  # The last few choices
        state = []
        # Get the original Response Time Matrix to get an optional location view

        """Gets the original response matrix"""
        temp = 0
        response_time_in_row = []
        permit_place_in_row = []
        max_time_value, min_time_value = 0, 10000
        for i in range(len(data_num)-1):
            temp += 1
            """ What happens here is that he takes a piece of the original data
                string, delimited by the indices of the "_" characters.
                AKA: what is String.split()?"""
            this_response_time = float(
                original_date_str[data_num[i] + 1:data_num[i + 1]])
            if this_response_time >= max_time_value:
                max_time_value = this_response_time
                # Take the maximum and minimum values and use them as normalization
            if min_time_value >= this_response_time > 0:
                min_time_value = this_response_time  # Take the maximum and minimum values and use them as normalization

            # Constructing taboo location information (in row units)
            if this_response_time == 0.:
                permit_place_in_row.append(0.)
            else:
                permit_place_in_row.append(1.)
            response_time_in_row.append(this_response_time)

            # If you've reached the last item in this row.
            if temp == self.X_dim:   # 换行--------------------------------------------------
                response_time_matrix.append(response_time_in_row)
                permit_place_matrix.append(permit_place_in_row)
                temp = 0
                response_time_in_row = []
                permit_place_in_row = []
        # print("response_time_matrix_1:", response_time_matrix)
        # print("max_time_value:", max_time_value)
        # print("min_time_value:", min_time_value)

        """Normalize response_time_matrix"""
        # for oneYDim in range(len(response_time_matrix)):
        #     for oneXDim in range(len(response_time_matrix[0])):
        #         if response_time_matrix[oneYDim][oneXDim] == 0:
        #             # Equal to 0 itself, no action is taken
        #             pass
        #         else:
        #             if max_time_value == min_time_value:
        #                 response_time_matrix[oneYDim][oneXDim] = 1.
        #             else:
        #                 response_time_matrix[oneYDim][oneXDim] = (max_time_value - response_time_matrix[oneYDim][oneXDim]) / \
        #                                                          (max_time_value - min_time_value)
        #                 # response_time_matrix[oneYDim][oneXDim] = (response_time_matrix[oneYDim][oneXDim] - min_time_value) / \
        #                 #                                          (max_time_value - min_time_value)
        #                 # response_time_matrix[oneYDim][oneXDim] =
        #                 # response_time_matrix[oneYDim][oneXDim] / max_time_value

        # Gets an optional location view
        action_num = 20
        all_action_length = len(self.allAction)
        last_action_matrix = [[0 for col in range(self.X_dim)] for row in range(self.Y_dim)]
        if all_action_length > 0:
            for i in range(min(action_num, all_action_length)):
                my_act = self.allAction[-1 * (i + 1)]
                this_line = my_act // self.X_dim
                this_pos = my_act % self.X_dim
                # print(my_act)
                last_action_matrix[this_line][this_pos] = 1.

        lasr_rt_num = self.StateNum-3
        lasr_rt_savenum = lasr_rt_num+1
        if len(self.last_response_time_matrix_list) < lasr_rt_savenum:
            for i in range(lasr_rt_num):
                self.last_response_time_matrix_list.append(self.last_response_time_matrix)
            self.last_response_time_matrix_list.append(copy.deepcopy(response_time_matrix))
        else:
            self.last_response_time_matrix_list.pop(0)
            self.last_response_time_matrix_list.append(copy.deepcopy(response_time_matrix))

        for i in range(lasr_rt_num):
            state.append(self.last_response_time_matrix_list[i])
        state.append(response_time_matrix)
        state.append(permit_place_matrix)
        state.append(last_action_matrix)
        # print("state:", state)

        # self.last_response_time_matrix = copy.deepcopy(response_time_matrix)  # Save the last matrix

        # This is the episode time.
        episode_time = float(original_date_str[data_num[len(data_num) - 1] + 2:])
        episode_time = round(episode_time, 1)

        """ For finishing: finish if warehouse is full or you've processed
        1000 orders."""
        # TODO: specify new end conditions (read above)
        finish_no = float(original_date_str[data_num[len(data_num)-1]+1])
        is_end = True if finish_no == 1 else False
        if finish_no == 1:
            is_end = True
            self.episodeTimes.append(episode_time)
        else:
            is_end = False

        return state, is_end, episode_time
