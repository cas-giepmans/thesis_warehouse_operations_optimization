# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:44 2022

Just a scratch document for testing some stuff.

@author: casgi
"""
import os
import numpy as np
import torch
import random
from Warehouse import Warehouse
from pathlib import Path
from datetime import date, datetime

# data_str = '14.5_15.5_16.5_17.5_18.5_19.5_20.5_21.5_12.3_13.3_14.3_15.3_16.3_17.3_18.3_19.3_13.5_14.5_15.5_16.5_0_18.5_19.5_20.5_24.9_25.9_26.9_27.9_28.9_29.9_30.9_31.9_12.9_0_14.9_15.9_16.9_17.9_18.9_19.9_13.1_14.1_15.1_0_17.1_18.1_19.1_20.1_14.5_15.5_16.5_17.5_18.5_0_20.5_21.5_12.3_13.3_14.3_15.3_0_17.3_18.3_19.3_13.5_14.5_15.5_0_17.5_18.5_19.5_0_24.9_25.9_26.9_27.9_28.9_0_30.9_0_12.9_13.9_14.9_15.9_16.9_17.9_18.9_19.9_0_14.1_15.1_16.1_0_0_19.1_20.1_0+91.27'
# indv_entries = data_str.split("_")
# print(f"{len(indv_entries)} entries")

# counter = 0
# for word in indv_entries:
#     if word == "0":
#         counter += 1
#     else:
#         continue

# print(f"Number of 0's among entries: {counter}")
# for word in indv_entries:
#     print(word)

# """
#     For the simulation matrix: generate random matrix, multiply with inverse
#     availability matrix (dot product). Then the positions that aren't available
#     get a value of 0.
# """


# def index_str(str1, str2):
#     """Finds the specified string str1 containing the full location of the specified substr2, Returns as a list"""
#     length_2 = len(str2)
#     length_1 = len(str1)
#     index_str2 = []
#     i = 0
#     while str2 in str1[i:]:
#         index_tmp = str1.index(str2, i, length_1)
#         index_str2.append(index_tmp)
#         i = (index_tmp + length_2)
#     return index_str2


# str2_index = index_str(data_str, "_")

# print(str2_index)

# #number = np.random.uniform(3.0, 32.0, 96)
# #print(np.round(number, 1).tolist())
# simString = '_'.join([str(x) for x in np.round(np.random.uniform(3.0, 32.0, 96)).tolist()])
# simString = '_' + simString + '_' + '0+22.35'
# print(simString)

# random_numbers = (np.random.rand(96) > 0.5).astype(int)
# print(random_numbers)
# random_array = np.asarray(random_numbers, dtype=bool, order=None)
# random_array = random_array.reshape((2, 6, 8))
# print(random_array)
# random_array = np.reshape(random_array, (12, 8))
# print(random_array)

# print(np.finfo(np.float32).eps.item())

# # class test_class():
# #     def __init__(self):
# #         self.param =5

# #     def calc(self, some_value):
# #         return self.param * some_value

# # test = test_class()

# # print(test(12))

# print(np.random.uniform())
# an_array = np.array([6, 10, 1, 2, 7])
# print(np.where(an_array > 5)[0].tolist())
# for array in [an_array]:
#     array = (array - np.min(array)) / (np.max(array) - np.min(array))
# print(an_array)

# wh = Warehouse()
# wh.CalcRTM()

# wh.PrintIdMatrix(print_it=True)
# id_matrix = wh.PrintIdMatrix(print_it=False)

# id_matrix = np.transpose(id_matrix, (1, 0, 2))
# print(id_matrix)
# id_matrix = np.reshape(id_matrix, (12, 8))
# print(id_matrix)
# id_matrix = id_matrix.transpose((1, 0)).flatten().tolist()
# print(id_matrix)

# id_tensor = torch.from_numpy(id_matrix)
# print(id_tensor)

# print(id_tensor.size())
# # id_tensor = id_tensor.view(-1, 96)
# id_tensor = id_tensor.reshape((1, 96))
# print(id_tensor)
# print(id_tensor.size())

# id_tensor = id_tensor.transpose(1, 2)
# print(id_tensor)


# c_dir = Path(os.getcwd())
# # m_dir = Path.parents
# os.chdir('../..')
# print(os.listdir())
# # path = Path(r"\Thesis\Experiments\LeiLuo's model")
# os.chdir(r"Experiments\LeiLuo's model")
# print(os.listdir())
# c_dir = os.getcwd()
# today_dir = f"{date.today().day}-{date.today().month}-{date.today().year}"
# try:
#     os.mkdir(today_dir)
# except FileExistsError:
#     print(f"Using the folder '{today_dir}'")

# os.chdir(today_dir)  # Now we're inside today's folder.

# nr_files = len(os.listdir())
# # fin_time = f"{datetime.now().hour}.{datetime.now().minute}.{datetime.now().second}"
# ex_dir = f"Exp. {nr_files + 1}"  # ", t_fin = {fin_time}"
# print(ex_dir)
# try:
#     os.mkdir(ex_dir)
# except FileExistsError:
#     print(f"Something went wrong, folder '{ex_dir}' already exists...?")


# numbers = np.arange(10).tolist()
# print(numbers[::-1])

# vt_time = 0.0
# for r in range(2):
#     for f in range(6):
#         for c in range(6):
#             vt_time += f * 1.2
# print(291.6 - vt_time)

# for i in range(7):
#     print(i)
# _sum = 0
# for i in range(5):
#     _sum

# print(sum([(2 * 1.2 / 1.65) * i for i in range(5)]))

# names = ['a', 'b', 'c', 'd', 'e']
# a_dict = {}
# for i in range(5):
#     a_dict[names[i]] = i

# for key, value in a_dict.items():
#     print(key, value)

# For init_fill: use values such that 0.0 < init_fill < 1.0
# init_fill = 37.0 / 72.0
# curr_fill = 36.0 / 72.0
# # if init_fill == 1.0 or curr_fill == 1.0:
# #     p_outf = 1.0
# # elif init_fill == 0.0 or curr_fill == 0.0:
# #     p_outf = 0.0
# p_outf = curr_fill**(init_fill / (1 - init_fill))
# print(p_outf)

# val = 0.0
# if val:
#     print("True")
# elif not val:
#     print("False")

# print(bool(1.0))

# a = 5
# b = 6
# print(abs(a-b))

# print(torch.finfo(torch.float32))

# # print(0/0)


# def logsumexp(x):
#     c = torch.max(x)
#     return c + torch.log(torch.sum(torch.exp(x - c)))


# c = torch.tensor(10.)
# d = torch.tensor(0.)
# e = torch.div(c, d)
# print(f"e: {e}")
# print(f"is e nan? {torch.isnan(e)}")
# e = torch.nan_to_num(e)
# print(f"e (nan to num): {e}")
# print(f"is e nan? {torch.isnan(e)}")

# print(np.log(3))
# f = torch.tensor([1000, 1000, 1000])
# print(torch.exp(f))
# # print(torch.logsumexp(f, -1, keepdim=True))
# print(torch.exp(torch.sub(f, logsumexp(f))))

# ten1 = torch.tensor([10, 11, 12])
# ten2 = torch.BoolTensor([True, True, False])
# print(torch.mul(ten1, ten2))

a = [1, 2, 3]
print([0 for i in a])

b = []
if not bool(b):
    print("hi")

my_dict = {
    0: "a",
    1: "b",
    2: "c"}

res = list(zip(*my_dict.items()))

print(my_dict.items())
print(list(res[0]))

c = [0.5, 0.2, 0.2]
res2 = random.choices(a, weights=c, k=1000)
print(res2.count(1), res2.count(2), res2.count(3))

print(*random.choices(a, weights=c, k=1))


def CalcInfeedProb(cur_f, des_f):
    if cur_f == des_f:
        return 0.5
    elif cur_f > des_f:
        return 1 - (0.5 / (1.0 - des_f) * cur_f - (0.5 / (1.0 - des_f) - 1))
    elif cur_f < des_f:
        return 1 - 0.5 / des_f * cur_f


print(CalcInfeedProb(1.0, 0.75))

print(int(5.9999))

something = np.array([5, 6, 3])
print(something)
something[1] -= 4
print(something)


bim = {}
print(len(bim.items()) != 0)

asdf = [[1, 2], [1, 2]]
print(sum([x[0] for x in asdf]))

hihi = int(2)
print(int(hihi + 0.5 / 4))

for i in range(1):
    print(i)
