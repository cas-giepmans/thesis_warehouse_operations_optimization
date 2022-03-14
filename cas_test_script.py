# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:44 2022

@author: casgi
"""
import os
import numpy as np

data_str = '14.5_15.5_16.5_17.5_18.5_19.5_20.5_21.5_12.3_13.3_14.3_15.3_16.3_17.3_18.3_19.3_13.5_14.5_15.5_16.5_0_18.5_19.5_20.5_24.9_25.9_26.9_27.9_28.9_29.9_30.9_31.9_12.9_0_14.9_15.9_16.9_17.9_18.9_19.9_13.1_14.1_15.1_0_17.1_18.1_19.1_20.1_14.5_15.5_16.5_17.5_18.5_0_20.5_21.5_12.3_13.3_14.3_15.3_0_17.3_18.3_19.3_13.5_14.5_15.5_0_17.5_18.5_19.5_0_24.9_25.9_26.9_27.9_28.9_0_30.9_0_12.9_13.9_14.9_15.9_16.9_17.9_18.9_19.9_0_14.1_15.1_16.1_0_0_19.1_20.1_0+91.27'
indv_entries = data_str.split("_")
print(f"{len(indv_entries)} entries")

counter = 0
for word in indv_entries:
    if word == "0":
        counter += 1
    else:
        continue

print(f"Number of 0's among entries: {counter}")
for word in indv_entries:
    print(word)
    
"""
    For the simulation matrix: generate random matrix, multiply with inverse 
    availability matrix (dot product). Then the positions that aren't available
    get a value of 0.
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

str2_index = index_str(data_str, "_")

print(str2_index)

#number = np.random.uniform(3.0, 32.0, 96)
#print(np.round(number, 1).tolist())
simString = '_'.join([str(x) for x in np.round(np.random.uniform(3.0, 32.0, 96)).tolist()])
simString = '_' + simString + '_' + '0+22.35'
print(simString)