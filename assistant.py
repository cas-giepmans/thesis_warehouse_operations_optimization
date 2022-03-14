import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder


class DataProcessing:
    def __init__(self, data):
        #print('g')
        self.data = data

    def getResponseTime(self):
        '''将字符串转换为数组'''
        dataNum = self.indexstr(self.data, '_')
        responseTime = []
        responseTime_record = []
        finish = []
        post_finsh =[]
        post_finsh_1 = []
        noMu = []
        noMu_1 = []
        hasMu = []
        hasMu_1 = []
        endTime = []
        temp = 0
        for i in range(len(dataNum)-1):
            thisResponseTime = float(self.data[dataNum[i]+1:dataNum[i+1]])
            responseTime.append(thisResponseTime)
        maxR = max(responseTime)
        minR = min(responseTime)

        for j in range(len(responseTime)):
            #responseTime[j] = (responseTime[j]-minR)/(maxR-minR)
            #responseTime[j] = responseTime[j]/ maxR
            responseTime_record.append(responseTime[j] / 1000)
            if responseTime_record[temp] == 0:
                noMu.append(0.)
                hasMu.append(0.1)
            else:
                noMu.append(0.1)
                hasMu.append(0.)
            temp += 1
            if temp==5:
                noMu_1.append(noMu)
                hasMu_1.append(hasMu)
                post_finsh.append(responseTime_record)

                noMu = []
                hasMu = []
                responseTime_record = []
                temp = 0
        post_finsh_1.append(noMu_1)
        post_finsh_1.append(hasMu_1)
        #post_finsh_1.append(post_finsh)
        finish.append(float(self.data[dataNum[len(dataNum)-1]+1]))
        endTime.append(float(self.data[dataNum[len(dataNum)-1]+2:]))
        return post_finsh_1, finish, endTime

    def indexstr(self,str1,str2):
        '''查找指定字符串str1包含指定子字符串str2的全部位置，
        以列表形式返回'''
        lenth2=len(str2)
        lenth1=len(str1)
        indexstr2=[]
        i=0
        while str2 in str1[i:]:
            indextmp = str1.index(str2, i, lenth1)
            indexstr2.append(indextmp)
            i = (indextmp + lenth2)
        return indexstr2




