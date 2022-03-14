import copy

class MathTreeGame():
    def __init__(self, xDim, yDim):
        #print("this is a mathTreeGame")
        self.XDim = xDim
        self.YDim = yDim

        #游戏要返回多少个state，共有多少个action应该由游戏本身决定
        #并依据这些结果，去确定创建神经网络的参数
        self.StateNum = 3 # 本游戏返回两个state网络，分别是：1.有哪些可用单元格；2.上次选取的单元格
        self.StateBoard = self.XDim * self.YDim
        self.ActionNum = self.XDim * self.YDim


        self.nowState = []
        self.AvailablePos = []
        self.chooseTime = 0 #本局游戏做了多少次选择了
        self.allAction = []

    def getInitState(self):
        #print("my init state")
        state = []
        state_allPos = [[1. for col in range(self.XDim)] for row in range(self.YDim)] # 当前局面
        state_lastState = state_allPos  # 上一次面对的局面
        state_lastAction = [[0. for col in range(self.XDim)] for row in range(self.YDim)] # 前几次的选择
        state.append(state_allPos)
        state.append(state_lastState)
        state.append(state_lastAction)

        #初始化模型
        self.AvailablePos = [1.] * self.XDim * self.YDim
        self.chooseTime = 1  # 本局游戏做了多少次选择了
        self.allAction = []
        self.nowState = [[1. for col in range(self.XDim)] for row in range(self.YDim)]

        return state

    def doAction(self, action):
        # 1.执行action
        # 2.改变AvailablePos

        #print("do action")
        state = []
        Reward = 0
        isEnd = False

        action = action
        line = action // self.XDim
        pos = action % self.XDim
        changePos = self.nowState[line][pos]
        #returnState = []
        if changePos == 0:
            isEnd = True
            state = self.nowState
            Reward = 0
        else:
            isEnd = False
            Reward = action / self.chooseTime
            # Reward = 1

            self.AvailablePos[action] = 0 #选择的位置不可再选择

            # 获取新的state
            self.nowState[line][pos] = 0.0
            myState_1 = self.nowState  # 可以选择的位置

            # 获取上一次的state（应该可以记录多次）
            """Using the standard deepcopy method is very inefficient, instead write a custom one inside the class, overriding the function call - Cas"""
            """Premature optimization is the death of all [software] - Kutalmis Gokalp Ince"""
            myState_2 = copy.deepcopy(myState_1)
            myState_2[line][pos] = 1.0
            # model.zero_originalState[line][pos] = 1.0
            # myState_2 = copy.deepcopy(model.last_State[0])
            #myState_2 = model.last_State[0].copy()
            #model.last_State[0] = myState_1

            # 回溯更多的动作
            actionNum = 3
            all_action_length = len(self.allAction)
            myState_3 = [[0 for col in range(self.XDim)] for row in range(self.YDim)]
            if all_action_length > 0:
                for i in range(min(actionNum, all_action_length)):
                    myAct = self.allAction[-1 * (i + 1)]
                    this_line = myAct // self.XDim
                    this_pos = myAct % self.XDim
                    myState_3[this_line][this_pos] = 1.
            state.append(myState_1)
            state.append(myState_2)
            state.append(myState_3)

            # 判定游戏是否完成所有选择
            if max(max(self.nowState)) == 0.:
                isEnd = True
                # print("have finished all the game")

            self.allAction.append(action)
            self.chooseTime += 1

        return state, Reward, isEnd

    def getAvailablePos(self):
        return self.AvailablePos

    def dolastAction(self):
        pass



