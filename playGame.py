
import numpy as np
import torch
from machineLearning_AC.PlantGame_AVSRS import PlantGame_AVSRS as plantGame
from machineLearning_AC.policy_net_AC_pytorch import PolicyValueNet as trainNet

def run():
    modelPath = 'plantPolicy_3_2_100.model'
    XDim, YDim = 3, 2

    My_Game_Model = plantGame(XDim, YDim)

    My_Train_NET = trainNet(My_Game_Model.XDim, My_Game_Model.YDim, My_Game_Model.StateNum,
                            My_Game_Model.StateBoard, My_Game_Model.ActionNum)
    # 需要去确认stateNum这个参数和模型的参数是否一样,否则会报错

    My_Train_NET.policy_value_net = torch.load(modelPath)

    myState = My_Game_Model.getInitState()
    availablePos = My_Game_Model.getAvailablePos()

    allAction = []

    while True:
        action = My_Train_NET.select_action(np.array(myState), availablePos)
        allAction.append(action)
        myState, thisReward, isEnd = My_Game_Model.doAction(action)
        availablePos = My_Game_Model.getAvailablePos()
        if isEnd:
            break
    print("gameOver")
    print("仿真时间：", My_Game_Model.episodeTime[0])
    print("顺序动作", allAction)

if __name__ == '__main__':
    run()