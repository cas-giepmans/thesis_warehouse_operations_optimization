class TrainingGame:
    def __init__(self, x_dim, y_dim):
        # print("this is a mathTreeGame")
        self.X_dim = x_dim
        self.Y_dim = y_dim

        # 游戏要返回多少个state，共有多少个action应该由游戏本身决定
        # 并依据这些结果，去确定创建神经网络的参数
        self.StateNum = 3  # 本游戏返回两个state网络，分别是：1.有哪些可用单元格；2.上次选取的单元格
        self.StateBoard = self.X_dim * self.Y_dim
        self.ActionNum = self.X_dim * self.Y_dim

        self.AvailablePos = []
        self.chooseTime = 0  # 本局游戏做了多少次选择了
        self.allAction = []

    def init_para(self):
        # 初始化模型
        print("initPara")

    def get_init_state(self):
        print("my init state")
        state = []
        self.init_para()
        return state

    def do_action(self, action):
        print("doAction")
        state = []
        reward = 0
        is_end = True
        return state, reward, is_end

    def get_available_pos(self):
        # print("getAvailablePos")
        return self.AvailablePos
