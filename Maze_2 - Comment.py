#         0       1       2
#     ╔═══════╦═══════╦═══════╗
# 0   ║   +4  |       ║ Sword ║
#     ╠───────────────║───────╣
# 1   ║       |       |       ║
#     ╠════════───────════════╣
# 2 → |   T   |       |   T   | ←
#     ╚═══════╩═══════╩═══════╝
#         ↑               ↑
#        +10         +18 (Sword)
import numpy as np
class Maze():
    def __init__(self):
        self.state_dimension = tuple((3,3,2,2)) # 状态的维度为 3×3×2×2 的矩阵
        # 前两个维度代表位置，第三个维度代表左上角的额外奖励是否还存在，第四个维度代表右上角的武器是否还存在
        self.terminate = np.array([[2,0],[2,2]]) # 左下角和右下角为终止状态
        self.all_actions = np.array([[-1,0],[1,0],[0,-1],[0,1]]) # 上下左右四个行动
        self.wall = np.array([[[0,1],[0,2]],[[1,0],[2,0]],[[1,2],[2,2]]]) # 墙的位置
    # 判断是否为不可能的状态
    def judge_impossible(self,state):
        position = state[0:2]
        if (position == np.array([0,0])).all() and state[2]==1: # 玩家到达左上角并且额外奖励还存在是不可能的
            return 1
        if (position == np.array([0,2])).all() and state[3]==1: # 玩家到达右上角并且武器还存在是不可能的
            return 1
        return 0
    # 判断是否到达终止状态
    def judge_terminate(self,position):
        is_terminate = 1 in np.all(position==self.terminate,axis=1)
        return is_terminate
    # 判断是否穿墙
    def judge_wall(self,current_position,next_position):
        is_wall_1 = 1 in np.all(np.array([current_position,next_position])==self.wall,axis=(1,2))
        is_wall_2 = 1 in np.all(np.array([next_position,current_position])==self.wall,axis=(1,2))
        return is_wall_1 or is_wall_2
    # 状态转移，输入当前状态和行动，输出下一时刻的状态和即时奖励
    def state_transition(self,current_state,action):
        current_position = current_state[0:2] # 读取位置信息
        next_position = np.clip(current_position + action,0,2) # np.clip() 确保不会超出迷宫的边缘
        if self.judge_wall(current_position,next_position): # 如果撞墙
            next_position = current_position.copy() # 状态不变
        next_state = current_state.copy() # 下一时刻的状态
        next_state[0:2] = next_position # 首先更新位置信息
        # 然后更新其他信息，同时计算奖励
        bonus = 0 # 额外奖励
        if (next_position == np.array([2,0])).all() : # 左下角终止状态额外奖励为 10
            bonus = 10
        if (next_position == np.array([0,0])).all() and current_state[2] == 1: # 左上角额外奖励为 4，前提是它存在
            next_state[2] = 0 # 现在不存在了
            bonus = 4
        if (next_position == np.array([0,2])).all() and current_state[3] == 1: # 右上角获得武器，前提是它存在
            next_state[3] = 0 # 现在不存在了
        if (next_position == np.array([2,2])).all() and current_state[3] == 0: # 右下角击杀木乃伊获得 18 额外奖励，前提是有武器
            bonus = 18
        reward = -1 + bonus
        return reward,next_state

class Agent():
    def __init__(self,Gamma=1,Precision=1e-10):
        self.gamma = Gamma # 折扣率
        self.precision = Precision # 判断是否收敛的精度
    # 价值迭代的主要函数
    def value_iteration(self,Maze):
        dimension = Maze.state_dimension # 环境给出的状态的维度，在这里是 3×3×2×2 的矩阵 dimension = (3,3,2,2)
        old_value = np.zeros(dimension) # np.zeros((3,3,2,2)) 储存旧价值估计
        new_value = np.zeros(dimension) # np.zeros((3,3,2,2)) 储存新价值估计
        Row = dimension[0] # 行数 = 3
        Column = dimension[1] # 列数 = 3
        delta = 1 # 初始默认新旧价值还差得远
        while delta>self.precision:
            for i in range(Row): # 遍历每一行
                for j in range(Column): # 遍历每一列
                    for m in range(2): # 左上角额外奖励是否存在
                        for n in range(2): # 右上角武器是否存在
                            current_state = np.array([i,j,m,n]) # 当前状态
                            current_position = current_state[0:2] # 读取位置信息
                            # 判断是否是终止状态或者不可能的状态
                            if not (Maze.judge_terminate(current_position) or Maze.judge_impossible(current_state)):
                                # 临时的价值向量，储存状态-行动最优价值的估计值
                                temp_value = np.zeros(len(Maze.all_actions))
                                for k in range(len(Maze.all_actions)):  # 遍历所有可能的行动
                                    # 调用环境的状态转移函数，得到即时奖励和下一时刻的状态
                                    reward,next_state = Maze.state_transition(current_state,Maze.all_actions[k])
                                    # 状态-行动最优价值的估计值 = 即时奖励 + γ×下一时刻的状态最优价值的估计值
                                    temp_value[k] = reward + self.gamma*old_value[tuple(next_state.astype(int))]
                                # 新的状态最优价值的估计值 = 最大的状态-行动最优价值的估计值
                                new_value[i,j,m,n] = np.max(temp_value)
            delta = np.max(np.abs(new_value-old_value)) # 计算新旧估计的差值，取绝对值，再取最大值
            old_value = new_value.copy() # 用新的估计值代替旧的估计值
        self.state_optimal_value = new_value.copy()# 收敛之后把状态最优价值储存
    # 最优策略函数
    def optimal_policy(self,Maze,state):
        # 和价值迭代的过程一样，只是现在我们利用价值迭代得到的状态最优价值
        temp_value = np.zeros(len(Maze.all_actions))
        for k in range(len(Maze.all_actions)):
            reward,next_state = Maze.state_transition(state,Maze.all_actions[k])
            temp_value[k] = reward + self.gamma*self.state_optimal_value[tuple(next_state.astype(int))]
        # 选择状态-行动最优价值最大的行动
        optimal_k = np.argmax(temp_value)
        optimal_action = Maze.all_actions[optimal_k]
        return optimal_action
    # 给一个初始状态，返回最优路径，以及回报值
    def optimal_trajectory(self,Maze,initial_state):
        total_return = 0 # 记录回报值
        power = 0 # 折扣率的指数，每一步都增加 1
        all_state = np.array([initial_state]) # 记录路径中的所有状态
        current_state = initial_state.copy()
        terminate = 0
        while not terminate:
            optimal_action = self.optimal_policy(Maze,current_state)
            reward,next_state = Maze.state_transition(current_state,optimal_action)
            total_return += reward*self.gamma**power # 将即时奖励添加到回报值当中
            power += 1 # 下一时刻的奖励将会打折扣
            all_state = np.r_[all_state,np.array([next_state])] # 把下一时刻的状态添加到路径当中
            current_state = next_state.copy() # 新状态替换旧状态
            terminate = Maze.judge_terminate(current_state[0:2]) # 判断是否到达终止状态
        return all_state,total_return

Maze = Maze() # 创建一个迷宫环境
Agent = Agent(Gamma=0.9) # 创建一个玩家
Agent.value_iteration(Maze) # 价值迭代
print("状态最优价值：")
print(Agent.state_optimal_value[:,:,1,1])
print("最优路径：")
all_state,total_return = Agent.optimal_trajectory(Maze,np.array([0,1,1,1])) # 从第一行中间位置出发
print(all_state)
print("奖励之和：")
print(total_return)
