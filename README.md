# chengxuchu
情境是这样的：任务是进行机组一天的出力决策，使花费(huafei)最小。研究案例中的电力系统包含内燃机、光伏、蓄电池及负荷。状态量是负荷需求(load)、光伏发电(pv)和电池的soc共3个量。其中，当前时刻的load 和 pv 从数据集中获取，soc已知初始值。动作量包括内燃机发电（pd_output）和蓄电池充放电（pb_output），其实还包括从主电网购电量（pg_output），但主电网购电量可以通过负荷-内燃机发电-蓄电池充放电得到，因此动作量包括这2个。由于我目前采用的是DQN，因此将动作量进行离散化了，我目前是将将这两个动作量分别离散化为10个值(动作间隔均为20KW)，所以我个人认为动作共有10*10=100个。在程序中我是这样编写的：

self.state_dim = 3      ##输入是离散的数据，包括负荷需求、光伏发电量、SOC
  self.action_dim = 100    ##内燃机发电和电池充放电分别离散成10个动作
读取当前时刻的load和pv数据：
        self.load = pd.read_csv('load.csv')
        self.pv = pd.read_csv('pv.csv')

def reset(self):
        self.current_soc = 0.6
        self.state = [self.load, self.pv, self.current_soc]
        self.action = [self.pd, self.pb]

        return np.array(self.state)
        return np.array(self.action)
        
    def pdpower(self, action):
        if action[0] == 0:
            pd_output = 0
        else:
            pd_output = action[0] * 20       ##内燃机的发电量
        return pd_output
        
    def pbpower(self, action):
        if action[1] == 0:
            pb_output = 0
        else:
            pb_output = action[1] * 20        ##蓄电池的充放电值
        return pb_output
        
    pg_output = self.load - pd_output - pb_output   ##从主电网购电

reward由两部分组成：第一大部分是成本(reward_chengben)，包括内燃机发电的成本和从主电网购电的成本（单价分别假设为了10和16）；第二大部分是蓄电池的处罚金额(reward_penalty)，将蓄电池的soc设置在[0.2,0.8]的范围内，如果超过该范围，将产生处罚成本。

reward_chengben = 10 * pd_output + 16 * pg_output       ###成本

if abs(self.current_soc - 0.5) > 0.3:
       reward_penalty = 300.0 * (abs(self.current_soc - 0.5) + 0.7)    ###处罚金额


reward = -(reward_penalty + reward_chengben) / 100.0   ##reward

huafei = huafei + env.pdpower(action) * 10 + pg_output * 16  ###文中的花费，本文目的即为花费最小

其中， SOC的更新语句为：
     self.current_soc = self.current_soc - pb_output / 200   ###蓄电池容量为200KW

