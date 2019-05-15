import numpy as np
import pandas as pd

class Environment():

    def __init__(self):
        self.current_soc = 0.6
        self.time_step = 0
        capacity=200
        self.load = pd.read_csv('load.csv')
        self.pv = pd.read_csv('pv.csv')
        self.reset()
        

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
            pd_output = action[0] * 20
        return pd_output
        
    def pbpower(self, action):
        if action[1] == 0:
            pb_output = 0
        else:
            pb_output = action[1] * 20
        return pb_output
        
    pg_output = self.load - pd_output - pb_output
        
    def step(self, action):
        ##state = self.state
        reward_penalty = 0
        reward_chengben = 0
        pg_output = []

        pd_output = self.pdpower(action)
        pb_output = self.pbpower(action)
        #########SOC的更新公式
        self.current_soc = round(self.current_soc - pb_output / 200, 4)   ###蓄电池容量为200KW
        
        reward_chengben = 10 * pd_output + 16 * pg_output 
        
        if abs(self.current_soc - 0.5) > 0.3:
            reward_penalty = 300.0 * (abs(self.current_soc - 0.5) + 0.7)
        ###reward = -(reward_penalty + ice_output) / 100000.0
        reward = -(reward_penalty + reward_chengben) / 100.0


        self.time_step = self.time_step + 1
        
        
        self.state = [self.load, self.pv, self.current_soc]
        return np.array(self.state), reward, {}
        

    

