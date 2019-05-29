# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:18 2019

@author: yuaner
"""

import numpy as np
import pandas as pd

class Environment():
    
    s_dim = 2
    a_dim = 3
    a_bound[0] = [-500,500]    ###出力者B的出力上下限,可为负值，即可做正功，也可以做负功
    a_bound[1] = [0,800]    ###出力者C的出力上下限
    a_bound[2] = [0,1000]    ###出力者D的出力上下限
    fb_output = []
    fc_output = []
    fd_output = []
    def __init__(self):
        self.current_b = 0.6
        self.time_step = 0
        #CAPACITY = 2000   ###出力者B的容量为2000
        self.load = pd.read_csv('load.csv')   ###load为总力需求减去出力者A出力后得到
        self.reset()
        

    def reset(self):
        self.current_b = 0.6
        self.state = [self.load, self.current_b]
        self.action = [self.fb_output, self.fc_output, self.fd_output]

        return np.array(self.state)
        return np.array(self.action)
        
    def fbpower(self, action):
        #fb_output = action[0] * self.load      ####若将动作进行归一化了
        fb_output = action[0]      ####未将动作归一化
        return fb_output
        
    def fcpower(self, action):
        #fc_output = action[1] * self.load      ####若将动作进行归一化了
        fc_output = action[1]      ####未将动作归一化
        return fc_output
        
    def fdpower(self, action):
        #fd_output = action[2] * self.load      ####若将动作进行归一化了
        fd_output = action[2]      ####未将动作归一化
        return fd_output   
        
    load = fb_output + fc_output + fd_output    ####力平衡
    
    def step(self, action):
        ##state = self.state
        reward_penalty = 0
        reward_chengben = 0

        fb_output = self.fbpower(action)
        fc_output = self.fcpower(action)
        fd_output = self.fdpower(action)
        
        #########b的更新公式
        self.current_b = self.current_b - fb_output / 2000   ###出力者B的容量为2000
        
        reward_chengben = 10 * fc_output + 16 * fd_output 
        
        if abs(self.current_b - 0.5) > 0.3:
            reward_penalty = 300.0 * (abs(self.current_b - 0.5) + 0.7)    ##将b限制在[0.2,0.8]的范围内，若超出此范围，则有处罚成本
            
        reward = -(reward_penalty + reward_chengben) / 100.0      ##总的reward


        self.time_step = self.time_step + 1
        
        
        self.state = [self.load, self.current_b]
        return np.array(self.state), reward, {}
        

    

