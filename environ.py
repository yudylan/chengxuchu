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
    #a_bound = [-1,1]
    a_bound[0] = [0,800]    ###出力者B的出力上下限,可为负值，即可做正功，也可以做负功
    a_bound[1] = [-500,500]    ###出力者C的出力上下限
    a_bound[2] = [0,3000]    ###出力者D的出力上下限
    fb_output = []
    fc_output = []
    fd_output = []
    def __init__(self):
        self.current_soc = 0.6
        self.time_step = 0
        self.load = pd.read_csv('load.csv')   ###load为总电量需求
        self.pv = pd.read_csv('pv.csv')   ###pv为光伏发电量
        self.reset()
        

    def reset(self):
        self.current_b = 0.6
        self.state = [self.load, self.pv, self.current_b]
        self.action = [self.fb_output, self.fc_output, self.fd_output]

        return np.array(self.state)
        return np.array(self.action)
        
    def fbpower(self, action):
        #fb_output = action[0] * capacityb        ####若将出力范围统一变换到[-1,1]，capacityb为发电机出力的上下界的差值
        fb_output = action[0]        ###action[0]直接表示发电机的发电量
        return fb_output
        
    def fcpower(self, action):
        #fc_output = action[1] * capacityc        ####若将出力范围统一变换到[-1,1]，capacityc为蓄电池出力的上下界的差值
        fc_output = action[1]        #action[1]直接表示蓄电池的出力值，即充电值或放电值，，若为正，则代表蓄电池放电，若该值为负，则代表给电池充电
        return fc_output
        
    def fdpower(self, action):
        #fd_output = action[2] * capacityd        ####若将出力范围统一变换到[-1,1]，capacityd为蓄电池出力的上下界的差值
        fd_output = action[2]      ###action[2]直接表示从上级电网的购电量
        return fd_output   
        
    load = pv + fb_output + fc_output + fd_output    ####电能平衡，即总用电量=总发电量   (注：购电量也称作为发电量吧)
    
    def step(self, action):
        ##state = self.state
        reward_penalty = 0
        reward_chengben = 0

        fb_output = self.fbpower(action)
        fc_output = self.fcpower(action)
        fd_output = self.fdpower(action)
        
        #########蓄电池soc的更新公式
        self.current_c = self.current_soc - fc_output / capacity  ##蓄电池的容量为capacity
        
        reward_chengben = 10 * fb_output + dianjia * fd_output   ##10为发电机发1度电的成本，dianjia为从上级电网买1度电的成本
        
        if abs(self.current_b - 0.5) > 0.3:
            reward_penalty = 300.0 * (abs(self.current_b - 0.5) + 0.7)    ##将soc限制在[0.2,0.8]的范围内，若超出此范围，则有处罚成本
            
        reward = -(reward_penalty + reward_chengben) / 10000.0      ##总的reward，除以10000是为了正则化


        self.time_step = self.time_step + 1
        
        
        self.state = [self.load, self.pv, self.current_soc]
        return np.array(self.state), reward, {}
        

    

