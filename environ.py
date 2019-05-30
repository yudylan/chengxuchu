# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:18 2019

@author: yuaner
"""

import numpy as np
import pandas as pd

class Environment():
    
    s_dim = 3
    a_dim = 3
    #a_bound = [-1,1]
    a_bound[0] = [0,1200]    ###发电机的出力上下限,也就是发电机的最大功率是1200KW
    a_bound[1] = [-800,800]    ###蓄电池的出力上下限，最大功率是800KW，也就是充电最大功率是800KW，放电最大功率也是800KW
    a_bound[2] = [-3000,3000]    ###上级电网出力的上下限，最大功率是3000KW，，也就是买电的上限是3000KW，卖电的上限也是3000KW
    fb_output = []
    fc_output = []
    fd_output = []
    def __init__(self):
        self.time_step = 0
        
        self.current_soc = 0.6
        #self.load = pd.read_csv('load.csv', index_col=0)   ###load为总电量需求 
        #self.pv = pd.read_csv('pv.csv')   ###pv为光伏发电量
        #self.dianjiabuy = pd.read_csv('dianjiabuy.csv')   ##dianjiabuy为从上级电网买1度电的电价，为分时电价，也就是一天的电价会随时间变化而变化
        #self.dianjiasell = pd.read_csv('dianjiasell.csv')   ##dianjiasell为卖给上级电网1度电的电价，，为也是分时电价
        
        self.load = pd.read_csv('shuju.csv', index_col=0)   ###表格的第1列为load，为总电量需求 
        self.pv = pd.read_csv('shuju.csv', index_col=1)   ###表格的第2列为pv,为光伏发电量
        self.dianjiabuy = pd.read_csv('shuju.csv', index_col=2)   ##表格的第2列为dianjiabuy,为从上级电网买1度电的电价，为分时电价
        self.dianjiasell = pd.read_csv('shuju.csv', index_col=3)   ##表格的第3列为dianjiasell,为卖给上级电网1度电的电价，，为也是分时电价
        
        ##  峰时段为10:00-15:00、18:00-21:00；平时段为07:00-10:00、15:00-18:00、21:00-23:00；谷时段为00:00-07:00、23:00-24:00
       
        self.reset()
        

    def reset(self):
        self.current_soc = 0.6
        self.state = [self.load, self.pv, self.current_soc]
        self.action = [self.fb_output, self.fc_output, self.fd_output]

        return np.array(self.state)
        return np.array(self.action)
        
    def fbpower(self, action):
        #fb_output = action[0] * capacityb        ####若将出力范围统一变换到[-1,1]，capacityb为发电机出力的上下界的差值,等于1200
        fb_output = action[0]        ###action[0]直接表示发电机的发电量
        return fb_output
        
    def fcpower(self, action):
        #fc_output = action[1] * capacityc        ####若将出力范围统一变换到[-1,1]，capacityc为蓄电池出力的上下界的差值,等于6000
        fc_output = action[1]        #action[1]直接表示蓄电池的出力值，即充电值或放电值，，若为正，则代表蓄电池放电，若该值为负，则代表给电池充电
        return fc_output
        
    def fdpower(self, action):
        #fd_output = action[2] * capacityd        ####若将出力范围统一变换到[-1,1]，capacityd为蓄电池出力的上下界的差值
        fd_output = action[2]      ###action[2]直接表示上级电网的出力值，为正时，是从上级电网的买电量；为负时，是卖给上级电网的电量
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
        self.current_soc = self.current_soc - fc_output / capacity  ##蓄电池的容量为capacity
        
        if fd_output >= 0:
            reward_chengben = 10 * fb_output *fb_output + dianjiabuy * fd_output   ##10为发电机发1度电的成本，dianjiabuy为从上级电网买1度电的成本
        eles
            reward_chengben = 10 * fb_output *fb_output + dianjiasell * fd_output   ##10为发电机发1度电的成本，dianjiasell为卖1度电的利润
        
        if abs(self.current_b - 0.5) > 0.3:
            reward_penalty = 300.0 * (abs(self.current_b - 0.5) + 0.7)    ##将soc限制在[0.2,0.8]的范围内，若超出此范围，则有处罚成本
            
        reward = -(reward_penalty + reward_chengben) / 10000.0      ##总的reward，除以10000是为了正则化


        self.time_step = self.time_step + 1
        
        
        self.state = [self.load, self.pv, self.current_soc]
        return np.array(self.state), reward, {}
        

    

