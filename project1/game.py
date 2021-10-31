import numpy as np
import datetime
import copy
from tqdm import tqdm
import math

from numpy.lib.function_base import append
from card import card, hand_cards, values


class single_game(object):
    def __init__(self, n = 17):

        '''类内变量'''
        self.cnum = n
        self.acards = []
        self.cards = []
        '''init dp to accelerate'''
        self.dp = np.empty((16, 16, 16, 16, 3))
        self.init_dp()
        '''最佳策略与最短步数'''
        self.strategy = []
        self.best_strategy = []
        self.steps = 15
        self.cardsfordfs = []
        self.xfordfs = 0
        self.flag = 0
        
        '''更新n'''
        self.renew_cards(n)
        
    def renew_cards(self, n):
        '''
        The function to change the cards number for single game\\
        Only the cards number are need\\
        then you can immediate acquire the strategy\\
        '''
        '''单人游戏总牌数'''
        self.cnum = n
        '''单人游戏手牌'''
        self.cards = hand_cards(self.cnum).mcard
        '''手牌分布数组'''
        self.acards = self.card_in_np()
        '''计算策略'''
        self.flag = 0
        self.strategy = []
        self.best_strategy = []
        self.steps = 15
        self.cardsfordfs = []
        self.xfordfs = 0
        '''for debugging'''
        
        #self.acards = [4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0]
        #self.acards = [4, 0, 4, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        #self.acards = [1, 3, 3, 3, 3, 2, 3, 2, 1, 1, 2, 2, 3, 0, 1]
        #self.acards = [2, 2, 1, 1, 0, 1, 0, 3, 3, 2, 2, 2, 3, 0, 1]
        #self.acards = [0, 1, 1, 1, 1, 1, 3, 4, 0, 0 ,2, 0, 0, 0, 0]
        #self.acards = [3, 2, 3, 3, 3, 1, 0, 2, 2, 3, 2, 1, 3, 1, 1]
        self.acards = [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0]
        #[2. 0. 3. 3. 3. 2. 2. 4. 1. 3. 1. 2. 3. 1. 0.]
        self.dfs()
        self.dfs_for_else(self.xfordfs)
        '''for debugging'''
        print("cards = ", self.acards)
        print("steps = ", self.steps)
        print("strategy:", self.best_strategy)

    def card_in_np(self):
        acards = np.zeros(15)
        for i in range(len(self.cards)):
            acards[self.cards[i].value.value] += 1
        #print(acards)
        return acards
    
    def dfs(self, x = 0):
        #print(x)
        #print(self.acards)
        '''超过步数，return'''
        if x >= self.steps: return
        '''统计牌数数组用于调用dp剪枝'''
        sum = np.zeros(5)
        for i in range(values.sjoker.value):
            if self.acards[i]:
                sum[int(self.acards[i] - 1)] += 1
        sum[4] = self.acards[values.sjoker.value] + self.acards[values.ljoker.value]
        sum = sum.astype(int)
        '''save the best point'''
        if x + self.dp[sum[0], sum[1], sum[2], sum[3], sum[4]] < self.steps:
            self.cardsfordfs = self.acards.copy()
            self.xfordfs = x
            self.best_strategy = copy.deepcopy(self.strategy)

        self.steps = min(self.steps, x + self.dp[sum[0], sum[1], sum[2], sum[3], sum[4]])

        '''顺子'''
        k = 0
        for i in range(values.two.value):#小王以下均可顺子
            if self.acards[i] == 0: k = 0 # 顺子中断
            else: # 通过回溯遍历不同长度的顺子
                k += 1
                if k >= 5:
                    for p in range(k, 4, -1):
                        # 出牌
                        for j in range(i, i - p, -1): self.acards[j] -= 1
                        self.strategy.append(list(range(i - p + 1, i + 1)))
                        #print(self.strategy)###
                        self.dfs(x + 1)
                        #回溯
                        for j in range(i, i - p, -1): self.acards[j] += 1
                        #print(self.strategy)###
                        del self.strategy[-1]
        '''间隔单顺子'''
        for q in range(2):
            k = 0
            for i in range(q, values.two.value, 2):
                if self.acards[i] == 0: k = 0 # 顺子中断
                else:
                    k += 1
                    if k >= 5:
                        for p in range(k, 4, -1):
                            for j in range(i, i - 2 * p, -2): self.acards[j] -= 1
                            self.strategy.append(list(range(i - 2 * p + 2, i + 2, 2)))
                            self.dfs(x + 1)
                            for j in range(i, i - 2 * p, -2): self.acards[j] += 1
                            del self.strategy[-1]
        '''双顺子'''
        k = 0
        for i in range(values.two.value):
            if self.acards[i] < 2: k = 0
            else:
                k += 1
                if k >= 3:
                    for p in range(k, 2, -1):
                        for j in range(i, i - p, -1): self.acards[j] -= 2
                        a = list(range(i - p + 1, i + 1))
                        b = a + a
                        b.sort()
                        self.strategy.append(b)
                        self.dfs(x + 1)
                        for j in range(i, i - p, -1): self.acards[j] += 2
                        del self.strategy[-1]
        '''间隔双顺子'''
        for q in range(2):
            k = 0
            for i in range(q, values.two.value, 2):
                if self.acards[i] < 2: k = 0
                else:
                    k += 1
                    if k >= 3:
                        for p in range(k, 2, -1):
                            for j in range(i, i - 2 * p, -2): self.acards[j] -= 2
                            a = list(range(i - 2 * p + 2, i + 2, 2))
                            b = a + a
                            b.sort()
                            self.strategy.append(b)
                            self.dfs(x + 1)
                            for j in range(i, i - 2 * p, -2): self.acards[j] += 2
                            del self.strategy[-1]
        '''三顺子'''
        k = 0
        for i in range(values.two.value):
            if self.acards[i] < 3: k = 0;
            else:
                k += 1
                if k >= 2:
                    for p in range(k, 1, -1):
                        for j in range(i, i - p, -1): self.acards[j] -= 3
                        a = list(range(i - p + 1, i + 1))
                        b = a + a + a
                        b.sort()
                        self.strategy.append(b)
                        self.dfs(x + 1)
                        for j in range(i, i - p, -1): self.acards[j] += 3
                        del self.strategy[-1]
        '''间隔三顺子'''
        for q in range(2):
            k = 0
            for i in range(q, values.two.value, 2):
                if self.acards[i] < 3: k = 0;
                else:
                    k += 1
                    if k >= 2:
                        for p in range(k, 1, -1):
                            for j in range(i, i - 2 * p, -2): self.acards[j] -= 3
                            a = list(range(i - 2 * p + 2, i + 2, 2))
                            b = a + a + a
                            b.sort()
                            #print(b, self.acards)
                            self.strategy.append(b)
                            
                            self.dfs(x + 1)
                            for j in range(i, i - 2 * p, -2): self.acards[j] += 3
                            del self.strategy[-1]
        '''统计牌数数组用于调用dp剪枝'''
        sum = np.zeros(5)
        for i in range(values.sjoker.value):
            if self.acards[i]:
                sum[int(self.acards[i] - 1)] += 1
        sum[4] = self.acards[values.sjoker.value] + self.acards[values.ljoker.value]
        sum = sum.astype(int)
        #print(self.acards)
        #print(sum)
        #print(self.dp[sum[0], sum[1], sum[2], sum[3], sum[4]])
        #if sum[4] < 0: 
        #    self.dp[0.02,0.02]
        '''save the best point'''
        if x + self.dp[sum[0], sum[1], sum[2], sum[3], sum[4]] < self.steps:
            self.cardsfordfs = self.acards.copy()
            self.xfordfs = x
            self.best_strategy = copy.deepcopy(self.strategy)

        self.steps = min(self.steps, x + self.dp[sum[0], sum[1], sum[2], sum[3], sum[4]])
              
    def dfs_for_else(self, x):
        if x > self.steps: return
        if self.flag == 1: return 
        '''带牌'''
        for i in range(values.sjoker.value):#枚举小王以下的牌可以作为带牌的主体
            if self.cardsfordfs[i] == 3:
                self.cardsfordfs[i] -= 3 # 减去带牌
                a = [i] * 3
                self.strategy.append(a)###
                #print(self.strategy)
                for j in range(values.ljoker.value + 1): # 带单张
                    if self.cardsfordfs[j] == 0: continue # 无牌(无需考虑相同)
                    self.cardsfordfs[j] -= 1 # 出牌
                    self.strategy[-1].append(j)
                    #print(self.strategy)###
                    self.dfs_for_else(x + 1)
                    self.cardsfordfs[j] += 1 # 回溯
                    #print(self.strategy)###
                    del self.strategy[-1][-1]
                    
                for j in range(values.sjoker.value): # 带一对
                    if self.cardsfordfs[j] <= 1: continue
                    self.cardsfordfs[j] -= 2
                    a = [j] * 2
                    self.strategy[-1].extend(a)
                    #print(self.strategy)
                    self.dfs_for_else(x + 1)
                    self.cardsfordfs[j] += 2
                    del self.strategy[-1][-2:]
                self.cardsfordfs[i] += 3 # 回溯
                del self.strategy[-1]
            elif self.cardsfordfs[i] == 4: # 四张也可以选择三带
                self.cardsfordfs[i] -= 3 # 先3带
                a = [i] * 3
                self.strategy.append(a)
                #print(self.strategy)
                for j in range(values.ljoker.value + 1): # 带单张
                    if (self.cardsfordfs[j] == 0) or (j == i): continue # 无牌或相同
                    self.cardsfordfs[j] -= 1 # 出牌
                    self.strategy[-1].append(j)
                    #print(self.strategy)
                    self.dfs_for_else(x + 1)
                    self.cardsfordfs[j] += 1 # 回溯
                    del self.strategy[-1][-1]
                for j in range(values.sjoker.value): # 带一对
                    if self.cardsfordfs[j] <= 1 : continue
                    self.cardsfordfs[j] -= 2 #出对子
                    a = [j] * 2
                    self.strategy[-1].extend(a)
                    #print(self.strategy)
                    self.dfs_for_else(x + 1)
                    self.cardsfordfs[j] += 2 # 回溯
                    del self.strategy[-1][-2:]
                self.cardsfordfs[i] += 3 # 回溯
                del self.strategy[-1]

                self.cardsfordfs[i] -= 4 # 四张带(包含炸弹)
                a = [i] * 4
                self.strategy.append(a)
                
                for j in range(values.ljoker.value + 1): # 两个单张
                    if self.cardsfordfs[j] == 0: continue
                    self.cardsfordfs[j] -= 1 # 出第一张牌
                    self.strategy[-1].append(j)
                    for m in range(values.ljoker.value + 1): # 另一个单张
                        if (self.cardsfordfs[m] == 0) or (j == m): continue
                        self.cardsfordfs[m] -= 1 #出第二张牌
                        self.strategy[-1].append(m)
                        self.dfs_for_else(x + 1)
                        self.cardsfordfs[m] += 1 # 回溯
                        del self.strategy[-1][-1]
                    self.cardsfordfs[j] += 1 # 回溯
                    del self.strategy[-1][-1]
                for j in range(values.sjoker.value): # 带两个对子
                    if self.cardsfordfs[j] <= 1: continue
                    self.cardsfordfs[j] -= 2
                    a = [j] * 2
                    self.strategy[-1].extend(a)
                    for m in range(values.sjoker.value):
                        if (self.cardsfordfs[m] <= 1) or (j == m): continue
                        self.cardsfordfs[m] -= 2
                        a = [m] * 2
                        self.strategy[-1].extend(a)
                        self.dfs_for_else(x + 1)
                        self.cardsfordfs[m] += 2
                        del self.strategy[-1][-2:]
                    self.cardsfordfs[j] += 2
                    del self.strategy[-1][-2:]
                self.cardsfordfs[i] += 4
                del self.strategy[-1]
        
        count = 0
        for i in range(values.sjoker.value):
            if self.cardsfordfs[i]: 
                x += 1
                count += 1
                a = [i] * int(self.cardsfordfs[i])
                self.strategy.append(a)
        if (self.cardsfordfs[values.sjoker.value]) or (self.cardsfordfs[values.ljoker.value]): 
            x += 1
            count += 1
            if (self.cardsfordfs[values.sjoker.value]) and (self.cardsfordfs[values.ljoker.value]):
                self.strategy.append([values.sjoker.value, values.ljoker.value])
            elif self.cardsfordfs[values.sjoker.value]:
                self.strategy.append([values.sjoker.value])
            else:
                self.strategy.append([values.ljoker.value])
        #if self.steps > x: 
            #print(x)
            #print(self.strategy)
        #self.steps = min(self.steps, x)
        if x == self.steps:
            self.flag = 1
            self.best_strategy.extend(copy.deepcopy(self.strategy))
        if count: del self.strategy[-count:]
        
    def init_dp(self):
        '''dp[i, j, k, z, l] -> times[1, 2, 3, 4, joker]'''
        self.dp[:, :, :, :, :] = math.inf
        self.dp[0, 0, 0, 0, 0] = 0
        #print(dp)
        for z in range(14):
            for k in range(14):
                for i in range(14):
                    for j in range(14):
                        for l in range(3):
                            x = 2019011455
                            if i > 0: x = min(x, self.dp[i - 1, j, k, z ,l] + 1)
                            if j > 0: x = min(x, self.dp[i, j - 1, k, z, l] + 1)
                            if k > 0: x = min(x, self.dp[i, j, k - 1, z, l] + 1)
                            if z > 0: x = min(x, self.dp[i ,j ,k ,z - 1, l] + 1)
                            if l > 0: x = min(x, self.dp[i, j, k, z, l - 1] + 1)
                            if l > 1: x = min(x, self.dp[i, j, k, z, l - 2] + 1)
                            '''三带一'''
                            if i > 0 and k > 0: x = min(x, self.dp[i - 1, j, k - 1, z, l] + 1)
                            if l > 0 and k > 0: x = min(x, self.dp[i, j, k - 1, z, l - 1] + 1)
                            '''三带二'''
                            if j > 0 and k > 0: x = min(x, self.dp[i, j - 1, k - 1, z, l] + 1)
                            '''四带二'''
                            if i > 1 and z > 0: x = min(x, self.dp[i - 2, j, k, z - 1, l] + 1)
                            if i > 0 and z > 0 and l > 0: x = min(x, self.dp[i - 1, j, k, z - 1, l - 1]  +1)
                            if l > 1 and z > 0: x = min(x, self.dp[i, j, k, z - 1, l - 2] + 1)
                            if j > 0 and z > 0: x = min(x, self.dp[i, j - 1, k, z - 1, l] + 1)
                            if j > 1 and z > 0: x = min(x, self.dp[i, j - 2, k, z - 1, l] + 1)
                            if z > 1: x = min(x, self.dp[i, j, k, z - 2, l] + 1)
                            '''*拆牌*'''
                            if z > 0: x = min(x, self.dp[i + 1, j, k + 1, z - 1, l])
                            if k > 0: x = min(x, self.dp[i + 1, j + 1, k - 1, z, l])
                            self.dp[i, j, k, z, l] = min(x, self.dp[i, j, k ,z, l])


if __name__ == "__main__":

    n = 100
    l = 30
    s = single_game(l)
    #starttime = datetime.datetime.now()
    #for i in tqdm(range(n)):
    #    s.renew_cards(l)
    print("cards number = ", l)
    #endtime = datetime.datetime.now()
    #time = (endtime - starttime).microseconds / n
    #print("%d average run time = " % l, (time),  "ms")


