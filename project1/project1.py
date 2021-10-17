import numpy as np
import datetime
from card import card, hand_cards, values


class single_game(object):
    def __init__(self, n):
        #单人游戏总牌数
        self.cnum = n
        #单人游戏手牌
        self.cards = hand_cards(self.cnum).mcard
        #手牌分布数组
        self.acards = self.card_in_np()
        #最佳策略与最短步数
        self.strategy = {}
        self.steps = 15
        print(self.acards)
        self.dfs()
        print(self.steps)
    
    def card_in_np(self):
        acards = np.zeros(15)
        for i in range(len(self.cards)):
            acards[self.cards[i].value.value] += 1
        #print(acards)
        return acards
    
    def dfs(self, x = 0):
        #print(x)
        #print(self.acards)
        if x > self.steps: return
        #顺子
        k = 0
        for i in range(values.two.value):#小王以下均可顺子
            if self.acards[i] == 0: k = 0 # 顺子中断
            else: # 通过回溯遍历不同长度的顺子
                k += 1
                if k >= 5:
                    # 出牌
                    for j in range(i, i - k, -1): self.acards[j] -= 1
                    self.dfs(x + 1)
                    #回溯
                    for j in range(i, i - k, -1): self.acards[j] += 1
        #间隔单顺子
        k = 0
        for q in range(2):
            for i in range(q, values.two.value, 2):
                if self.acards[i] == 0: k = 0 # 顺子中断
                else:
                    k += 1
                    if k >= 5:
                        for j in range(i, i - 2 * k, -2): self.acards[j] -= 1
                        self.dfs(x + 1)
                        for j in range(i, i - 2 * k, -2): self.acards[j] += 1
        #双顺子
        k = 0
        for i in range(values.two.value):
            if self.acards[i] < 2: k = 0
            else:
                k += 1
                if k >= 3:
                    for j in range(i, i - k, -1): self.acards[j] -= 2
                    self.dfs(x + 2)
                    for j in range(i, i - k, -1): self.acards[j] += 2
        #间隔双顺子
        k = 0
        for q in range(2):
            for i in range(q, values.two.value, 2):
                if self.acards[i] < 2: k = 0
                else:
                    k += 1
                    if k >= 3:
                        for j in range(i, i - 2 * k, -2): self.acards[j] -= 2
                        self.dfs(x + 1)
                        for j in range(i, i - 2 * k, -2): self.acards[j] += 2
        #三顺子
        for i in range(values.two.value):
            if self.acards[i] < 3: k = 0;
            else:
                k += 1
                if k >= 2:
                    for j in range(i, i - k, -1): self.acards[j] -= 3
                    self.dfs(x + 1)
                    for j in range(i, i - k, -1): self.acards[j] -= 3
        #间隔三顺子
        for q in range(2):
            for i in range(q, values.two.value, 2):
                if self.acards[i] < 3: k = 0;
                else:
                    k += 1
                    if k >= 2:
                        for j in range(i, i - 2 * k, -2): self.acards[j] -= 3
                        self.dfs(x + 1)
                        for j in range(i, i - 2 * k, -2): self.acards[j] -= 3
        #带牌
        for i in range(values.sjoker.value):#枚举小王及以下的牌可以作为带牌的主体
            if self.acards[i] <= 3:
                if self.acards[i] <= 2: continue # 排除两张以下
                self.acards[i] -= 3 # 减去带牌
                for j in range(values.ljoker.value + 1): # 带单张
                    if self.acards[j] == 0: continue # 无牌(无需考虑相同)
                    self.acards[j] -= 1 # 出牌
                    self.dfs(x + 1)
                    self.acards[j] += 1 # 回溯
                for j in range(values.sjoker.value): # 带一对
                    if self.acards[j] <= 1: continue
                    self.acards[j] -= 2
                    self.dfs(x + 1)
                    self.acards[j] += 2
                self.acards[i] += 3 # 回溯
            else: # 四张也可以选择三带
                self.acards[i] -= 3 # 先3带
                for j in range(values.ljoker.value + 1): # 带单张
                    if (self.acards[j] == 0) or (j == i): continue # 无牌或相同
                    self.acards[j] -= 1 # 出牌
                    self.dfs(x + 1)
                    self.acards[j] += 1 # 回溯
                for j in range(values.sjoker.value): # 带一对
                    if self.acards[j] <= 1: continue
                    self.acards[j] -= 2 #出对子
                    self.dfs(x + 1)
                    self.acards[j] += 2 # 回溯
                self.acards[i] += 3 # 回溯

                self.acards[i] -= 4 # 四张带(包含炸弹)
                for j in range(values.ljoker.value + 1): # 两个单张
                    if self.acards[j] == 0: continue
                    self.acards[j] -= 1 # 出第一张牌
                    for m in range(values.ljoker.value + 1): # 另一个单张
                        if (self.acards[m] == 0) or (j == m): continue
                        self.acards[m] -= 1 #出第二张牌
                        self.dfs(x + 1)
                        self.acards[m] += 1 # 回溯
                    self.acards[j] += 1 # 回溯
                for j in range(values.sjoker.value): # 带两个对子
                    if self.acards[j] <= 1: continue
                    self.acards[j] -= 2
                    for m in range(values.sjoker.value):
                        if (self.acards[m] <= 1) or (j == m): continue
                        self.acards[m] -= 2
                        self.dfs(x + 1)
                        self.acards[m] += 2
                    self.acards[j] += 2
                self.acards[i] += 4
        for i in range(values.ljoker.value + 1):
            if self.acards[i]: x += 1
        if self.steps > x: print(x)
        self.steps = min(self.steps, x)
              
    

single_game(30)

