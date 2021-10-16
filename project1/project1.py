import numpy as np
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
        self.steps = 0

        #print(self.cards)
    
    def card_in_np(self):
        acards = np.zeros(15)
        k = 0
        for i in range(len(self.cards)):
            acards[self.cards[i].value.value] += 1
        #print(acards)
        return acards
    
    def dfs(self, x = 0):
        if x > self.steps: return
        #顺子
        k = 0
        for i in range(values.two.value):#小王以下均可顺子
            if self.acards[i] == 0: k = 0 # 顺子中断
            else:
                k += 1
                if k > 5:
                    for 
                
    

#single_game(30)

