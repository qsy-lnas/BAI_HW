import random
from enum import Enum

import numpy as np
import copy


class colors(Enum):
    # 黑桃 红桃 梅花 方片
    none = 0
    spade = 1
    heart = 2 
    club = 3
    diam = 4

class values(Enum):
    three = 0
    four = 1
    five = 2
    six = 3
    seven = 4
    eight = 5
    nine = 6
    ten = 7
    J = 8
    Q = 9
    K = 10
    A = 11
    two = 12
    sjoker = 13
    ljoker = 14

rules = {
    'threecard': 3,
    'threeandone': 3,
    'threeandtwo': 3,
    'fourcard': 3,
    'fourandtwo': 4,
    'fourandfour': 4,
    'shunza': 5,
    'dshunza': 6,
    'tshunza': 7
}

    
class card(object):
    def __init__(self, color, value):
       self.color = colors(color)
       self.value = values(value)

    def __repr__(self):
        return "<card.{}.{}>".format(self.color.name, self.value.name)
    def __lt__(self, other):
        return self.value.value < other.value.value
    
    def __eq__(self, other):
        return self.value.value == other.value.value

class hand_cards(object):
    def __init__(self, n):
        self.cnum = n
        self.mcard = []
        #self.seed = 2019011455
        self.random_card()
        self.sort_card()

    def random_card(self):
        #random.seed(self.seed)
        pokers = []
        for i in range(1, 5):
            for j in range(values.sjoker.value):
                c = card(i, j)
                pokers.append(c)
        c = card(0, values.sjoker.value)
        pokers.append(c)
        c = card(0, values.ljoker.value)
        pokers.append(c)
        random.shuffle(pokers)
        self.mcard = pokers[0:self.cnum]
        #print(self.mcard)

    def sort_card(self):
        self.mcard.sort()
        #print(self.mcard)

class dhand_cards(object):
    def __init__(self, n1 = 17, n2 = 17):
        self.cnum1 = n1
        self.mcard1 = []

        self.cnum2 = n2
        self.mcard2 = []
        self.random_card()
        self.sort_card()

    def random_card(self):
        pokers = []
        '''make a full pokers'''
        for i in range(1, 5):
            for j in range(values.sjoker.value):
                c = card(i, j)
                pokers.append(c)
        c = card(0, values.sjoker.value)
        pokers.append(c)
        c = card(0, values.ljoker.value)
        pokers.append(c)
        '''shuffle the full pokers'''
        random.shuffle(pokers)
        self.mcard1 = pokers[0:self.cnum1]
        self.mcard2 = pokers[self.cnum1: self.cnum2 + self.cnum1]

    def sort_card(self):
        self.mcard1.sort()
        self.mcard2.sort()

class player(object):
    def __init__(self, cnum = 0, mcards = []):
        '''
        cnum = the card number for this player
        mcards = [] the cards for this player
        '''
        '''total cards'''
        self.cnum = cnum
        self.cnum_remain = cnum
        '''mcards with <card>'''
        self.mcards = mcards
        '''acards with card in array'''
        self.acards = self.card_in_np(self.mcards)

    def card_in_np(self, c):
        ''''change the card to a total array'''
        acards = np.zeros(15)
        for i in range(len(c)):
            acards[c[i].value.value] += 1
        #print(acards)
        return acards

    def card_to_array(self, c):
        ret = []
        for i in range(len(c)):
            ret.append(c[i].value.value)
        return ret

    def play_card(self, e):
        ret = []
        ecards = self.card_to_array(e)
        if len(ecards) == 1:
            '''找到有单牌的牌号'''
            x = np.where(self.acards == 1)[0]
            '''出最小的一张'''
            for i in range(len(x)):
                if x[i] > ecards[0]:
                    self.acards[x[i]] -= 1
                    for j in range(len(self.mcards)):
                        if self.mcards[j].value.value == x[i]:
                            ret = self.mcards[j]
                            del self.mcards[j]
                            break
                    return ret
                else: continue
        elif len(ecards) == 2:
            '''找到有对子的牌号'''
            x = np.where(self.acards == 2)[0]
            '''出牌'''
            for i in range(len(x)):
                if x[i] > ecards[0]:
                    self.acards[x[i]] -= 2
                    flag = 0 #记录删除了几张牌
                    for j in range(len(self.mcards)):
                        if self.mcards[j - flag].value.value == x[i]:
                            ret.append(self.mcards[j - flag])
                            del self.mcards[j - flag]
                            flag += 1
                            if flag == 2: break
                    return ret
                else: continue
        elif len(ecards) == 3:
            '''找到有三牌的牌号'''
            x = np.where(self.acards == 3)[0]
            '''出牌'''
            for i in range(len(x)):
                if x[i] > ecards[0]:
                    self.acards[x[i]] -= 3
                    flag = 0 #记录删除了几张牌
                    for j in range(len(self.mcards)):
                        if self.mcards[j - flag].value.value == x[i]:
                            ret.append(self.mcards[j - flag])
                            del self.mcards[j - flag]
                            flag += 1
                            if flag == 3: break
                    return ret
                else: continue
        elif len(ecards) == 4:
            if ecards[0] == ecards[3]: # 是炸弹
                '''找到有四牌的牌号'''
                x = np.where(self.acards == 4)[0]
                '''出牌'''
                for i in range(len(x)):
                    if x[i] > ecards[0]:
                        self.acards[x[i]] -= 4
                        flag = 0 #记录删除了几张牌
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[i]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 4: break
                        return ret
                    else: continue
            else: # 是三带一
                '''找到有三牌的牌号'''
                x = np.where(self.acards == 3)[0]
                '''出牌'''
                for i in range(len(x)):
                    if x[i] > ecards[0]:
                        self.acards[x[i]] -= 3
                        flag = 0 #记录删除了几张牌
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[i]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 3: break
                    else: continue
                #找到或没找到可出的3牌
                if ret == []: return []
                else:
                    '''找到有单牌的牌号'''
                    x = np.where(self.acards == 1)[0]
                    '''出最小的一张'''
                    #无单牌可带
                    if len(x) == 0: return []
                    else:
                        self.acards[x[0]] -= 1
                        for j in range(len(self.mcards)):
                            if self.mcards[j].value.value == x[0]:
                                ret.append(self.mcards[j])
                                del self.mcards[j]
                                break
                        return ret
        elif len(ecards) == 5:
            if ecards[0] == ecards[1]: # 是三带二
                '''找到有三牌的牌号'''
                x = np.where(self.acards == 3)[0]
                '''出牌'''
                for i in range(len(x)):
                    if x[i] > ecards[0]:
                        self.acards[x[i]] -= 3
                        flag = 0 #记录删除了几张牌
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[i]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 3: break
                    else: continue
                #找到或没找到可出的3牌
                if ret == []: return []
                else:
                    '''找到有单牌的牌号'''
                    x = np.where(self.acards == 2)[0]
                    '''出最小的一对'''
                    #无对牌可带
                    if len(x) == 0: return []
                    else:
                        self.acards[x[0]] -= 2
                        flag = 0
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[0]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 2: break
                        return ret
            elif ecards[0] + 1 == ecards[1]: # 是单顺子
                k = 0
                for i in range(ecards[0], values.two.value):#2以下均可顺子
                    if self.acards[i] == 0 or i <= ecards[0]: k = 0 # 顺子中断或小于目的牌
                    else: # 通过回溯遍历不同长度的顺子
                        k += 1
                        if k == 5:# 找到了符合要求的最小顺子
                            # 出牌
                            for j in range(i, i - 5, -1):self.acards[j] -= 1
                            flag = 0
                            for j in range(len(self.mcards)):
                                if self.mcards[j - flag].value.value == (i - 4 + flag):
                                    ret.append(self.mcards[j - flag])
                                    del self.mcards[j - flag]
                                    flag += 1
                                    if flag == 5: break
                            return ret



            elif ecards[0] + 2 == ecards[1]: # 是间隔单顺子

                pass

        return []


if __name__ == "__main__":
    #a = hand_cards(3)
    b = hand_cards(25)
    c = player(25, b.mcard)

    print(c.card_to_array(b.mcard))
    print(c.play_card([card(1, 0), card(1, 2), card(1, 3), card(1, 4), card(1, 5)]))
    print(c.card_to_array(b.mcard))
