import numpy as np
import datetime
import copy
from tqdm import tqdm
import math
import os

from numpy.lib.function_base import append
from utils import card, hand_cards, values, dhand_cards
from game import single_game

class dgame(object):
    def __init__(self, n1 = 17, n2 = 17):
        '''类内变量'''
        self.cnum1 = n1
        self.cnum2 = n2
        '''手牌'''
        self.hcards = dhand_cards(n1, n2)

        self.player1 = player(n1, self.hcards.mcard1)
        self.player2 = player(n2, self.hcards.mcard2)

        
    def start_game(self, n1 = 17, n2 = 17):
        '''
        depend on the mcards1/2 to start the game
        player1 will take first
        '''
        
        if n1 != self.cnum1 or n2 != self.cnum2:
            print("...", end = '')
            self.cnum1 = n1
            self.cnum2 = n2
            self.hcards = dhand_cards(n1, n2)
            self.player1 = player(n1, self.hcards.mcard1)
            self.player2 = player(n2, self.hcards.mcard2)
        
        print(end = '\r')
        print('                                                 ', end = '\r')
        os.system('pause')
        print("Cards :   [3, 4, 5, 6, 7, 8, 9,10, J, Q, K, A, 2,sJ,lJ]")
        print("Player 1:", list(map(int, self.player1.acards.tolist())), self.player1.cnum_remain)
        print("Player 2:", list(map(int, self.player2.acards.tolist())), self.player2.cnum_remain)
        print("-----------------------Start Game----------------------")
        output = []
        last_output = [1]
        while len(self.player1.mcards) != 0 and len(self.player2.mcards) != 0:
            print("\n")
            print("-------------------Round for Player 1------------------")
            print("Cards :   [3, 4, 5, 6, 7, 8, 9,10, J, Q, K, A, 2,sJ,lJ]")
            print("Player 1:", list(map(int, self.player1.acards.tolist())), self.player1.cnum_remain)
            last_output = output
            output = self.player1.play_card(last_output)
            print("Player 1 play cards:", output)
            print("Player 1:", list(map(int, self.player1.acards.tolist())), self.player1.cnum_remain)
            if len(self.player1.mcards) == 0 or len(self.player2.mcards) == 0: break
            print("\n")
            print("-------------------Round for Player 2------------------")
            print("Cards :   [3, 4, 5, 6, 7, 8, 9,10, J, Q, K, A, 2,sJ,lJ]")
            print("Player 2:", list(map(int, self.player2.acards.tolist())), self.player2.cnum_remain)
            last_output = output
            output = self.player2.play_card(last_output)
            print("Player 2 play cards:", output)
            print("Player 2:", list(map(int, self.player2.acards.tolist())), self.player2.cnum_remain)
        print("\n")
        if len(self.player1.mcards) == 0:
            print("----------------------Player 1 Win---------------------")
        elif len(self.player2.mcards) == 0:
            print("----------------------Player 2 Win---------------------")
        print("------------------------Game End-----------------------")
     
class player(object):
    def __init__(self, cnum = 0, mcards = []):
        '''
        cnum = the card number for this player
        mcards = [] the cards for this player
        '''
        '''total cards'''
        self.cnum = cnum
        '''cards in hand'''
        self.cnum_remain = cnum
        '''mcards with <card>'''
        self.mcards = mcards
        '''acards with card in array'''
        self.acards = self.card_in_np(self.mcards)
        '''single game for strategy'''
        self.sg = single_game(cnum)

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
        '''
        the core function to deal with the input cards
        e = [<card>, <card>, ...]
        return [<card>, <card>, ...]
        '''
        ret = []
        ecards = self.card_to_array(e)
        self.cnum_remain = len(self.mcards)
        if len(self.mcards) == 0: return []
        if len(ecards) == 0:
            self.sg.set_cards_renew(self.cnum, self.mcards)
            str = self.sg.best_strategy[0]
            for i in range(len(str)):
                self.acards[str[i]] -= 1
            flag = 0
            while flag < len(str):
                flag0 = 0
                for i in range(len(self.mcards)):
                    if self.mcards[i - flag0].value.value == str[flag]:
                        ret.append(self.mcards[i - flag0])
                        del self.mcards[i - flag0]
                        flag += 1
                        flag0 += 1
                        if flag == len(str): break
            self.cnum_remain = len(self.mcards)
            return ret
        elif len(ecards) == 1:
            '''找到有单牌的牌号'''
            x = np.where(self.acards == 1)[0]
            '''出最小的一张'''
            for i in range(len(x)):
                if x[i] > ecards[0]:
                    self.acards[x[i]] -= 1
                    for j in range(len(self.mcards)):
                        if self.mcards[j].value.value == x[i]:
                            ret.append(self.mcards[j])
                            del self.mcards[j]
                            break
                    self.cnum_remain = len(self.mcards)
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
                    self.cnum_remain = len(self.mcards)
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
                    self.cnum_remain = len(self.mcards)
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
                        self.cnum_remain = len(self.mcards)
                        return ret
                    else: continue
            else: # 是三带一
                '''找到有单牌的牌号'''
                x = np.where(self.acards == 1)[0]
                '''出最小的一张'''
                #无单牌可带
                if len(x) == 0: 
                    self.cnum_remain = len(self.mcards)
                    return []
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
                    if flag == 3: break
                #找到或没找到可出的3牌
                if ret == []: 
                    self.cnum_remain = len(self.mcards)
                    return []
                else:
                    '''找到有单牌的牌号'''
                    x = np.where(self.acards == 1)[0]
                    '''出最小的一张'''
                    #无单牌可带
                    if len(x) == 0: 
                        self.cnum_remain = len(self.mcards)
                        return []
                    else:
                        self.acards[x[0]] -= 1
                        for j in range(len(self.mcards)):
                            if self.mcards[j].value.value == x[0]:
                                ret.append(self.mcards[j])
                                del self.mcards[j]
                                break
                        self.cnum_remain = len(self.mcards)
                        return ret
        elif len(ecards) == 5:
            if ecards[0] == ecards[1]: # 是三带二
                '''找到有单牌的牌号'''
                x = np.where(self.acards == 2)[0]
                '''出最小的一对'''
                #无对牌可带
                if len(x) == 0: 
                    self.cnum_remain = len(self.mcards)
                    return []
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
                    if flag == 3: break
                #找到或没找到可出的3牌
                if ret == []: 
                    self.cnum_remain = len(self.mcards)
                    return []
                else:
                    '''找到有单牌的牌号'''
                    x = np.where(self.acards == 2)[0]
                    '''出最小的一对'''
                    #无对牌可带
                    if len(x) == 0: 
                        self.cnum_remain = len(self.mcards)
                        return []
                    else:
                        self.acards[x[0]] -= 2
                        flag = 0
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[0]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 2: break
                        self.cnum_remain = len(self.mcards)
                        return ret
            elif ecards[0] + 1 == ecards[1]: # 是单顺子
                k = 0
                for i in range(ecards[0] + 1, values.two.value):#2以下均可顺子
                    if self.acards[i] == 0: k = 0 # 顺子中断或小于目的牌
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
                            self.cnum_remain = len(self.mcards)
                            return ret
            elif ecards[0] + 2 == ecards[1]: # 是间隔单顺子
                for q in range(1, 3):
                    k = 0
                    for i in range(ecards[0] + q, values.two.value, 2):#2以下均可顺子
                        if self.acards[i] == 0: k = 0 # 顺子中断或小于目的牌
                        else: # 通过回溯遍历不同长度的顺子
                            k += 1
                            if k == 5:# 找到了符合要求的最小顺子
                                # 出牌
                                for j in range(i, i - 10, -2):self.acards[j] -= 1
                                flag = 0
                                for j in range(len(self.mcards)):
                                    if self.mcards[j - flag].value.value == (i - 8 + flag * 2):
                                        ret.append(self.mcards[j - flag])
                                        del self.mcards[j - flag]
                                        flag += 1
                                        if flag == 5: break
                                self.cnum_remain = len(self.mcards)
                                return ret
        elif ecards[0] == ecards[1] == ecards[2] == ecards[3]: # 
            if len(ecards) == 6: # 四带二
                '''找到有单牌的牌号'''
                x = np.where(self.acards == 1)[0]
                '''出最小的两个'''
                #无对牌可带
                if len(x) <= 1: 
                    self.cnum_remain = len(self.mcards)
                    return []
                '''找到有4牌的牌号'''
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
                    else: continue
                    if flag == 4: break
                #找到或没找到可出的4牌
                if ret == []: 
                    self.cnum_remain = len(self.mcards)
                    return []
                else:
                    '''找到有单牌的牌号'''
                    x = np.where(self.acards == 1)[0]
                    '''出最小的两个'''
                    #无对牌可带
                    if len(x) <= 1: 
                        self.cnum_remain = len(self.mcards)
                        return []
                    else:
                        self.acards[x[0]] -= 1
                        self.acards[x[1]] -= 1
                        flag = 0
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[flag]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 2: break
                        self.cnum_remain = len(self.mcards)
                        return ret
            else: # 四带二对
                '''找到有2张的牌号'''
                x = np.where(self.acards == 2)[0]
                '''出最小的两对'''
                #无对牌可带
                if len(x) <= 1: 
                    self.cnum_remain = len(self.mcards)
                    return []
                '''找到有4牌的牌号'''
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
                    else: continue
                    if flag == 4: break
                #找到或没找到可出的4牌
                if ret == []: 
                    self.cnum_remain = len(self.mcards)
                    return []
                else:
                    '''找到有2张的牌号'''
                    x = np.where(self.acards == 2)[0]
                    '''出最小的两对'''
                    #无对牌可带
                    if len(x) <= 1: 
                        self.cnum_remain = len(self.mcards)
                        return []
                    else:
                        self.acards[x[0]] -= 2
                        self.acards[x[1]] -= 2
                        flag = 0
                        for j in range(len(self.mcards)):
                            if self.mcards[j - flag].value.value == x[flag // 2]:
                                ret.append(self.mcards[j - flag])
                                del self.mcards[j - flag]
                                flag += 1
                                if flag == 4: break
                        self.cnum_remain = len(self.mcards)
                        return ret
        else: # 单双三间不间隔顺子
            length = len(ecards)
            if ecards[0] + 1 == ecards[1]: # 单顺子
                k = 0
                for i in range(ecards[0] + 1, values.two.value):#2以下均可顺子
                    if self.acards[i] == 0: k = 0 # 顺子中断或小于目的牌
                    else: # 通过回溯遍历不同长度的顺子
                        k += 1
                        if k == length:# 找到了符合要求的最小顺子
                            # 出牌
                            for j in range(i, i - length, -1):self.acards[j] -= 1
                            flag = 0
                            for j in range(len(self.mcards)):
                                if self.mcards[j - flag].value.value == (i - length + 1 + flag):
                                    ret.append(self.mcards[j - flag])
                                    del self.mcards[j - flag]
                                    flag += 1
                                    if flag == length: break
                            self.cnum_remain = len(self.mcards)
                            return ret
            elif ecards[0] + 2 == ecards[1]: # 间隔单顺子
                for q in range(1, 3):
                    k = 0
                    for i in range(ecards[0] + q, values.two.value, 2):#2以下均可顺子
                        if self.acards[i] == 0: k = 0 # 顺子中断或小于目的牌
                        else: # 通过回溯遍历不同长度的顺子
                            k += 1
                            if k == length:# 找到了符合要求的最小顺子
                                # 出牌
                                for j in range(i, i - length * 2, -2):self.acards[j] -= 1
                                flag = 0
                                for j in range(len(self.mcards)):
                                    if self.mcards[j - flag].value.value == (i - 2 * (length - 1) + flag * 2):
                                        ret.append(self.mcards[j - flag])
                                        del self.mcards[j - flag]
                                        flag += 1
                                        if flag == length: break
                                self.cnum_remain = len(self.mcards)
                                return ret
            elif ecards[0] == ecards[1] and ecards[1] + 1 == ecards[2]: # 双顺子
                k = 0
                for i in range(ecards[0] + 1, values.two.value):#2以下均可顺子
                    if self.acards[i] < 2: k = 0 # 顺子中断或小于目的牌
                    else: # 通过回溯遍历不同长度的顺子
                        k += 1
                        if k == length / 2:# 找到了符合要求的最小顺子
                            # 出牌
                            for j in range(i, i - length // 2, -1):self.acards[j] -= 2
                            flag = 0
                            for j in range(len(self.mcards)):
                                if self.mcards[j - flag].value.value == (i - length // 2 + 1 + flag // 2):
                                    ret.append(self.mcards[j - flag])
                                    del self.mcards[j - flag]
                                    flag += 1
                                    if flag == length: break
                            self.cnum_remain = len(self.mcards)
                            return ret
            elif ecards[0] == ecards[1] and ecards[1] + 2 == ecards[2]: # 间隔双顺子
                for q in range(1, 3):
                    k = 0
                    for i in range(ecards[0] + q, values.two.value, 2):#2以下均可顺子
                        if self.acards[i] < 2: k = 0 # 顺子中断或小于目的牌
                        else: # 通过回溯遍历不同长度的顺子
                            k += 1
                            if k == length // 2:# 找到了符合要求的最小顺子
                                # 出牌
                                for j in range(i, i - length, -2):self.acards[j] -= 2
                                flag = 0
                                for j in range(len(self.mcards)):
                                    if self.mcards[j - flag].value.value == (i - (length - 2) + (flag // 2) * 2):
                                        ret.append(self.mcards[j - flag])
                                        del self.mcards[j - flag]
                                        flag += 1
                                        if flag == length: break
                                self.cnum_remain = len(self.mcards)
                                return ret
            elif ecards[0] == ecards[1] == ecards[2] and ecards[2] + 1 == ecards[3]: # 三顺子
                k = 0
                for i in range(ecards[0] + 1, values.two.value):#2以下均可顺子
                    if self.acards[i] < 3: k = 0 # 顺子中断或小于目的牌
                    else: # 通过回溯遍历不同长度的顺子
                        k += 1
                        if k == length / 3:# 找到了符合要求的最小顺子
                            # 出牌
                            for j in range(i, i - length // 3, -1):self.acards[j] -= 3
                            flag = 0
                            for j in range(len(self.mcards)):
                                if self.mcards[j - flag].value.value == (i - length // 3 + 1 + flag // 3):
                                    ret.append(self.mcards[j - flag])
                                    del self.mcards[j - flag]
                                    flag += 1
                                    if flag == length: break
                            self.cnum_remain = len(self.mcards)
                            return ret
            elif ecards[0] == ecards[1] == ecards[2] and ecards[2] + 2 == ecards[3]: # 间隔三顺子
                for q in range(1, 3):
                    k = 0
                    for i in range(ecards[0] + q, values.two.value, 2):#2以下均可顺子
                        if self.acards[i] < 3: k = 0 # 顺子中断或小于目的牌
                        else: # 通过回溯遍历不同长度的顺子
                            k += 1
                            if k == length // 3:# 找到了符合要求的最小顺子
                                # 出牌
                                for j in range(i, i - length // 3 * 2, -2):self.acards[j] -= 3
                                flag = 0
                                for j in range(len(self.mcards)):
                                    if self.mcards[j - flag].value.value == (i - (length // 3 - 1) * 2 + (flag // 3) * 2):
                                        ret.append(self.mcards[j - flag])
                                        del self.mcards[j - flag]
                                        flag += 1
                                        if flag == length: break
                                self.cnum_remain = len(self.mcards)
                                return ret
        self.cnum_remain = len(self.mcards)
        return []


if __name__ == "__main__":
    
    c1 = int(input("You want set the number of cards for player 1: "))
    c2 = int(input("You want set the number of cards for player 2: "))
    print("Initializing... this may take a few seconds", end = '...')
    d = dgame()
    x = 1
    while x == 1:
        d.start_game(c1, c2)
        inp = 0
        while inp != 'n' and inp != 'y':
            inp = input("Do you want to continue[y/n]:")
        if inp == 'n': x = 0
        else: 
            c1 = int(input("You want set the number of cards for player 1: "))
            c2 = int(input("You want set the number of cards for player 2: "))
        

    #os.system('pause')




