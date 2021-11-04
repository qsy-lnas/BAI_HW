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
       self.id = self.color.value * 15 + self.value.value

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

if __name__ == "__main__":
    d = dhand_cards(2,2)
    print(d.mcard1)