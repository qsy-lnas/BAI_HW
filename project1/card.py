import random
from enum import Enum

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
        self.random_card()
        self.sort_card()

    def random_card(self):
        pokers = []
        for i in range(1, 5):
            for j in range(13):
                c = card(i, j)
                pokers.append(c)
        c = card(0, 14)
        pokers.append(c)
        c = card(0, 15)
        pokers.append(c)
        random.shuffle(pokers)
        self.mcard = pokers[0:self.cnum]
        #print(self.mcard)

    def sort_card(self):
        self.mcard.sort()
        #print(self.mcard)

""" c = card(4, 4)
b = card(2, 2)
print(b) """