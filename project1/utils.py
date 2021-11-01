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
        self.acards1 = []
        self.cards = []

        self.cnum2 = n2
        self.acards = []
        self.cards = []

if __name__ == "__main__":
    """ c = card(4, 4)
    b = card(2, 2)
    print(b) """