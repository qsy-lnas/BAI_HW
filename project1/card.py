from enum import Enum

class colors(Enum):
    # 黑桃 红桃 梅花 方片
    spade = 1
    heart = 2 
    club = 3
    diam = 4

class values(Enum):
    three = 1
    four = 2
    five = 3
    six = 4
    seven = 5
    eight = 6
    nine = 7
    ten = 8 
    J = 9
    Q = 1
    K = 11
    A = 12
    two = 13
    sjoker = 14
    ljoker = 15
    
class card(object):
    def __init__(self, color, value):
       self.color = colors(color)
       self.value = values(value)


""" c = card(4, 4)
print(c.color) """