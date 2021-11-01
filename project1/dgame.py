import numpy as np
import datetime
import copy
from tqdm import tqdm
import math

from numpy.lib.function_base import append
from card import card, hand_cards, values

class dgame(object):
    def __init__(self, n1  =17, n2 = 17):

        '''类内变量'''
        self.cnum1 = n1
        self.acards1 = []
        self.cards = []

        self.cnum2 = n2
        self.acards = []
        self.cards = []

        '''分别计算出牌策略'''
        



