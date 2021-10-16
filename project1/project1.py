import numpy as np
from card import card, hand_cards


class single_game(object):
    def __init__(self, n):
        self.cnum = n
        self.cards = hand_cards(self.cnum).mcard
        self.strategy = {}
        self.acards = self.card_in_np()
        #print(self.cards)
    
    def card_in_np(self):
        cards = np.array(range(1, 16), np.zeros(15))
        print(cards)

        

