import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QWidget, QInputDialog, QPushButton, QApplication, QMessageBox, QTextBrowser
import Ui_main
from game import single_game
from utils import card, values 

Ui_MainWindow = Ui_main.Ui_MainWindow

class MyUi(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        创建主窗口的Qt对象，即斗地主应用程序
        '''
        self.sg = single_game()
        self.totalcard = [0] * 73
        self.playedcard = [0] * 25
        self.cardnum = [0] * 15
        self.warned = 0
        QtWidgets.QMainWindow.__init__(self)
        '''主界面对象初始化'''
        Ui_MainWindow.__init__(self)
        '''配置主界面对象'''
        self.setupUi(self)
        self.my_init()
        self.card_hide()
        self.textBrowser.setText("Welcome to the program, press Start Search to start")

    def my_init(self):
        self.totalcard[13] = self.card_13
        self.totalcard[14] = self.card_14
        self.totalcard[15] = self.card_15
        self.totalcard[16] = self.card_16
        self.totalcard[17] = self.card_17
        self.totalcard[18] = self.card_18
        self.totalcard[19] = self.card_19
        self.totalcard[20] = self.card_20
        self.totalcard[21] = self.card_21
        self.totalcard[22] = self.card_22
        self.totalcard[23] = self.card_23
        self.totalcard[24] = self.card_24
        self.totalcard[25] = self.card_25
        self.totalcard[26] = self.card_26
        self.totalcard[27] = self.card_27
        self.totalcard[30] = self.card_30
        self.totalcard[31] = self.card_31
        self.totalcard[32] = self.card_32
        self.totalcard[33] = self.card_33
        self.totalcard[34] = self.card_34
        self.totalcard[35] = self.card_35
        self.totalcard[36] = self.card_36
        self.totalcard[37] = self.card_37
        self.totalcard[38] = self.card_38
        self.totalcard[39] = self.card_39
        self.totalcard[40] = self.card_40
        self.totalcard[41] = self.card_41
        self.totalcard[42] = self.card_42
        self.totalcard[45] = self.card_45
        self.totalcard[46] = self.card_46
        self.totalcard[47] = self.card_47
        self.totalcard[48] = self.card_48
        self.totalcard[49] = self.card_49
        self.totalcard[50] = self.card_50
        self.totalcard[51] = self.card_51
        self.totalcard[52] = self.card_52
        self.totalcard[53] = self.card_53
        self.totalcard[54] = self.card_54
        self.totalcard[55] = self.card_55
        self.totalcard[56] = self.card_56
        self.totalcard[57] = self.card_57
        self.totalcard[60] = self.card_60
        self.totalcard[61] = self.card_61
        self.totalcard[62] = self.card_62
        self.totalcard[63] = self.card_63
        self.totalcard[64] = self.card_64
        self.totalcard[65] = self.card_65
        self.totalcard[66] = self.card_66
        self.totalcard[67] = self.card_67
        self.totalcard[68] = self.card_68
        self.totalcard[69] = self.card_69
        self.totalcard[70] = self.card_70
        self.totalcard[71] = self.card_71
        self.totalcard[72] = self.card_72
        self.cardnum[0] = self.spinBox_0
        self.cardnum[1] = self.spinBox_1
        self.cardnum[2] = self.spinBox_2
        self.cardnum[3] = self.spinBox_3
        self.cardnum[4] = self.spinBox_4
        self.cardnum[5] = self.spinBox_5
        self.cardnum[6] = self.spinBox_6
        self.cardnum[7] = self.spinBox_7
        self.cardnum[8] = self.spinBox_8
        self.cardnum[9] = self.spinBox_9
        self.cardnum[10] = self.spinBox_10
        self.cardnum[11] = self.spinBox_11
        self.cardnum[12] = self.spinBox_12
        self.cardnum[13] = self.spinBox_13
        self.cardnum[14] = self.spinBox_14

    def card_hide(self):
        for i in range(len(self.totalcard)):
            if self.totalcard[i] == 0: continue
            self.totalcard[i].setVisible(False)
        self.PlaycardButton.setVisible(False)
        self.PlaycardButton.setEnabled(False)
        self.playedcard = [0] * 25

    def card_show(self):
        for i in range(len(self.sg.cards)):
            self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
            self.totalcard[self.sg.cards[i].id].raise_()
            self.totalcard[self.sg.cards[i].id].setVisible(True)

    def show_single_card(self, id, i):
        self.totalcard[id].move(10 + 18 * i, 330)
        self.totalcard[id].raise_()
        self.totalcard[id].setVisible(True)

    def search_begin(self):
        self.CalpushButton.setEnabled(False)
        type = self.comboBox.currentIndex()
        self.card_hide()
        '''no score; set num'''
        if type == 0: 
            self.warned = 0
            cnum = self.CardNumBox.value()
            self.textBrowser.setText("Randomly generate %d cards, press Play Card to continue"%cnum)
            self.sg.renew_cards(cnum)
            self.textBrowser.append("The min steps is %d"%int(self.sg.steps))
            for i in range(cnum):
                self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
                self.totalcard[self.sg.cards[i].id].raise_()
                self.totalcard[self.sg.cards[i].id].setVisible(True)
        '''no score; set cards'''
        if type == 1:
            cards = [0] * 15
            mcards = []
            for i in range(values.ljoker.value + 1):
                cards[i] = self.cardnum[i].value()
            '''2 以下'''
            for i in range(values.sjoker.value): 
                for j in range(cards[i]):
                    a = card(j + 1, i)
                    mcards.append(a)
            '''单独处理大小王'''
            for i in range(values.sjoker.value, values.ljoker.value + 1):
                if cards[i] == 1: 
                    a = card(0, i)
                    mcards.append(a)
            if len(mcards) > 24 and self.warned == 0:
                
                self.textBrowser.setText("Warning: You choose too many cards")
                self.textBrowser.append("If you still want to use these cards, click one more time to search")
                self.warned = 1
                self.CalpushButton.setEnabled(True)
                return
            if self.warned == 1: self.warned = 0
            self.textBrowser.setText("you select %d cards, press Play Card to continue"%len(mcards))
            self.sg.set_cards_renew(len(mcards), mcards)
            self.textBrowser.append("The min steps is %d"%int(self.sg.steps))
            #print(self.sg.cards)
            #print(self.sg.acards)
            #print(self.sg.best_strategy)
            for i in range(len(mcards)):
                self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
                self.totalcard[self.sg.cards[i].id].raise_()
                self.totalcard[self.sg.cards[i].id].setVisible(True)
        '''with score; set num'''
        if type == 2:
            self.warned = 0
            cnum = self.CardNumBox.value()
            self.sg.renew_cards_value(cnum)
            self.textBrowser.setText("Randomly generate %d cards, press Play Card to continue"%cnum)
            self.textBrowser.append("The max score is %.2f" % self.sg.minvalue)
            for i in range(cnum):
                self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
                self.totalcard[self.sg.cards[i].id].raise_()
                self.totalcard[self.sg.cards[i].id].setVisible(True)
        '''with score; set cards'''
        if type == 3: 
            cards = [0] * 15
            mcards = []
            for i in range(values.ljoker.value + 1):
                cards[i] = self.cardnum[i].value()
            '''2 以下'''
            for i in range(values.sjoker.value): 
                for j in range(cards[i]):
                    a = card(j + 1, i)
                    mcards.append(a)
            '''单独处理大小王'''
            for i in range(values.sjoker.value, values.ljoker.value + 1):
                if cards[i] == 1: 
                    a = card(0, i)
                    mcards.append(a)
            if len(mcards) > 20 and self.warned == 0:
                self.textBrowser.setText("##--!--!--Warning--!--!--## You choose too many cards")
                self.textBrowser.append("If you still want to use these cards, click one more time to search")
                self.warned = 1
                self.CalpushButton.setEnabled(True)
                return
            if self.warned == 1: self.warned = 0
            self.textBrowser.setText("you select %d cards, press Play Card to continue"%len(mcards))
            self.sg.set_cards_renew_value(len(mcards), mcards)
            self.textBrowser.append("The max score is %d"%int(self.sg.minvalue))
            for i in range(len(mcards)):
                self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
                self.totalcard[self.sg.cards[i].id].raise_()
                self.totalcard[self.sg.cards[i].id].setVisible(True)         



        self.CalpushButton.setEnabled(True)
        self.PlaycardButton.setVisible(True)
        self.PlaycardButton.setEnabled(True)
            
    def play_card(self):
        if len(self.sg.cards) == 0: 
            self.textBrowser.append("----Play Card finished----")
            self.card_hide()
            return 
        i = 0
        while type(self.playedcard[i]) != int:
            self.totalcard[self.playedcard[i].id].setVisible(False)
            self.playedcard[i] = 0
            i += 1
        a = self.sg.best_strategy[0]
        flag = 0
        while flag < len(a):
            flag0 = 0
            for i in range(len(self.sg.cards)):
                if self.sg.cards[i - flag0].value.value == a[flag]:
                    self.show_single_card(self.sg.cards[i - flag0].id, flag)
                    self.playedcard[flag] = self.sg.cards[i - flag0]
                    del self.sg.cards[i - flag0]
                    flag += 1
                    flag0 += 1
                    if flag == len(a): break
        del self.sg.best_strategy[0]

    def function_changed(self):
        self.warned = 0
        type = self.comboBox.currentIndex()
        if type == 0:
            self.card_hide()
            self.textBrowser.setText("You changed the function to [set numbers, no scores]")
        elif type == 1:
            self.card_hide()
            self.textBrowser.setText("You changed the function to [set cards, no scores]")
        elif type == 2:
            self.card_hide()
            self.textBrowser.setText("You changed the function to [set numbers, with scores]")
        elif type == 3:
            self.card_hide()
            self.textBrowser.setText("You changed the function to [set cards, with scores]")



    

    #def initUI():
        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    '''创建Qt对象'''
    window = MyUi()
    window.show()
    sys.exit(app.exec_())
