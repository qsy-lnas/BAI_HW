import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QWidget, QInputDialog, QPushButton, QApplication, QMessageBox, QTextBrowser
import Ui_main
from game import single_game 

Ui_MainWindow = Ui_main.Ui_MainWindow

class MyUi(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        创建主窗口的Qt对象，即斗地主应用程序
        '''
        self.sg = single_game()
        self.totalcard = [0] * 73
        QtWidgets.QMainWindow.__init__(self)
        '''主界面对象初始化'''
        Ui_MainWindow.__init__(self)
        '''配置主界面对象'''
        self.setupUi(self)
        self.card_init()
        self.card_hide()

    def card_init(self):
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
        
    def card_hide(self):
        for i in range(len(self.totalcard)):
            if self.totalcard[i] == 0: continue
            self.totalcard[i].setVisible(False)
        self.PlaycardButton.setVisible(False)
        self.PlaycardButton.setEnabled(False)

    def card_show(self):
        for i in range(len(self.sg.cards)):
            self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
            self.totalcard[self.sg.cards[i].id].raise_()
            self.totalcard[self.sg.cards[i].id].setVisible(True)


    def search_begin(self):
        self.CalpushButton.setEnabled(False)
        type = self.comboBox.currentIndex()
        self.card_hide()
        if type == 0: # no score; set num
            cnum = self.CardNumBox.value()
            self.sg.renew_cards(cnum)
            for i in range(cnum):
                self.totalcard[self.sg.cards[i].id].move(10 + 18 * i, 250)
                self.totalcard[self.sg.cards[i].id].raise_()
                self.totalcard[self.sg.cards[i].id].setVisible(True)



        self.CalpushButton.setEnabled(True)
        self.PlaycardButton.setVisible(True)
        self.PlaycardButton.setEnabled(True)
            
    def play_card(self):
        if len(self.sg.best_strategy) == 0: 
            self.card_hide()

        a = self.sg.best_strategy[0]
        

                    



    

    #def initUI():
        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    '''创建Qt对象'''
    window = MyUi()
    window.show()
    sys.exit(app.exec_())
