import sys

from PyQt5 import QtGui, QtWidgets

import Ui_main
from game import single_game 

Ui_MainWindow = Ui_main.Ui_MainWindow

class MyUi(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        创建主窗口的Qt对象，即斗地主应用程序
        '''
        self.sg = single_game()
        self.line1Label = []
        QtWidgets.QMainWindow.__init__(self)
        '''主界面对象初始化'''
        Ui_MainWindow.__init__(self)
        '''配置主界面对象'''
        self.setupUi(self)

    def search_begin(self):
        type = self.comboBox.currentIndex()
        if type == 0: # no score; set num
            cnum = self.CardNumBox.value()
            if self.sg.cnum != cnum: # need renew cnum
                self.sg.renew_cards(cnum)
                for i in range(cnum):
                    imagelabel = QtWidgets.QLabel(self.CardWidget)
                    self.line1Label.append(imagelabel)
                    self.line1Label[i].move(22 * i, 0)
                    self.line1Label[i].resize(150, 225)
                    img = QtGui.QPixmap("./cards" + str(self.sg.cards[i].id + '.png')).scaled(self.line1Label[0].width(), self.line1Label[0].height())
                    self.line1Label[i].setPixmap(img)
                self.CardWidget.show()
                    
                    



    

    #def initUI():
        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    '''创建Qt对象'''
    window = MyUi()
    window.show()
    sys.exit(app.exec_())
