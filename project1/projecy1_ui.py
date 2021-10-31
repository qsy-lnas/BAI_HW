import sys

from PyQt5 import QtWidgets

import Ui_main
from game import single_game 

Ui_MainWindow = Ui_main.Ui_MainWindow

class MyUi(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        创建主窗口的Qt对象，即斗地主应用程序
        '''
        sg = single_game()
        QtWidgets.QMainWindow.__init__(self)
        '''主界面对象初始化'''
        Ui_MainWindow.__init__(self)
        '''配置主界面对象'''
        self.setupUi(self)

    def
    

    def initUI():
        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    '''创建Qt对象'''
    window = MyUi()
    window.show()
    sys.exit(app.exec_())
