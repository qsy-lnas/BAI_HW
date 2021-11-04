import sys
from types import FrameType
from PyQt5.QtWidgets import QLabel, QWidget, QInputDialog, QPushButton, QApplication, QMessageBox, QTextBrowser
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QCursor
from utils import minStepWoStraight, randCardGenerate, manualCardGenerate
from game import single_game
import time
import numpy as np

class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        QMessageBox.about(self, "提示", "程序需要数秒进行初始化，请等待")
        self.sg = single_game()
        self.textOutput.append('初始化完成')
        self.sg_cardnumber = 17

    def initUI(self):

        self.randomCardBtn = QPushButton('随机发牌', self)
        self.randomCardBtn.resize(self.randomCardBtn.sizeHint())
        self.randomCardBtn.move(50, 50)   
        self.randomCardBtn.clicked.connect(self.inputDialog)

        self.manualCardBtn = QPushButton('手动选牌', self)
        self.manualCardBtn.resize(self.manualCardBtn.sizeHint())
        self.manualCardBtn.move(150, 50)
        self.manualCardBtn.clicked.connect(self.inputDialog)

        self.startBtn = QPushButton('搜索最小步数', self)
        self.startBtn.resize(200, 50)
        self.startBtn.move(50, 100)
        self.startBtn.clicked.connect(self.startGame1)

        self.startBtn = QPushButton('搜索最大分值', self)
        self.startBtn.resize(200, 50)
        self.startBtn.move(50, 175)
        self.startBtn.clicked.connect(self.startGame1)
        
        self.currentCardLabel = QLabel('当前手牌：', self)
        self.currentCardLabel.move(300, 50)
        self.currentCardLabel.setVisible(False)

        self.currentStepLabel = QLabel('当前步骤：', self)
        self.currentStepLabel.move(300, 350)
        self.currentStepLabel.setVisible(True)

        self.textOutput = QTextBrowser(self)
        self.textOutput.move(25, 300)
        self.textOutput.resize(250,500)

        self.line1Label = []
        self.line2Label = []
        self.line3Label = []
        for i in range(54):
            self.imageLabel = QLabel(self)
            self.imageLabel.move(300 + 22 * i, 100)
            self.imageLabel.resize(150, 225)
            self.line1Label.append(self.imageLabel)
        self.setGeometry(300, 300, 300, 200)
        self.resize(1700, 900)
        self.setWindowTitle('欢乐斗地主')
        self.show()

    def inputDialog(self):
        sender = self.sender()
        if sender == self.randomCardBtn:
            text, ok = QInputDialog.getText(self, '随机发牌', '输入发牌数')
            if ok:
                self.randCardGenerateUI(int(text))
                self.sg_cardnumber = int(text)
                self.textOutput.append('请点击开始求取最小步方法')
            else:
                return
        elif sender == self.manualCardBtn:
            self.randCardGenerateUI(54)
            text, ok = QInputDialog.getText(self, '手动选牌', '输入选牌数')
            if ok:
                self.manualCardList = []
                self.manualCardIndex = []
                self.textOutput.append('需要选取%d张牌' % int(text))
                self.sg_cardnumber = int(text)
                pass
            else:
                return

    def startGame1(self):
        if self.manualCardNumber > 0:
            self.textOutput.append('选牌未完毕，不能开始游戏')
            return
        startTime = time.time()
        self.printIndex = 0
        self.bestStep, self.bestRouteStraight, self.bestRouteWoStraight = minStepSearch(self.cardList, self.cardKind)
        endTime = time.time()
        self.textOutput.append('搜索完成')
        self.textOutput.append('搜索时间：%.6f 秒' % (endTime - startTime))
        self.textOutput.append('最小步数：%d步' % self.bestStep)
        self.straightRouteDecoder()
        self.woRouteDecoder()

    def straightRouteDecoder(self):
        for i in range(len(self.bestRouteStraight)):
            currentList = self.cardList[self.bestRouteStraight[-i]]
            print(currentList)

    def woRouteDecoder(self):
        pass

    def manualCardChoose(self):
        cursorPosX = QCursor.pos().x() - self.frameGeometry().left()
        cursorPosY = QCursor.pos().y() - self.frameGeometry().top()
        if cursorPosX in range(self.line1Label[0].x(), self.line1Label[-1].x() + self.line1Label[-1].width()) \
            and cursorPosY in range(self.line1Label[0].y(), self.line1Label[0].y() + self.line1Label[1].height()):
            cardIndex = (cursorPosX - self.line1Label[0].x()) // 22
            if cardIndex > 53:
                cardIndex = 53
            if cardIndex in self.manualCardIndex:
                self.textOutput.append('请勿选择已选牌')
                return
            self.manualCardList.append(self.cardList[cardIndex])
            self.manualCardIndex.append(cardIndex)
            print(self.manualCardList)
            self.manualCardNumber -= 1
            if self.manualCardNumber > 0:
                self.textOutput.append('还需选取%d张牌' % self.manualCardNumber)
                self.line1Label[cardIndex].move(self.line1Label[cardIndex].x(), self.line1Label[cardIndex].y() - 30)
            else:
                self.cardList = np.array(self.manualCardList)
                self.cardList = self.cardList[np.argsort(self.cardList[:, 0])]
                self.refreshCurrentCard()
                self.textOutput.append('选牌完毕，请点击开始求取最小步方法')

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.manualCardNumber > 0:
            self.manualCardChoose()
        return super().mousePressEvent(a0)

    def refreshCurrentCard(self):
        for i in range(54):
            self.line1Label[i].move(300 + 22 * i, 100)
            self.line1Label[i].setPixmap(QtGui.QPixmap(""))
        for idx, card in enumerate(self.cardList):
            cardIndex = card[0] - 3 + 13 * card[1]
            jpg = QtGui.QPixmap('./img/' + str(cardIndex) + '.jpg').scaled(self.line1Label[0].width(), self.line1Label[0].height())
            self.line1Label[idx].setPixmap(jpg)

    def randCardGenerateUI(self, num):
        self.cardList = randCardGenerate(num)
        self.currentCardLabel.setVisible(True)
        self.refreshCurrentCard()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = mainWindow()
    sys.exit(app.exec_())