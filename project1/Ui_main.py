# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\清华 自96\学习\大三秋\人工智能基础\作业\project1\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(501, 434)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.CardNumBox = QtWidgets.QSpinBox(self.centralwidget)
        self.CardNumBox.setGeometry(QtCore.QRect(380, 240, 71, 31))
        self.CardNumBox.setMinimum(1)
        self.CardNumBox.setMaximum(54)
        self.CardNumBox.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.CardNumBox.setProperty("value", 1)
        self.CardNumBox.setObjectName("CardNumBox")
        self.CardWidget = QtWidgets.QWidget(self.centralwidget)
        self.CardWidget.setGeometry(QtCore.QRect(80, 280, 301, 51))
        self.CardWidget.setObjectName("CardWidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 250, 51, 21))
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(270, 0, 236, 168))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 6, 0, 1, 1)
        self.spinBox_3 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_3.setObjectName("spinBox_3")
        self.gridLayout.addWidget(self.spinBox_3, 1, 2, 1, 1)
        self.spinBox_10 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_10.setObjectName("spinBox_10")
        self.gridLayout.addWidget(self.spinBox_10, 1, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 1, 1, 1)
        self.spinBox_12 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_12.setObjectName("spinBox_12")
        self.gridLayout.addWidget(self.spinBox_12, 5, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 4, 3, 1, 1)
        self.spinBox_8 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_8.setObjectName("spinBox_8")
        self.gridLayout.addWidget(self.spinBox_8, 5, 1, 1, 1)
        self.spinBox_6 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_6.setObjectName("spinBox_6")
        self.gridLayout.addWidget(self.spinBox_6, 3, 2, 1, 1)
        self.spinBox_7 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_7.setObjectName("spinBox_7")
        self.gridLayout.addWidget(self.spinBox_7, 5, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.spinBox_4 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_4.setObjectName("spinBox_4")
        self.gridLayout.addWidget(self.spinBox_4, 3, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 4, 2, 1, 1)
        self.spinBox_9 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_9.setObjectName("spinBox_9")
        self.gridLayout.addWidget(self.spinBox_9, 5, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 4, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 6, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 6, 1, 1, 1)
        self.spinBox_5 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_5.setObjectName("spinBox_5")
        self.gridLayout.addWidget(self.spinBox_5, 3, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 3, 1, 1)
        self.spinBox_11 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_11.setObjectName("spinBox_11")
        self.gridLayout.addWidget(self.spinBox_11, 3, 3, 1, 1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_2.setObjectName("spinBox_2")
        self.gridLayout.addWidget(self.spinBox_2, 1, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 4, 1, 1, 1)
        self.spinBox_13 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_13.setObjectName("spinBox_13")
        self.gridLayout.addWidget(self.spinBox_13, 7, 0, 1, 1)
        self.spinBox_14 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_14.setObjectName("spinBox_14")
        self.gridLayout.addWidget(self.spinBox_14, 7, 1, 1, 1)
        self.spinBox_15 = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_15.setObjectName("spinBox_15")
        self.gridLayout.addWidget(self.spinBox_15, 7, 2, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(0, 0, 121, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "斗地主"))
        self.label.setText(_translate("MainWindow", "牌数"))
        self.label_14.setText(_translate("MainWindow", "2"))
        self.label_7.setText(_translate("MainWindow", "9"))
        self.label_6.setText(_translate("MainWindow", "8"))
        self.label_13.setText(_translate("MainWindow", "A"))
        self.label_2.setText(_translate("MainWindow", "3"))
        self.label_3.setText(_translate("MainWindow", "4"))
        self.label_4.setText(_translate("MainWindow", "5"))
        self.label_5.setText(_translate("MainWindow", "7"))
        self.label_10.setText(_translate("MainWindow", "K"))
        self.label_11.setText(_translate("MainWindow", "6"))
        self.label_8.setText(_translate("MainWindow", "J"))
        self.label_16.setText(_translate("MainWindow", "Red Joker"))
        self.label_15.setText(_translate("MainWindow", "Black Joker"))
        self.label_12.setText(_translate("MainWindow", "10"))
        self.label_9.setText(_translate("MainWindow", "Q"))
        self.comboBox.setItemText(0, _translate("MainWindow", "规定牌数_无分数"))
        self.comboBox.setItemText(1, _translate("MainWindow", "自选手牌_无分数"))
        self.comboBox.setItemText(2, _translate("MainWindow", "规定牌数_有分数"))
        self.comboBox.setItemText(3, _translate("MainWindow", "自选手牌_有分数"))
