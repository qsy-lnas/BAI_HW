import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
 
import Ui_main
""" c = []
a = range(3)
print(a + a)
b = a + a
b.sort()
c.append(b)
print(c) """
c = [[0, 1, 2, 3, 4], [0, 0, 0, 3], [1, 1, 1, 4], [2, 2, 2, 13]]
del c[-2:]
print(c)
if 1 and 0:
    print(1)
elif 1:
    print(2)
else :
    print(3)