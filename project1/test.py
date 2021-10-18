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
b = np.array([0, 1, 2, 3])
a = [[1, 1, 1]]
a.append([2] * b[2])
print(a)
if 1 and 0:
    print(1)
elif 1:
    print(2)
else :
    print(3)