import pypylon.pylon as py
from slmPy import slmpy
import numpy as np
import time


slm = slmpy.SLMdisplay()
resX, resY = slm.getSize()

A = np.zeros((resY, resX), dtype=np.uint8)
A[:, resX//2:] = 100
slm.updateArray(A)  # update slm

meas = np.zeros((256, 1000))


for val in range(256):
    A[:, resX // 2:] = val
    slm.updateArray(A)
    time.sleep(0.2)
    #take photo
    # meas[val] = ...




