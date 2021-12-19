import os
import cv2
import numpy as np
import easyocr

# INPUT
folderPath = '/home/xense/HDD/Workspaces/pythonWorkspace/ML/projectML/ocr/'
dataPath = folderPath + 'Data/'
allImages = os.listdir(dataPath)

# initialize the easyocr Reader object
reader = easyocr.Reader(['en'])

for each in allImages:
    plateOnly = cv2.imread(dataPath + each)
    plateOnly = cv2.cvtColor(plateOnly, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        plateOnly, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 59, 15)
    text = reader.readtext(thr, detail=0)
    print(text)

    cv2.imshow('img', thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
