import cv2 as cv
import numpy as np


img = "./images/MiseEnCorrespondanceIMG/thing.jpg"
imgSize = [1280, 960]
#ce code a été grandement inspiré de la documentation de openCV
#https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
def createDepthMap():
    image = cv.imread(img, 0)
    imgG = image[:, :int(imgSize[0])]
    imgD = image[:, int(imgSize[0]):]
    minDisparity = 0
    maxDisparity = 16 * 15
    blockSize = 3
    sgbm = cv.StereoSGBM_create(minDisparity = minDisparity,
                                         numDisparities = maxDisparity,
                                         blockSize=blockSize,
                                         P1 = 16 * 3 * blockSize * blockSize,
                                         P2 = 16 * 7 * blockSize * blockSize)

    map = sgbm.compute(imgG, imgD)
    cv.normalize(map, map, alpha = 255, beta = 0, norm_type = cv.NORM_MINMAX)
    map = np.uint8(map)
    map = cv.medianBlur(map, 21)
    SauvegardeImgEnsemble(map, imgD)

def SauvegardeImgEnsemble(imgG, imgD):
    res = cv.hconcat([imgG, imgD])
    cv.imwrite("./images/Resultat/depthMap.png", res)