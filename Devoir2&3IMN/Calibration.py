import numpy as np
import cv2 as cv
import glob
import MiseEnCorrespondance as MC
import Reconstruction3D
# chessboard size
sizeChessboard = (7, 7)

# image size
imgSize = [1280, 960]
#taille de chaque carré de l'échiquier en mm
sizeEachSquare = 20

# calibration images
ImgLAndR = glob.glob("./images/ChessBoard/LeftAndRight/*")
#tableau qui garde en mémoire les points dans le monde reel et image de toutes les images du chessboard
objPoints = []  # points dans le 3d space (monde reel)
imgPointsL = []  # points dans le repère image de l'image de gauche (Left)
imgPointsR = []  # points dans le repère image de l'image de droite (Right)

#nous avons suivi ce tutoriel pour nous aider avec la calibration
#https://www.youtube.com/watch?v=yKypaVl6qQo&t=1339s&ab_channel=NicolaiNielsen-ComputerVision%26AI
def calibrageCamera():
    SeparationImagesBetweenLeftAndRight()
    # critère pour terminer la calibration
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # initialisation du tableau avec les points de l'objet
    pointObjet = np.zeros((sizeChessboard[0] * sizeChessboard[1], 3), np.float32)
    pointObjet[:, :2] = np.mgrid[0:sizeChessboard[0], 0:sizeChessboard[1]].T.reshape(-1, 2)

    pointObjet = pointObjet * sizeEachSquare

    imagesLeft = sorted(glob.glob('images/ChessBoard/Left/*.png'))
    imagesRight = sorted(glob.glob('images/ChessBoard/Right/*.png'))

    compteur = 0;
    #partie de la fonction sert a trouver les coins du chessboard
    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        #mettre les images de gauche et droite en niveau de gris
        imgL = cv.cvtColor(cv.imread(imgLeft), cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(cv.imread(imgRight), cv.COLOR_BGR2GRAY)

        #trouve les coins du chessboard
        retL, cornersL = cv.findChessboardCorners(imgL, sizeChessboard, None)
        retR, cornersR = cv.findChessboardCorners(imgR, sizeChessboard, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:
            print(compteur)
            objPoints.append(pointObjet)

            cornersL = cv.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
            imgPointsL.append(cornersL)

            cornersR = cv.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)
            imgPointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(imgL, sizeChessboard, cornersL, retL)
            cv.imwrite('images/ChessBoard/Corners/Left/Left' + str(compteur) + '.png', imgL)
            cv.drawChessboardCorners(imgR, sizeChessboard, cornersR, retR)
            cv.imwrite('images/ChessBoard/Corners/Right/Right' + str(compteur) + '.png', imgR)
        compteur += 1

    # calibration de la caméra de gauche
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, imgSize, None, None)
    heightL, widthL, channelsL = cv.imread(imgLeft).shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1,(widthL, heightL))

    # calibration de la caméra de droite
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsL, imgSize, None, None)
    heightR, widthR, channelsR = cv.imread(imgRight).shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1,(widthR, heightR))

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, matRotation, matTranslation, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objPoints, imgPointsL, imgPointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, imgL.shape[::-1],criteria_stereo, flags)

    #calcul les stereo maps pour pouvoir rectifier la rotation des caméra.
    rectifyScale = 1
    rectL, rectR, matProjL, matProjR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, imgL.shape[::-1], matRotation, matTranslation, rectifyScale, (0, 0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, matProjL, imgL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, matProjR, imgR.shape[::-1], cv.CV_16SC2)


    cv_file = cv.FileStorage('Rectification/stereoMap.xml', cv.FILE_STORAGE_WRITE)
    cv_file.write('stereoMapL_x', stereoMapL[0])
    cv_file.write('stereoMapL_y', stereoMapL[1])
    cv_file.write('stereoMapR_x', stereoMapR[0])
    cv_file.write('stereoMapR_y', stereoMapR[1])
    cv_file.release()

    print("Matrice Intrinsèque caméra gauche:")
    print(newCameraMatrixL)
    print("Matrice Intrinsèque caméra droite:")
    print(newCameraMatrixR)
    print("Matrice Fondamentale:")
    print(fundamentalMatrix)
    print("Matrice de translation:")
    print(matTranslation)
    print("Matrice de rotation:")
    print(matRotation)

    return  fundamentalMatrix

def SeparationImagesBetweenLeftAndRight():
    compteur = 0;
    for name in ImgLAndR:
        img = cv.imread(name)
        half = imgSize[0]
        imgL = img[:, :int(half)]
        imgR = img[:, int(half):]
        cv.imwrite('images/ChessBoard/Left/Left' + str(compteur) + '.png', imgL)
        cv.imwrite('images/ChessBoard/Right/Right' + str(compteur) + '.png', imgR)
        compteur += 1


if __name__ == "__main__":
    matriceFond = calibrageCamera()
    imgG, imgD = MC.MiseEnCorrespondance(matriceFond)
    Reconstruction3D.createDepthMap()