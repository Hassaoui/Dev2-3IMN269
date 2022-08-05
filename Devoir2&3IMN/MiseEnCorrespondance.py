import numpy as np
import cv2 as cv
import glob

image = "./images/MiseEnCorrespondanceIMG/hey.jpg"
imgSize = [1280, 960]

# Copier de la doc
# Voir https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html
def PointColoreImage(img, tabPoints):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for pt in tabPoints:
        color = tuple(np.random.randint(0,255,3).tolist())
        img = cv.circle(img, tuple(pt), 5, color, -1)
    return img


# Copier de la doc
# Voir https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html
def drawLines(img1, img2, pts1, pts2):
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for indx in range(len(pts1)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        p1 = pts1[indx]
        p2 = pts2[indx]
        img1 = cv.circle(img1, p1, 5, color, -1)
        img2 = cv.circle(img2, p2, 5, color, -1)
        img2 = cv.line(img2, p2, p1, color, 1)
        img1 = cv.line(img1, p1, p2, color, 1)
    return img1, img2

#cette fonction a été créé car lorsqu'on avait un voisinage qui serait hors des limites de l'image il y avait des problème
# donc mainenant si le voisinage à l'aire d'aller trop loin il force le voisinage a arrêter aux limites de l'image
def creeTableauPourVoisinage(img, coordo, sizeX, sizeY):
    xG = 0
    xD = len(img[0])
    yG = 0
    yD = len(img[1])
    if coordo[0] - sizeX > 0 :
        xG = coordo[0] - sizeX
    if coordo[0] + sizeX < len(img[0]):
        xD = coordo[0] + sizeX
    if coordo[1] - sizeY > 0:
        yG = coordo[1] - sizeY
    if coordo[1] + sizeY < len(img[0]):
        yD = coordo[1] + sizeY
    return img[yG:yD, xG:xD]


def SauvegardeImgEnsemble(imgG, imgD, nomFichier):
    res = cv.hconcat([imgG, imgD])
    cv.imwrite("./images/Resultat/" + nomFichier, res)


def trouverPointEntreImages(imgG, imgD, pointsG, pointsD, tailleVoisinage, maxDiffY, maxDiffX, confidenceBound):
    bonPointD = []
    bonPointG = []
    aberrantG = []
    nb = 0
    for pt in pointsG:
        meilleurPointPossible = []
        maxMeilleurPointPossible = []
        #fonction qui utilise ransac pour trouver le point qui "match" celui cherché dans les points
        voisinageSize = creeTableauPourVoisinage(imgG, pt, tailleVoisinage, tailleVoisinage)
        candidat = []
        for point in pointsD:
            diffX = point[0] - pt[0]
            diffY = point[1] - pt[1]
            # si le point regarder est dans le range en x et y il est possiblement un poinr que l'on veut
            if abs(diffX) < maxDiffX and  abs(diffY) < maxDiffY:
                candidat.append(point)
        for candi in candidat:
            voisinageCandidat = creeTableauPourVoisinage(imgG, pt, tailleVoisinage, tailleVoisinage)
            correlation = cv.matchTemplate(voisinageCandidat, voisinageSize, cv.TM_CCORR_NORMED)
            haut = 0
            for x in correlation:
                if max(x) > haut:
                    haut = max(x)
            if haut >= confidenceBound:
                meilleurPointPossible.append(candi)
                maxMeilleurPointPossible.append(haut)
        if len(meilleurPointPossible) > 0:
            meilleurPoint = meilleurPointPossible[0]
            plushaut = maxMeilleurPointPossible[0]
            for p in range(len(meilleurPointPossible)):
                if maxMeilleurPointPossible[p] > plushaut:
                    plushaut = maxMeilleurPointPossible[p]
                    meilleurPoint = meilleurPointPossible[p]
            bonPointG.append(pt)
            bonPointD.append(meilleurPoint)
        else:
            aberrantG.append(pt)
    np.int32(bonPointD)
    np.int32(bonPointG)
    np.int32(aberrantG)

    return bonPointG, bonPointD, aberrantG


# Copier de la doc
# Voir https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html
def darInfiniteLine(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def MiseEnCorrespondance(matriceFond):
    #paramètre utiliser pour différents opérations
    confidenceBound = 0.80
    tailleVoisinage = 80
    dispariteMaxY = 7
    dispariteMaxX = 70

    #séparation image en gauche et droite
    img = cv.imread(image, 0)
    imgG = img[:, :int(imgSize[0])]
    imgD = img[:, int(imgSize[0]):]

    cv_file = cv.FileStorage()
    cv_file.open('Rectification/stereoMap.xml', cv.FileStorage_READ)

    stereoMapG_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapG_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapD_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapD_y = cv_file.getNode('stereoMapR_y').mat()

    imgDNoDistorision = cv.remap(imgD, stereoMapD_x, stereoMapD_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imgGNoDistorision = cv.remap(imgG, stereoMapG_x, stereoMapG_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    SauvegardeImgEnsemble(imgG, imgGNoDistorision, "ImageGaucheAvantEtApresRectification.jpg")
    SauvegardeImgEnsemble(imgD, imgDNoDistorision, "ImageDroiteAvantEtApresRectification.jpg")


    print("sift")
    #fonction qui trouve les points d'inéret dans une image
    sift = cv.SIFT_create()
    pInteretG, x = sift.detectAndCompute(imgGNoDistorision, None)
    pInteretD, y = sift.detectAndCompute(imgDNoDistorision, None)

    pointsG = np.int32([k.pt for k in pInteretG])
    pointsD = np.int32([k.pt for k in pInteretD])

    imgPointInteretG = PointColoreImage(imgGNoDistorision, pointsG)

    #mise en correspondance une fois les points d'intéret trouvé
    print("Cherche points")
    bonPointG, bonPointD, aberrantG = trouverPointEntreImages(imgGNoDistorision, imgDNoDistorision, pointsG, pointsD, tailleVoisinage, dispariteMaxY, dispariteMaxX, confidenceBound)

    imgPAberant = PointColoreImage(imgGNoDistorision, aberrantG)
    SauvegardeImgEnsemble(imgPointInteretG, imgPAberant, "BonPoint_PointsAberant_08_7_70.jpg")

    MCgauche, MCdroite = drawLines(imgGNoDistorision, imgDNoDistorision, bonPointG, bonPointD)

    SauvegardeImgEnsemble(MCgauche, MCdroite, "Correlation_Droite_Gauche_08_7_70.jpg")

    #vérification mise en correspondance
    pointsG = np.array(bonPointG)
    pointsD = np.array(bonPointD)
    pt1 = np.reshape(pointsG, (1, len(pointsG), 2))
    pt2 = np.reshape(pointsD, (1, len(pointsG), 2))

    p1, p2 = cv.correctMatches(matriceFond, pt1, pt2)

    lines = cv.computeCorrespondEpilines(pointsG.reshape(-1, 1, 2), 2, matriceFond)
    lines = lines.reshape(-1, 3)
    resG, resD = darInfiniteLine(imgGNoDistorision, imgDNoDistorision, lines, pointsG, bonPointD)
    SauvegardeImgEnsemble(resG, resD, "droiteepipolaire.jpg")

    return imgG, imgD


